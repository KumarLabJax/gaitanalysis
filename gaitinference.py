import affine
import argparse
import h5py
import itertools
import math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import scipy.stats

NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

# we will reject any points with a lower confidence score
# MIN_CONF_THRESH = 200
# MIN_CONF_THRESH = 75
MIN_CONF_THRESH = 0.3

# the maximum length of a segment that we tolerate for good frames
# (eg. mid spine to base neck or right rear paw to base of tail)
MAX_SEGMENT_LEN_THRESH = 5

# TODO should I use timestamps instead of FRAMES_PER_SECOND?
FRAMES_PER_SECOND = 30
CM_PER_PIXEL = 19.5 * 2.54 / 400


## Peak detection function borrowed from Eli Billauer
def peakdet(v, delta, x = None):
    """
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """

    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        raise Exception('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        raise Exception('Input argument delta must be a scalar')
    if delta <= 0:
        raise Exception('Input argument delta must be positive')

    mn, mx = np.Inf, np.NINF
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab).reshape(-1, 2), np.array(mintab).reshape(-1, 2)


class FrameInterval(object):
    """
    A simple class for defining frame intervals. The start frame is inclusive and the stop
    frame is exclusive.
    """

    def __init__(self, start_frame, stop_frame_exclu):
        self.start_frame = start_frame
        self.stop_frame_exclu = stop_frame_exclu

    def __len__(self):
        return self.stop_frame_exclu - self.start_frame


class Stride(FrameInterval):
    """
    A stride interval which is deliniated by foot strike events of the left rear paw
    """

    # pylint: disable=unsubscriptable-object

    def __init__(self, start_frame, stop_frame_exclu, speed_cm_per_sec, angular_velocity, cm_per_px=CM_PER_PIXEL):
        super().__init__(start_frame, stop_frame_exclu)
        self.speed_cm_per_sec = speed_cm_per_sec
        self.angular_velocity = angular_velocity
        self.cm_per_px = cm_per_px

        self.rr_paw_strike_frame = None
        self.rr_paw_strike_xy = None
        self.lr_paw_strike1_xy = None
        self.lr_paw_strike2_xy = None
        self.median_position_xy = None
        self.median_position_proportional_xy = None

        self.confidence = 0

        self.lr_duty_factor = 0.0
        self.rr_duty_factor = 0.0

        self.all_frames_ok = True

        self.nose_lateral_displacement = 0
        self.tip_tail_lateral_displacement = 0
        self.base_tail_lateral_displacement = 0

        self.nose_lateral_change = 0
        self.tip_tail_lateral_change = 0
        self.base_tail_lateral_change = 0

        self.nose_lateral_displacement_phase = 0
        self.tip_tail_lateral_displacement_phase = 0
        self.base_tail_lateral_displacement_phase = 0

        self.nose_confidence = 0.0
        self.left_ear_confidence = 0.0
        self.right_ear_confidence = 0.0
        self.base_neck_confidence = 0.0
        self.left_front_paw_confidence = 0.0
        self.right_front_paw_confidence = 0.0
        self.center_spine_confidence = 0.0
        self.left_rear_paw_confidence = 0.0
        self.right_rear_paw_confidence = 0.0
        self.base_tail_confidence = 0.0
        self.mid_tail_confidence = 0.0
        self.tip_tail_confidence = 0.0

    @property
    def has_all_strikes(self):
        """
        determine if this stride has the two left and one right
        strike that make up a well formed stride
        """
        return (
            self.lr_paw_strike1_xy is not None
            and self.lr_paw_strike2_xy is not None
            and self.rr_paw_strike_xy is not None
        )

    @property
    def is_good(self):
        return self.all_frames_ok and self.has_all_strikes

    @property
    def temporal_symmetry(self):
        lr_rr_diff = self.lr_duty_factor - self.rr_duty_factor
        lr_rr_sum = self.lr_duty_factor + self.rr_duty_factor

        return lr_rr_diff / lr_rr_sum

    @property
    def limb_duty_factor(self):
        return (self.lr_duty_factor + self.rr_duty_factor) / 2.0

    @property
    def step_width(self):
        """
        To calculate step width we first form a line between the two
        left foot strikes, we then calculate the shortest distance
        (ie the perpendicular) from the right paw strike to this line
        """

        if not self.has_all_strikes:
            return float('nan')

        else:
            return abs(perp_dist_cm(
                self.lr_paw_strike1_xy,
                self.lr_paw_strike2_xy,
                self.rr_paw_strike_xy,
                cm_per_px=self.cm_per_px,
            ))

    @property
    def step_length1(self):
        if not self.has_all_strikes:
            return float('nan')

        else:
            left_strike1_x = self.lr_paw_strike1_xy[0]
            left_strike1_y = self.lr_paw_strike1_xy[1]
            left_strike2_x = self.lr_paw_strike2_xy[0]
            left_strike2_y = self.lr_paw_strike2_xy[1]
            right_strike_x = self.rr_paw_strike_xy[0]
            right_strike_y = self.rr_paw_strike_xy[1]

            x_diff = left_strike2_x - left_strike1_x
            y_diff = left_strike2_y - left_strike1_y

            if x_diff == 0:
                # special case if line is vertical
                step_len = abs(right_strike_y - left_strike1_y)
            elif y_diff == 0:
                # special case if line is horizontal
                step_len = abs(right_strike_x - left_strike1_x)
            else:
                slope = y_diff / x_diff
                y_intercept = left_strike1_y - slope * left_strike1_x

                x_intersect = (
                    (right_strike_x + slope * right_strike_y - slope * y_intercept)
                    / (slope * slope + 1)
                )
                y_intersect = slope * x_intersect + y_intercept

                step_x_diff = x_intersect - left_strike1_x
                step_y_diff = y_intersect - left_strike1_y

                step_len = math.sqrt(step_x_diff * step_x_diff + step_y_diff * step_y_diff)

            return step_len * self.cm_per_px

    @property
    def step_length2(self):
        if not self.has_all_strikes:
            return float('nan')

        else:
            left_strike1_x = self.lr_paw_strike1_xy[0]
            left_strike1_y = self.lr_paw_strike1_xy[1]
            left_strike2_x = self.lr_paw_strike2_xy[0]
            left_strike2_y = self.lr_paw_strike2_xy[1]
            right_strike_x = self.rr_paw_strike_xy[0]
            right_strike_y = self.rr_paw_strike_xy[1]

            x_diff = left_strike2_x - left_strike1_x
            y_diff = left_strike2_y - left_strike1_y

            if x_diff == 0:
                # special case if line is vertical
                step_len = abs(left_strike2_y - right_strike_y)
            elif y_diff == 0:
                # special case if line is horizontal
                step_len = abs(left_strike2_x - right_strike_x)
            else:
                slope = y_diff / x_diff
                y_intercept = left_strike1_y - slope * left_strike1_x

                x_intersect = (
                    (right_strike_x + slope * right_strike_y - slope * y_intercept)
                    / (slope * slope + 1)
                )
                y_intersect = slope * x_intersect + y_intercept

                step_x_diff = left_strike2_x - x_intersect
                step_y_diff = left_strike2_y - y_intersect

                step_len = math.sqrt(step_x_diff * step_x_diff + step_y_diff * step_y_diff)

            return step_len * self.cm_per_px

    @property
    def stride_length(self):
        left_strike1_x = self.lr_paw_strike1_xy[0]
        left_strike1_y = self.lr_paw_strike1_xy[1]
        left_strike2_x = self.lr_paw_strike2_xy[0]
        left_strike2_y = self.lr_paw_strike2_xy[1]

        x_diff = left_strike2_x - left_strike1_x
        y_diff = left_strike2_y - left_strike1_y

        return math.sqrt(x_diff * x_diff + y_diff * y_diff) * self.cm_per_px


def intervals_overlap(inter1, inter2):
    return (
        inter1.start_frame < inter2.stop_frame_exclu
        and inter1.stop_frame_exclu > inter2.start_frame
    )


def comp_inter_start(inter1, inter2):
    return inter1.start_frame - inter2.start_frame


class Track(FrameInterval):

    def __init__(self, start_frame, stop_frame_exclu):
        super().__init__(start_frame, stop_frame_exclu)

        self.lrp_steps = []
        self.rrp_steps = []
        self.strides = []

        self.confidence = 0

    def _stances(self, steps):
        prev_step_stop_exclu = self.start_frame

        for step in steps:
            if step.start_frame > prev_step_stop_exclu:
                yield FrameInterval(prev_step_stop_exclu, step.start_frame)
                prev_step_stop_exclu = step.start_frame

        if prev_step_stop_exclu < self.stop_frame_exclu:
            yield FrameInterval(prev_step_stop_exclu, self.stop_frame_exclu)

    @property
    def lrp_stances(self):
        return self._stances(self.lrp_steps)

    @property
    def rrp_stances(self):
        return self._stances(self.rrp_steps)

    @property
    def inner_strides(self):
        return self.strides[1:-1]

    @property
    def good_strides(self):
        return [s for s in self.inner_strides if s.is_good]


def stepdet(paw_speeds, base_tail_speeds, peakdelta=10, approx_still=15):
    """
    generator which detects step events for a single paw
    """

    speed_maxs, speed_mins = peakdet(paw_speeds, peakdelta)
    speed_maxs = speed_maxs[:, 0].astype(np.int32)
    speed_mins = speed_mins[:, 0].astype(np.int32)

    for i, speed_max_frame in enumerate(speed_maxs):

        # print('speed_max_frame:', speed_max_frame)
        toe_off_index = speed_max_frame
        while (toe_off_index > 0
                and paw_speeds[toe_off_index] > approx_still
                and paw_speeds[toe_off_index] > base_tail_speeds[toe_off_index]):
            toe_off_index -= 1

        # we may need to step forward one frame
        if paw_speeds[toe_off_index] <= approx_still and toe_off_index < len(paw_speeds) - 1:
            toe_off_index += 1

        # if we stepped past the previous local min we should adjust the toe-off index
        if i > 0 and i < len(speed_mins):
            prev_speed_min_frame = speed_mins[i - 1]
            if prev_speed_min_frame > toe_off_index:
                toe_off_index = prev_speed_min_frame + 1

        strike_index = speed_max_frame
        while (strike_index < len(paw_speeds) - 1
                and paw_speeds[strike_index] > approx_still
                and paw_speeds[strike_index] > base_tail_speeds[strike_index]):
            strike_index += 1

        # if we stepped past the next local min we should adjust the strike index
        if i >= 0 and i < len(speed_mins):
            next_speed_min_frame = speed_mins[i]
            if next_speed_min_frame < strike_index:
                strike_index = next_speed_min_frame

        if strike_index > toe_off_index:
            yield FrameInterval(toe_off_index, strike_index)


def trackdet(base_tail_speeds, speed_thresh=5):
    """
    Detect "track" intervals for the given `base_tail_speeds`
    """
    speed_over_thresh = base_tail_speeds >= speed_thresh
    grp_frame_index = 0
    for grp_key, grp_vals in itertools.groupby(speed_over_thresh):
        grp_count = len(list(grp_vals))
        if grp_key:
            yield Track(grp_frame_index, grp_frame_index + grp_count)

        grp_frame_index += grp_count


def trackstridedet(lr_paw_speeds, rr_paw_speeds, base_tail_speeds, angular_velocities, cm_per_px=CM_PER_PIXEL):
    """
    This function will detect tracks along with the strides that belong to those tracks
    """
    lr_steps = list(stepdet(lr_paw_speeds, base_tail_speeds))
    rr_steps = list(stepdet(rr_paw_speeds, base_tail_speeds))
    tracks = trackdet(base_tail_speeds)

    lr_step_cursor = 0
    rr_step_cursor = 0

    for track in tracks:

        # find steps that belong to the track
        while lr_step_cursor < len(lr_steps):
            curr_lr_step = lr_steps[lr_step_cursor]
            if intervals_overlap(track, curr_lr_step):
                track.lrp_steps.append(curr_lr_step)

            if curr_lr_step.start_frame >= track.stop_frame_exclu:
                break
            else:
                lr_step_cursor += 1

        while rr_step_cursor < len(rr_steps):
            curr_rr_step = rr_steps[rr_step_cursor]
            if intervals_overlap(track, curr_rr_step):
                track.rrp_steps.append(curr_rr_step)

            if curr_rr_step.start_frame >= track.stop_frame_exclu:
                break
            else:
                rr_step_cursor += 1

        # now that steps have been associated with the current track we need
        # to associate step pairs as strides. We start with the left rear paw
        # steps.
        prev_stride_stop = track.start_frame
        for step in track.lrp_steps:

            # strides will start with the finish of the previous stride (or
            # beginning of current track) and finish with the end of the left step
            stride_stop = min(step.stop_frame_exclu, track.stop_frame_exclu - 1)
            if (step.start_frame >= prev_stride_stop
                    and step.start_frame < track.stop_frame_exclu):

                speed_cm_per_sec = np.mean(
                    base_tail_speeds[prev_stride_stop : stride_stop + 1])
                angular_velocity = np.mean(
                    angular_velocities[prev_stride_stop : stride_stop + 1])
                stride = Stride(
                    prev_stride_stop,
                    stride_stop + 1,
                    speed_cm_per_sec,
                    angular_velocity,
                    cm_per_px=cm_per_px)
                track.strides.append(stride)

            prev_stride_stop = stride_stop

        # now we assiciate the right rear paw step with the stride
        for stride in track.strides:
            for step in track.rrp_steps:
                if step.stop_frame_exclu > stride.start_frame:
                    if step.stop_frame_exclu < stride.stop_frame_exclu:
                        stride.rr_paw_strike_frame = step.stop_frame_exclu
                    break

            stride.lr_duty_factor = duty_factor(stride, track.lrp_steps)
            stride.rr_duty_factor = duty_factor(stride, track.rrp_steps)

        yield track


def gait_inference(group, base_tail_smooth, max_duration_frames=None, cm_per_px=CM_PER_PIXEL):

    def limit_frames(xs):
        if max_duration_frames is None:
            return xs
        else:
            return xs[:max_duration_frames, ...]

    base_tail_speed = limit_frames(calc_speed(
        group,
        BASE_TAIL_INDEX,
        smoothing_window=base_tail_smooth,
        cm_per_px=cm_per_px))
    left_fore_paw_speed = limit_frames(calc_speed(group, LEFT_FRONT_PAW_INDEX, cm_per_px=cm_per_px))
    right_fore_paw_speed = limit_frames(calc_speed(group, RIGHT_FRONT_PAW_INDEX, cm_per_px=cm_per_px))
    left_rear_paw_speed = limit_frames(calc_speed(group, LEFT_REAR_PAW_INDEX, cm_per_px=cm_per_px))
    right_rear_paw_speed = limit_frames(calc_speed(group, RIGHT_REAR_PAW_INDEX, cm_per_px=cm_per_px))

    left_step_intervals = list(stepdet(
        left_rear_paw_speed,
        base_tail_speed))
    right_step_intervals = list(stepdet(
        right_rear_paw_speed,
        base_tail_speed))

    angle_deg = limit_frames(calc_angle_deg(group))
    angular_speed = list(calc_angle_speed_deg(angle_deg, smoothing_window=5))

    tracks = list(trackstridedet(
        left_rear_paw_speed,
        right_rear_paw_speed,
        base_tail_speed,
        angular_speed,
        cm_per_px=cm_per_px))

    left_rear_paw_xy = limit_frames(get_xy_pos(group, LEFT_REAR_PAW_INDEX))
    right_rear_paw_xy = limit_frames(get_xy_pos(group, RIGHT_REAR_PAW_INDEX))

    left_rear_paw_conf = limit_frames(get_conf(group, LEFT_REAR_PAW_INDEX))
    right_rear_paw_conf = limit_frames(get_conf(group, RIGHT_REAR_PAW_INDEX))
    base_tail_conf = limit_frames(get_conf(group, BASE_TAIL_INDEX))

    if tracks:
        # TODO should we mark bad strides before the body length calc?
        points = limit_frames(group['points'][:]).astype(np.double)
        ginf.add_median_xy_pos_to_strides(tracks, points)
        del points
        add_xy_pos_to_strides(tracks, left_rear_paw_xy, right_rear_paw_xy)
        add_conf_to_strides(group, tracks)
        add_conf_to_tracks(tracks, left_rear_paw_conf, right_rear_paw_conf, base_tail_conf)
        mark_bad_strides(tracks, group, cm_per_px=cm_per_px)

    return {
        'base_tail_speed': base_tail_speed,
        'left_fore_paw_speed': left_fore_paw_speed,
        'right_fore_paw_speed': right_fore_paw_speed,
        'left_rear_paw_speed': left_rear_paw_speed,
        'right_rear_paw_speed': right_rear_paw_speed,
        'left_step_intervals': left_step_intervals,
        'right_step_intervals': right_step_intervals,
        'angular_speed': angular_speed,
        'tracks': tracks,
        'left_rear_paw_xy': left_rear_paw_xy,
        'right_rear_paw_xy': right_rear_paw_xy,
        'left_rear_paw_conf': left_rear_paw_conf,
        'right_rear_paw_conf': right_rear_paw_conf,
        'base_tail_conf': base_tail_conf,
    }


def add_xy_pos_to_strides(tracks, lr_paw_xy, rr_paw_xy):
    """
    add X/Y paw strike positions to strides
    """
    for track in tracks:
        for stride in track.strides:
            if stride.rr_paw_strike_frame is not None:
                stride.rr_paw_strike_xy = rr_paw_xy[stride.rr_paw_strike_frame, :]
            stride.lr_paw_strike1_xy = lr_paw_xy[stride.start_frame, :]
            stride.lr_paw_strike2_xy = lr_paw_xy[stride.stop_frame_exclu - 1, :]


def add_median_xy_pos_to_strides(tracks, points, open_field_dims=None):
    """
    add median X, Y position for strides.
    """
    for track in tracks:
        for stride in track.strides:
            # we just use the median X, Y position of all points except
            # for mid tail and tip tail
            stride_interval_points = points[stride.start_frame:stride.stop_frame_exclu, :-2, :]
            stride_median_xy = np.flip(np.median(stride_interval_points.reshape(-1, 2), axis=0))
            stride.median_position_xy = stride_median_xy

            if (open_field_dims is not None and open_field_dims['width'] != 0
                    and open_field_dims['height'] != 0):
                min_x = open_field_dims['min_x']
                width = open_field_dims['width']
                min_y = open_field_dims['min_y']
                height = open_field_dims['height']
                norm_x = (stride_median_xy[0] - min_x) / width
                norm_y = (stride_median_xy[1] - min_y) / height
            else:
                norm_x = float('nan')
                norm_y = float('nan')

            stride.median_position_proportional_xy = np.array([norm_x, norm_y], dtype=np.float)


#def add_conf_to_strides(tracks, lr_paw_conf, rr_paw_conf, lf_paw_conf, rf_paw_conf, base_tail_conf, tip_tail_conf, nose_conf):
def add_conf_to_strides(group, tracks):
    """
    For each stride within the given tracks, point confidences will be averaged and
    assigned to the stride's `confidence` value
    """
    nose_confidence = get_conf(group, NOSE_INDEX)
    left_ear_confidence = get_conf(group, LEFT_EAR_INDEX)
    right_ear_confidence = get_conf(group, RIGHT_EAR_INDEX)
    base_neck_confidence = get_conf(group, BASE_NECK_INDEX)
    left_front_paw_confidence = get_conf(group, LEFT_FRONT_PAW_INDEX)
    right_front_paw_confidence = get_conf(group, RIGHT_FRONT_PAW_INDEX)
    center_spine_confidence = get_conf(group, CENTER_SPINE_INDEX)
    left_rear_paw_confidence = get_conf(group, LEFT_REAR_PAW_INDEX)
    right_rear_paw_confidence = get_conf(group, RIGHT_REAR_PAW_INDEX)
    base_tail_confidence = get_conf(group, BASE_TAIL_INDEX)
    mid_tail_confidence = get_conf(group, MID_TAIL_INDEX)
    tip_tail_confidence = get_conf(group, TIP_TAIL_INDEX)

    for track in tracks:
        for stride in track.strides:
            stride_detection_confs = np.concatenate([
                left_rear_paw_confidence[stride.start_frame : stride.stop_frame_exclu],
                right_rear_paw_confidence[stride.start_frame : stride.stop_frame_exclu],
                base_tail_confidence[stride.start_frame : stride.stop_frame_exclu],
            ])
            stride.confidence = np.average(stride_detection_confs)

            stride.nose_confidence = np.average(nose_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.left_ear_confidence = np.average(left_ear_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.right_ear_confidence = np.average(right_ear_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.base_neck_confidence = np.average(base_neck_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.left_front_paw_confidence = np.average(left_front_paw_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.right_front_paw_confidence = np.average(right_front_paw_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.center_spine_confidence = np.average(center_spine_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.left_rear_paw_confidence = np.average(left_rear_paw_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.right_rear_paw_confidence = np.average(right_rear_paw_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.base_tail_confidence = np.average(base_tail_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.mid_tail_confidence = np.average(mid_tail_confidence[stride.start_frame : stride.stop_frame_exclu])
            stride.tip_tail_confidence = np.average(tip_tail_confidence[stride.start_frame : stride.stop_frame_exclu])


def add_lateral_displacement_to_strides(group, tracks, body_len_cm, interpolation='cyclic_spline', cm_per_px=CM_PER_PIXEL):
    for track in tracks:
        for stride in track.good_strides:
            stride_points = group['points'][stride.start_frame:stride.stop_frame_exclu, ...].astype(np.double)
            normalized_points = _normalize_stride_points(stride_points, body_len_cm, cm_per_px=cm_per_px)

            if interpolation in set(['spline', 'cyclic_spline']):
                try:
                    _, point_count, dim_count = normalized_points.shape

                    new_normalized_points = np.empty(
                        [360, point_count, dim_count],
                        dtype=normalized_points.dtype)

                    for point in range(point_count):
                        for dim in range(dim_count):
                            new_normalized_points[:, point, dim] = spline_interpolate(
                                normalized_points[:, point, dim],
                                360,
                                cyclic=(interpolation == 'cyclic_spline'))

                    normalized_points = new_normalized_points
                except TypeError:
                    # this error is caused by https://github.com/scipy/scipy/issues/7589
                    # and there's not much we can do other than ignoring it
                    pass

            lateral_displacement_amplitude = lat_disp_amp(normalized_points)
            stride.nose_lateral_displacement = lateral_displacement_amplitude[NOSE_INDEX]
            stride.base_tail_lateral_displacement = lateral_displacement_amplitude[BASE_TAIL_INDEX]
            stride.tip_tail_lateral_displacement = lateral_displacement_amplitude[TIP_TAIL_INDEX]

            stride.nose_lateral_change = np.abs(
                normalized_points[-1, NOSE_INDEX, 1] - normalized_points[0, NOSE_INDEX, 1])
            stride.tip_tail_lateral_change = np.abs(
                normalized_points[-1, TIP_TAIL_INDEX, 1] - normalized_points[0, TIP_TAIL_INDEX, 1])
            stride.base_tail_lateral_change = np.abs(
                normalized_points[-1, BASE_TAIL_INDEX, 1] - normalized_points[0, BASE_TAIL_INDEX, 1])

            lateral_displacement_phase = lat_disp_phase(normalized_points)
            stride.nose_lateral_displacement_phase = lateral_displacement_phase[NOSE_INDEX]
            stride.base_tail_lateral_displacement_phase = lateral_displacement_phase[BASE_TAIL_INDEX]
            stride.tip_tail_lateral_displacement_phase = lateral_displacement_phase[TIP_TAIL_INDEX]


def add_conf_to_tracks(tracks, lr_paw_conf, rr_paw_conf, base_tail_conf):
    """
    For each track, point confidences will be averaged and
    assigned to the track's `confidence` value
    """
    for track in tracks:
        all_confs = np.concatenate([
            lr_paw_conf[track.start_frame : track.stop_frame_exclu],
            rr_paw_conf[track.start_frame : track.stop_frame_exclu],
            base_tail_conf[track.start_frame : track.stop_frame_exclu],
        ])

        track.confidence = np.average(all_confs)


def mark_bad_strides(tracks, group, cm_per_px=CM_PER_PIXEL):

    all_ok = None

    # check to see that segment lengths are OK
    segment_pairs = [
        (BASE_NECK_INDEX, NOSE_INDEX),
        (BASE_NECK_INDEX, LEFT_EAR_INDEX),
        (BASE_NECK_INDEX, RIGHT_EAR_INDEX),
        (BASE_NECK_INDEX, LEFT_FRONT_PAW_INDEX),
        (BASE_NECK_INDEX, RIGHT_FRONT_PAW_INDEX),
        (BASE_NECK_INDEX, CENTER_SPINE_INDEX),

        (BASE_TAIL_INDEX, CENTER_SPINE_INDEX),
        (BASE_TAIL_INDEX, LEFT_REAR_PAW_INDEX),
        (BASE_TAIL_INDEX, RIGHT_REAR_PAW_INDEX),
        (BASE_TAIL_INDEX, MID_TAIL_INDEX),

        (MID_TAIL_INDEX, TIP_TAIL_INDEX),
    ]
    for pt_index1, pt_index2 in segment_pairs:
        curr_points1 = get_xy_pos(group, pt_index1)
        curr_points2 = get_xy_pos(group, pt_index2)
        curr_len = np.linalg.norm(curr_points1 - curr_points2, axis=1) * cm_per_px
        curr_ok = curr_len <= MAX_SEGMENT_LEN_THRESH

        if all_ok is None:
            all_ok = curr_ok
        else:
            all_ok = np.logical_and(all_ok, curr_ok)

    # check to see that point confidence is OK
    pt_qual_indexes = [
        NOSE_INDEX,
        BASE_NECK_INDEX,
        CENTER_SPINE_INDEX,
        BASE_TAIL_INDEX,
        LEFT_REAR_PAW_INDEX,
        RIGHT_REAR_PAW_INDEX,
        MID_TAIL_INDEX,
        TIP_TAIL_INDEX,
    ]
    for pt_index in pt_qual_indexes:
        curr_ok = get_conf(group, pt_index) >= MIN_CONF_THRESH

        if all_ok is None:
            all_ok = curr_ok
        else:
            all_ok = np.logical_and(all_ok, curr_ok)

    # if any frame in a stride is not OK we say that the whole stride is
    # not OK
    for track in tracks:
        for stride in track.strides:
            stride.all_frames_ok = np.all(all_ok[stride.start_frame : stride.stop_frame_exclu])


def _smooth(vec, smoothing_window):
    if smoothing_window <= 1 or len(vec) == 0:
        return vec.astype(np.float)
    else:
        assert smoothing_window % 2 == 1, 'expected smoothing_window to be odd'
        half_conv_len = smoothing_window // 2
        smooth_tgt = np.concatenate([
            np.full(half_conv_len, vec[0], dtype=vec.dtype),
            vec,
            np.full(half_conv_len, vec[-1], dtype=vec.dtype),
        ])

        smoothing_val = 1 / smoothing_window
        conv_arr = np.full(smoothing_window, smoothing_val)

        return np.convolve(smooth_tgt, conv_arr, mode='valid')


def get_xy_pos(group, point_index, start_index=None, stop_index=None):
    xy_pos = group['points'][start_index : stop_index, point_index, :].astype(np.double)
    return xy_pos


def get_conf(group, point_index, start_index=None, stop_index=None):
    conf = group['confidence'][start_index : stop_index, point_index].astype(np.double)
    return conf


def calc_speed(group, point_index,
               start_index=None, stop_index=None,
               smoothing_window=1, cm_per_px=CM_PER_PIXEL):

    xy_pos = group['points'][start_index : stop_index, point_index, :].astype(np.double)
    xy_pos[:, 0] = _smooth(xy_pos[:, 0], smoothing_window)
    xy_pos[:, 1] = _smooth(xy_pos[:, 1], smoothing_window)

    xy_pos *= cm_per_px
    velocity = np.gradient(xy_pos, axis=0)
    speed_cm_per_sec = np.linalg.norm(velocity, axis=1) * FRAMES_PER_SECOND

    return speed_cm_per_sec


def get_distance_traveled_px(xy_pos, xy_conf, smoothing_window):

    xy_pos = np.array(xy_pos, copy=True, dtype=np.float)

    # only consider frames over our confidence threshold
    good_frames = xy_conf >= MIN_CONF_THRESH

    # erode good frames to create a safety margin
    good_frames = scipy.ndimage.binary_erosion(good_frames, iterations=3)
    num_good_frames = np.sum(good_frames)

    if num_good_frames < 2:
        return num_good_frames, 0

    else:
        xy_pos = xy_pos[good_frames, :]
        xy_pos[:, 0] = _smooth(xy_pos[:, 0], smoothing_window)
        xy_pos[:, 1] = _smooth(xy_pos[:, 1], smoothing_window)

        # find frame to frame distance and sum it all up
        xy_diffs = xy_pos[1:, :] - xy_pos[:-1, :]
        dists = np.linalg.norm(xy_diffs, axis=1)

        return num_good_frames, np.sum(dists)


def get_distance_traveled_cm(xy_pos, xy_conf, smoothing_window, cm_per_px=CM_PER_PIXEL):

    num_good_frames, dist_px = get_distance_traveled_px(xy_pos, xy_conf, smoothing_window)
    return num_good_frames, cm_per_px * dist_px


def median_body_length_cm(group, tracks, cm_per_px=CM_PER_PIXEL):
    body_len_arrs = []
    for track in tracks:
        for stride in track.good_strides:
            base_neck_points = get_xy_pos(group, BASE_NECK_INDEX, stride.start_frame, stride.stop_frame_exclu)
            center_spine_points = get_xy_pos(group, CENTER_SPINE_INDEX, stride.start_frame, stride.stop_frame_exclu)
            base_tail_points = get_xy_pos(group, BASE_TAIL_INDEX, stride.start_frame, stride.stop_frame_exclu)

            segment1_len = np.linalg.norm(base_tail_points - center_spine_points, axis=1) * cm_per_px
            segment2_len = np.linalg.norm(center_spine_points - base_neck_points, axis=1) * cm_per_px
            body_len = segment1_len + segment2_len

            body_len_arrs.append(body_len)

    return np.median(np.concatenate(body_len_arrs))


def calc_angle_deg(group):
    """
    calculates the angle of the orientation of the mouse in degrees
    """

    base_tail_xy = group['points'][:, BASE_TAIL_INDEX].astype(np.double)
    base_neck_xy = group['points'][:, BASE_NECK_INDEX].astype(np.double)
    base_neck_offset_xy = base_neck_xy - base_tail_xy

    angle_rad = np.arctan2(base_neck_offset_xy[:, 1], base_neck_offset_xy[:, 0])

    return angle_rad * (180 / math.pi)


def _gen_calc_angle_speed_deg(angles):
    # we need smooth out the -180-180 breakpoint in order to calculate speed
    # correctly
    for i in range(len(angles) - 1):

        angle1 = angles[i]
        angle1 = angle1 % 360
        if angle1 < 0:
            angle1 += 360

        angle2 = angles[i + 1]
        angle2 = angle2 % 360
        if angle2 < 0:
            angle2 += 360

        diff1 = angle2 - angle1
        abs_diff1 = abs(diff1)
        diff2 = (360 + angle2) - angle1
        abs_diff2 = abs(diff2)
        diff3 = angle2 - (360 + angle1)
        abs_diff3 = abs(diff3)

        if abs_diff1 <= abs_diff2 and abs_diff1 <= abs_diff3:
            yield diff1
        elif abs_diff2 <= abs_diff3:
            yield diff2
        else:
            yield diff3

    yield 0


def calc_angle_speed_deg(angles, smoothing_window=1):
    """
    Calculate angular velocity from the given angles.
    """
    speed_deg = np.array(list(_gen_calc_angle_speed_deg(angles))) * FRAMES_PER_SECOND
    speed_deg = _smooth(speed_deg, smoothing_window)

    return speed_deg


def accum_steps(val_arr, stride, steps):
    """
    Accumulate steps into the value array for a hildebrand plot
    """

    # Note: here we exclude the last stride frame from the accumulated result because that
    #       frame is shared with the subsequent stride and gives visually confusing results
    #       when we plot it

    for step in steps:
        overlap_start = max(step.start_frame, stride.start_frame)
        overlap_stop_exclu = min(step.stop_frame_exclu, stride.stop_frame_exclu - 1)
        if overlap_start <= overlap_stop_exclu:
            start_stride_proportion = (overlap_start - stride.start_frame) / (len(stride) - 1)
            stop_stride_proportion = (overlap_stop_exclu - stride.start_frame) / (len(stride) - 1)

            start_stride_val_index = start_stride_proportion * len(val_arr)
            stop_stride_val_index = stop_stride_proportion * len(val_arr)
            for i in range(int(round(start_stride_val_index)), int(round(stop_stride_val_index))):
                val_arr[i] += 1


def duty_factor(stride, steps):
    """
    duty factor calculates the proportion of time that a paw
    spends in stance (as opposed to swing)
    """
    curr_df = 0
    for step in steps:
        overlap_start = max(step.start_frame, stride.start_frame)
        overlap_stop_exclu = min(step.stop_frame_exclu, stride.stop_frame_exclu)
        if overlap_start <= overlap_stop_exclu:
            overlap_size = overlap_stop_exclu - overlap_start
            curr_df += overlap_size / len(stride)

    return 1.0 - curr_df


class GaitSummary(object):

    """
    gait summary accumulates statistics across many strides
    """

    def __init__(self, speed_bin, angular_velocity_bin, stride_resolution=None, keep_strides=True, cm_per_px=CM_PER_PIXEL):
        self.speed_bin = speed_bin
        self.angular_velocity_bin = angular_velocity_bin

        self._frame_count_accum = 0
        self.stride_resolution = stride_resolution
        if stride_resolution is None:
            self._left_rear_accum = None
            self._right_rear_accum = None

        else:
            self._left_rear_accum = np.zeros(stride_resolution)
            self._right_rear_accum = np.zeros(stride_resolution)

        self.stride_count = 0
        self._speed_cm_per_sec_accum = 0
        self._limb_duty_factor_accum = 0
        self._temporal_symmetry_accum = 0
        self._step_width_accum = 0
        self._step_length1_accum = 0
        self._step_length2_accum = 0
        self._stride_length_accum = 0
        self._angular_velocity_accum = 0
        self._nose_lateral_displacement_accum = 0
        self._base_tail_lateral_displacement_accum = 0
        self._tip_tail_lateral_displacement_accum = 0

        self.normalized_stride_points = []

        self.all_strides = None
        if keep_strides:
            self.all_strides = []

        self.cm_per_px = cm_per_px

    @property
    def left_rear_hildebrand(self):
        if self._left_rear_accum is None:
            return None
        else:
            return self._left_rear_accum / self.stride_count

    @property
    def right_rear_hildebrand(self):
        if self._right_rear_accum is None:
            return None
        else:
            return self._right_rear_accum / self.stride_count

    @property
    def avg_frame_count(self):
        return self._frame_count_accum / self.stride_count

    @property
    def median_frame_count(self):
        return np.median([len(s) for s in self.all_strides])

    @property
    def avg_speed_cm_per_sec(self):
        return self._speed_cm_per_sec_accum / self.stride_count

    @property
    def median_speed_cm_per_sec(self):
        return np.median([s.speed_cm_per_sec for s in self.all_strides])

    @property
    def avg_limb_duty_factor(self):
        return self._limb_duty_factor_accum / self.stride_count

    @property
    def median_limb_duty_factor(self):
        return np.median([(s.lr_duty_factor + s.rr_duty_factor) / 2.0 for s in self.all_strides])

    @property
    def avg_temporal_symmetry(self):
        return self._temporal_symmetry_accum / self.stride_count

    @property
    def median_temporal_symmetry(self):
        return np.median([s.temporal_symmetry for s in self.all_strides])

    @property
    def avg_step_width(self):
        return self._step_width_accum / self.stride_count

    @property
    def median_step_width(self):
        return np.median([s.step_width for s in self.all_strides])

    @property
    def avg_step_length1(self):
        return self._step_length1_accum / self.stride_count

    @property
    def median_step_length1(self):
        return np.median([s.step_length1 for s in self.all_strides])

    @property
    def avg_step_length2(self):
        return self._step_length2_accum / self.stride_count

    @property
    def median_step_length2(self):
        return np.median([s.step_length2 for s in self.all_strides])

    @property
    def avg_stride_length(self):
        return self._stride_length_accum / self.stride_count

    @property
    def median_stride_length(self):
        return np.median([s.stride_length for s in self.all_strides])

    @property
    def avg_angular_velocity(self):
        return self._angular_velocity_accum / self.stride_count

    @property
    def median_angular_velocity(self):
        return np.median([s.angular_velocity for s in self.all_strides])

    @property
    def avg_nose_lateral_displacement(self):
        return self._nose_lateral_displacement_accum / self.stride_count

    @property
    def median_nose_lateral_displacement(self):
        return np.median([s.nose_lateral_displacement for s in self.all_strides])

    @property
    def avg_base_tail_lateral_displacement(self):
        return self._base_tail_lateral_displacement_accum / self.stride_count

    @property
    def median_base_tail_lateral_displacement(self):
        return np.median([s.base_tail_lateral_displacement for s in self.all_strides])

    @property
    def avg_tip_tail_lateral_displacement(self):
        return self._tip_tail_lateral_displacement_accum / self.stride_count

    @property
    def median_tip_tail_lateral_displacement(self):
        return np.median([s.tip_tail_lateral_displacement for s in self.all_strides])



    @property
    def avg_nose_lateral_displacement_phase(self):
        return scipy.stats.circmean(
            [s.nose_lateral_displacement_phase for s in self.all_strides],
            high=1.0,
            low=0.0,
        )

    @property
    def avg_base_tail_lateral_displacement_phase(self):
        return scipy.stats.circmean(
            [s.base_tail_lateral_displacement_phase for s in self.all_strides],
            high=1.0,
            low=0.0,
        )

    @property
    def avg_tip_tail_lateral_displacement_phase(self):
        return scipy.stats.circmean(
            [s.tip_tail_lateral_displacement_phase for s in self.all_strides],
            high=1.0,
            low=0.0,
        )

    def accum_stride(self, group, track, stride, body_length_cm):

        if self.all_strides is not None:
            self.all_strides.append(stride)

        stride_points = group['points'][stride.start_frame:stride.stop_frame_exclu, ...].astype(np.double)

        if self._left_rear_accum is not None:
            accum_steps(self._left_rear_accum, stride, track.lrp_steps)
            accum_steps(self._right_rear_accum, stride, track.rrp_steps)

        # store normalized stride points
        norm_stride_pts = _normalize_stride_points(
            stride_points,
            body_length_cm,
            cm_per_px=self.cm_per_px,
        )
        self.normalized_stride_points.append(norm_stride_pts)

        self._speed_cm_per_sec_accum += stride.speed_cm_per_sec
        self._limb_duty_factor_accum += stride.lr_duty_factor / 2
        self._limb_duty_factor_accum += stride.rr_duty_factor / 2
        self._frame_count_accum += len(stride)
        self._temporal_symmetry_accum += stride.temporal_symmetry
        self._step_width_accum += stride.step_width
        self._step_length1_accum += stride.step_length1
        self._step_length2_accum += stride.step_length2
        self._stride_length_accum += stride.stride_length
        self._angular_velocity_accum += stride.angular_velocity
        self._nose_lateral_displacement_accum += stride.nose_lateral_displacement
        self._base_tail_lateral_displacement_accum += stride.base_tail_lateral_displacement
        self._tip_tail_lateral_displacement_accum += stride.tip_tail_lateral_displacement
        self.stride_count += 1

    def __str__(self):
        fmt_str = (
            'Gait Summary: speed bin={}'
            ', ang. vel. bin={}'
            ', # strides={}'
            ', avg. # frames={}'
            ', avg. limb duty factor={}'
        )
        return fmt_str.format(
            self.speed_bin,
            self.angular_velocity_bin,
            self.stride_count,
            self.avg_frame_count,
            self.avg_limb_duty_factor,
        )


def _normalize_stride_points(stride_points, body_len_cm, cm_per_px=CM_PER_PIXEL):

    """
    A normalization method that uses the stride's displacement
    vector and the animals body length in order to perform
    normalization
    """

    frame_count, point_count, dim_count = stride_points.shape
    assert frame_count >= 2, 'cannot interpolate stride with fewer than two frames'
    assert point_count == 12, 'twelve points expected'
    assert dim_count == 2, '2D points expected'

    fst_center_spine_x, fst_center_spine_y = stride_points[0, CENTER_SPINE_INDEX, :]
    lst_center_spine_x, lst_center_spine_y = stride_points[-1, CENTER_SPINE_INDEX, :]
    x_diff = lst_center_spine_x - fst_center_spine_x
    y_diff = lst_center_spine_y - fst_center_spine_y

    # the displacement vector is used to calculate a stride_theta
    # which we will use later to normalize rotation
    stride_theta = math.atan2(y_diff, x_diff)

    # the step size is used to move the "camera" incrementally each
    # frame by an equal proportion of the stride. This means that
    # translation is not slaved to the center of the animal at each
    # frame.
    x_step_size = x_diff / (frame_count - 1)
    y_step_size = y_diff / (frame_count - 1)

    avg_center_x, avg_center_y = np.mean(stride_points[:, CENTER_SPINE_INDEX, :], axis=0)

    rot_mat = affine.Affine.rotation(-math.degrees(stride_theta))
    scale_mat = affine.Affine.scale(cm_per_px / body_len_cm, cm_per_px / body_len_cm)


    trans_stride_points = np.empty_like(stride_points)
    for frame_index in range(frame_count):
        # calculate the transformation for this frame
        curr_offset_x = (x_diff / 2.0 - avg_center_x) - x_step_size * frame_index
        curr_offset_y = (y_diff / 2.0 - avg_center_y) - y_step_size * frame_index

        translate_mat = affine.Affine.translation(curr_offset_x, curr_offset_y)
        transform_mat = scale_mat * rot_mat * translate_mat

        # apply the transformation to each frame
        for point_index in range(12):
            curr_pt_xy = stride_points[frame_index, point_index, :]
            trans_stride_points[frame_index, point_index, :] = transform_mat * curr_pt_xy

    return trans_stride_points


def perp_dist_cm(line_xy1, line_xy2, pt_xy, cm_per_px=CM_PER_PIXEL):
    """
    Calculate the perpendicular distance from the line defined by the
    two line points to the given point
    """

    line_x1 = line_xy1[0]
    line_y1 = line_xy1[1]
    line_x2 = line_xy2[0]
    line_y2 = line_xy2[1]
    pt_x = pt_xy[0]
    pt_y = pt_xy[1]

    x_diff = line_x2 - line_x1
    y_diff = line_y2 - line_y1

    theta = math.atan2(y_diff, x_diff)
    cos_neg_theta = math.cos(-theta)
    sin_neg_theta = math.sin(-theta)

    # translate the point such that line_xy1 looks like the origin
    pt_x_trans = pt_x - line_x1
    pt_y_trans = pt_y - line_y1

    pt_y_rot = sin_neg_theta * pt_x_trans + cos_neg_theta * pt_y_trans

    pt_dist = pt_y_rot

    return pt_dist * cm_per_px


def get_speed_and_av_bin(
        speed_cm_per_sec,
        angular_velocity,
        speed_bin_size,
        angular_velocity_bin_size):

    # we need the two integer half bin sizes to center
    # angular velocity bins on zero
    if angular_velocity_bin_size is not None:
        half_av_bin_size1 = angular_velocity_bin_size // 2
        half_av_bin_size2 = angular_velocity_bin_size - half_av_bin_size1
    else:
        half_av_bin_size1 = None
        half_av_bin_size2 = None

    speed_bin = None
    if speed_bin_size is not None:
        speed_bin = speed_cm_per_sec // speed_bin_size
        speed_bin *= speed_bin_size

    angular_velocity_bin = None
    if angular_velocity_bin_size is not None:
        angular_velocity_bin = angular_velocity // angular_velocity_bin_size
        angular_velocity_bin *= angular_velocity_bin_size

        # we center the angular velocity bins around zero
        if angular_velocity_bin + half_av_bin_size2 <= angular_velocity:
            angular_velocity_bin += half_av_bin_size2
        else:
            angular_velocity_bin -= half_av_bin_size1

    return (speed_bin, angular_velocity_bin)


def summarize_gait_dict(
        group,
        tracks,
        speed_bin_size,
        angular_velocity_bin_size,
        stride_resolution,
        body_length_cm,
        cm_per_px=CM_PER_PIXEL):

    """
    Create a summary dict by splitting strides up into bins by size and/or
    angular velocity and summarizing each bin. Dictionary keys will be the
    tuple of (bin_speed, bin_angular_velocity)
    """

    all_strides_summary = GaitSummary(None, None, stride_resolution, cm_per_px=cm_per_px)

    summary_dict = dict()
    for track in tracks:
        for stride in track.good_strides:

            summary_key = get_speed_and_av_bin(
                stride.speed_cm_per_sec,
                stride.angular_velocity,
                speed_bin_size,
                angular_velocity_bin_size)
            speed_bin, angular_velocity_bin = summary_key

            if summary_key in summary_dict:
                gait_summary = summary_dict[summary_key]
            else:
                gait_summary = GaitSummary(
                    speed_bin,
                    angular_velocity_bin,
                    stride_resolution,
                    cm_per_px=cm_per_px,
                )
                summary_dict[summary_key] = gait_summary

            gait_summary.accum_stride(group, track, stride, body_length_cm)
            all_strides_summary.accum_stride(group, track, stride, body_length_cm)

    return all_strides_summary, summary_dict


def summarize_gait(
        group,
        tracks,
        speed_bin_size,
        angular_velocity_bin_size,
        stride_resolution,
        body_length_cm,
        cm_per_px=CM_PER_PIXEL):

    """
    Similar to summarize_gait_dict except that summary results are placed in
    a list rather than a dictionary
    """
    gait_dict = summarize_gait_dict(
        group, tracks,
        speed_bin_size, angular_velocity_bin_size,
        stride_resolution, body_length_cm, cm_per_px=cm_per_px)
    return list(gait_dict.values())


def gen_speed_bins(bin_size, start_bin, stop_bin):
    """
    a simple function for generating speed bins
    """
    for i in range(start_bin, stop_bin):
        yield bin_size * i


def gen_angular_velocity_bins(bin_size, bin_count):
    """
    a simple function for generating angular velocity bins
    """
    half_bin_size = bin_size // 2
    half_bin_count = bin_count // 2
    for i in range(-half_bin_count, bin_count - half_bin_count):
        yield bin_size * i - half_bin_size


def gen_speed_and_av_bins(
        speed_bin_size, speed_start_bin, speed_stop_bin,
        av_bin_size, av_bin_count):

    bin_tuples = itertools.chain.from_iterable(
        (
            (speed_bin, av_bin)
            for av_bin
            in gen_angular_velocity_bins(av_bin_size, av_bin_count)
        )
        for speed_bin
        in gen_speed_bins(speed_bin_size, speed_start_bin, speed_stop_bin)
    )

    return list(bin_tuples)


def num_to_label_str(num):
    num = int(num)
    if num < 0:
        return 'neg' + str(-num)
    else:
        return str(num)


def speed_av_bin_tup_to_str(tup):
    curr_speed, curr_av = tup
    return 'speed_{}_ang_vel_{}'.format(
        num_to_label_str(curr_speed),
        num_to_label_str(curr_av))


def restore_stride_points_shape(stride_points_list):
    for stride_points in stride_points_list:
        yield np.reshape(stride_points, (-1, 12, 2))


def lat_disp_amp(normalized_stride_points):
    stride_ys = normalized_stride_points[:, :, 1]
    stride_amp = stride_ys.max(0) - stride_ys.min(0)

    return stride_amp


def lat_disp_phase(normalized_stride_points):
    num_frames, num_points, _ = normalized_stride_points.shape

    if num_frames <= 2:
        return np.full([num_points], float('nan'))
    else:
        stride_ys = normalized_stride_points[:, :, 1]
        stride_argmax = stride_ys.argmax(0)
        stride_phase_offset = stride_argmax / (num_frames - 1)

        return stride_phase_offset


def interpolate_stride_points(stride_points, target_frame_count):

    '''
    linear interpolation of points
    '''

    frame_count, point_count, dim_count = stride_points.shape
    assert frame_count >= 1, 'cannot interpolate empty array'
    assert point_count == 12, 'twelve points expected'
    assert dim_count == 2, '2D points expected'

    tgt_stride_pts = np.empty((target_frame_count, point_count, dim_count), dtype=np.double)
    for point_index in range(point_count):
        for dim_index in range(dim_count):
            tgt_stride_pts[:, point_index, dim_index] = list(interpolate(
                stride_points[:, point_index, dim_index],
                target_frame_count))

    return tgt_stride_pts


def spline_interpolate(arr_1d, target_len, cyclic=False):

    assert len(arr_1d) >= 2, 'array must contain at least two items'

    if cyclic:
        # for the cyclic case we pad the end values with the next item
        # in the cycle for better interpolation
        x_in = np.arange(-1, len(arr_1d) + 1) / (len(arr_1d) - 1)
        arr_1d = np.concatenate([[arr_1d[-2]], arr_1d, [arr_1d[1]]])
        tck = scipy.interpolate.splrep(x_in, arr_1d, s=0)
    else:
        x_in = np.arange(len(arr_1d)) / (len(arr_1d) - 1)
        tck = scipy.interpolate.splrep(x_in, arr_1d, s=0)

    x_out = np.arange(target_len) / (target_len - 1)
    arr_1d_out = scipy.interpolate.splev(x_out, tck, der=0)

    return arr_1d_out


def interpolate(arr_1d, target_len):

    '''
    1D linear interpolation
    '''

    in_len = len(arr_1d)

    for tgt_index in range(target_len):
        interp_index = tgt_index * (in_len - 1) / (target_len - 1)
        if interp_index.is_integer():
            yield arr_1d[int(interp_index)]
        else:
            ceil_index = int(math.ceil(interp_index))
            ceil_val = arr_1d[ceil_index]
            floor_index = int(math.floor(interp_index))
            floor_val = arr_1d[floor_index]

            ceil_proportion = interp_index - floor_index

            yield ceil_val * ceil_proportion + floor_val * (1.0 - ceil_proportion)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-files',
        help='the HDF5 file(s) to use for gait inference',
        required=True,
        nargs='+',
    )
    parser.add_argument(
        '--base-tail-smooth',
        help='The window size that should be used for smoothing base tail speed.'
             ' Base tail speed acts as a surrogate for overall mouse speed'
             ' and this smoothing is used to reduce the effect of jitter on'
             ' our estimate of speed.',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--stride-count-thresh',
        help='Tracks must have at least this number of strides to be included'
             ' in analysis.',
        type=int,
        default=4,
    )

    # TODO add an argument for smothing distance

    args = parser.parse_args()

    left_paw_accum = np.zeros(50)
    right_paw_accum = np.zeros(50)
    stride_count = 0
    for data_file_name in args.data_files:
        data_file = h5py.File(data_file_name, 'r')
        for group in data_file.values():
            base_tail_speed = calc_speed(
                group,
                BASE_TAIL_INDEX,
                smoothing_window=args.base_tail_smooth)
            left_rear_paw_speed = calc_speed(group, LEFT_REAR_PAW_INDEX)
            right_rear_paw_speed = calc_speed(group, RIGHT_REAR_PAW_INDEX)

            angle_deg = calc_angle_deg(group)
            angular_speed = list(calc_angle_speed_deg(angle_deg, smoothing_window=5))

            tracks = list(trackstridedet(
                left_rear_paw_speed,
                right_rear_paw_speed,
                base_tail_speed,
                angular_speed))

            for track in tracks:
                strides = list(track.strides)
                left_steps = track.lrp_steps
                right_steps = track.rrp_steps
                if len(strides) >= args.stride_count_thresh:

                    for stride in strides[1:-1]:
                        accum_steps(left_paw_accum, stride, left_steps)
                        accum_steps(right_paw_accum, stride, right_steps)
                        stride_count += 1

    if stride_count > 0:
        left_paw_accum /= stride_count
        right_paw_accum /= stride_count

    print(left_paw_accum)
    print(right_paw_accum)


if __name__ == '__main__':
    main()
