import argparse
import imageio
import itertools
import math
import os
from collections import OrderedDict
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.backends.tkagg as tkagg
# from matplotlib.backends import _backend_tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import skimage.draw as skidraw
import skimage.measure as skimeas
import h5py

from PIL import Image, ImageTk

import gaitinference as ginf


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# # TODO should I use timestamps instead of FRAMES_PER_SECOND?
# FRAMES_PER_SECOND = 30
# CM_PER_PIXEL = 19.5 * 2.54 / 400


class PlotData(object):

    def __init__(
            self,
            x_vals=None, y_vals=None,
            track_intervals=None, step_intervals=None, stride_intervals=None):

        self.x_vals = x_vals
        self.y_vals = y_vals
        self.track_intervals = track_intervals
        self.step_intervals = step_intervals


def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
    # _backend_tk.blit(photo, figure_canvas_agg.get_renderer()._renderer)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo


class LabelingApp(tk.Tk):

    # both_rear_paw_angle_label = 'Both Rear Paw Angle'
    paw_tail_speed_label = 'Both Rear Paws and Base of Tail Speed'
    # lateral_point_dists_label = 'Lateral Point Distances'
    fore_paw_tail_speed_label = 'Both Fore Paws and Base of Tail Speed'

    def __init__(self, video_file_name=None, data_file_name=None):
        tk.Tk.__init__(self)

        self.load_pending = False

        self.image_reader = None
        self.video_name = None
        self.data_file = None
        self.frame_index = 0
        self.skip_frame_count = 1
        self.image_width = 512
        self.image_height = 512
        self.plot_window_size = 100
        self.plot_height = 300
        self.smoothing_window = 5
        self.mouse_speed_thresh = 10
        self.peak_count_thresh = 4

        self.frame_data = None
        self.frame_image = None

        # used for preventing garbage collection
        self._not_garbage = []

        menu_bar = tk.Menu(self)

        # the file menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label='Open Video...',
            command=self._open_video,
            image=self._load_icon('open-video-16.png'),
            compound=tk.LEFT,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label='Exit',
            command=self._close,
        )
        menu_bar.add_cascade(label='File', menu=file_menu)

        # the video navigation menu
        self.vid_ctrl_menu = tk.Menu(menu_bar, tearoff=0)
        self.vid_ctrl_menu.add_command(
            label='Next Frame',
            command=self._next_frame,
            image=self._load_icon('next-16.png'),
            compound=tk.LEFT,
            accelerator='right-arrow',
        )
        self.vid_ctrl_menu.add_command(
            label='Previous Frame',
            command=self._prev_frame,
            image=self._load_icon('previous-16.png'),
            compound=tk.LEFT,
            accelerator='left-arrow',
        )
        self.vid_ctrl_menu.add_command(
            label='Set Frame Skip Count (1)...',
            command=self._update_frame_skip_count,
            image=self._load_icon('skip-count-16.png'),
            compound=tk.LEFT,
            accelerator='ctrl+k',
        )
        menu_bar.add_cascade(label='Video Controls', menu=self.vid_ctrl_menu)

        self.config(menu=menu_bar)

        # the metrics menu
        self.metrics_menu = tk.Menu(menu_bar, tearoff=0)
        self.metric_mode = tk.StringVar()
        self.metric_mode.set(LabelingApp.paw_tail_speed_label)
        self.metrics_menu.add_radiobutton(
            label=LabelingApp.paw_tail_speed_label,
            value=LabelingApp.paw_tail_speed_label,
            variable=self.metric_mode,
            command=self._load_frame_idle,
        )
        # self.metrics_menu.add_radiobutton(
        #     label=LabelingApp.lateral_point_dists_label,
        #     value=LabelingApp.lateral_point_dists_label,
        #     variable=self.metric_mode,
        #     command=self._load_frame_idle,
        # )
        self.metrics_menu.add_radiobutton(
            label=LabelingApp.fore_paw_tail_speed_label,
            value=LabelingApp.fore_paw_tail_speed_label,
            variable=self.metric_mode,
            command=self._load_frame_idle,
        )
        menu_bar.add_cascade(label='Metrics', menu=self.metrics_menu)

        # create a canvas to draw frames on
        self.canvas = tk.Canvas(
            self,
            width=self.image_width,
            height=self.image_height)
        self.canvas.pack()

        # global application bindings
        self.bind_all('<Key-Right>', self._next_frame)
        self.bind_all('<Key-Left>', self._prev_frame)
        self.bind_all('<Control-Right>', self._next_frame_no_skip)
        self.bind_all('<Control-Left>', self._prev_frame_no_skip)

        self.protocol('WM_DELETE_WINDOW', self._close)

        if video_file_name:
            self._open_video(video_file_name)

        if data_file_name:
            self._open_data_file(data_file_name)

    def _load_icon(self, icon_filename):
        icon_path = os.path.join(SCRIPT_DIR, 'icons', icon_filename)
        icon = ImageTk.PhotoImage(Image.open(icon_path))
        self._not_garbage.append(icon)

        return icon

    def _open_video(self, filename=None):

        if self.image_reader is not None:
            self.image_reader.close()
            self.image_reader = None

        if filename is None:
            filename = tk.filedialog.askopenfilename(
                title = 'Select Video',
                filetypes = (
                    ('AVI files', '*.avi'),
                    ('MP4 files', '*.mp4'),
                    ('all files', '*.*'),
                ),
            )

        if filename:
            self.image_reader = imageio.get_reader(filename)

            # the HDF5 filename will be based on the video filename
            filename_root, _ = os.path.splitext(filename)
            self.video_name = os.path.basename(filename_root)

            self._load_frame_idle()

    def _open_data_file(self, filename=None):

        if self.data_file is not None:
            self.data_file.close()
            self.data_file = None

        if filename is None:
            filename = tk.filedialog.askopenfilename(
                title = 'Select Data File',
                filetypes = (
                    ('HDF5 files', '*.h5'),
                    ('all files', '*.*'),
                ),
            )

        if filename:
            self.data_file = h5py.File(filename, 'r')
            group = self._get_data_group()

            ginf_dict = ginf.gait_inference(group, self.smoothing_window)

            self.base_tail_speed = ginf_dict['base_tail_speed']
            self.left_rear_paw_speed = ginf_dict['left_rear_paw_speed']
            self.right_rear_paw_speed = ginf_dict['right_rear_paw_speed']
            self.left_fore_paw_speed = ginf_dict['left_fore_paw_speed']
            self.right_fore_paw_speed = ginf_dict['right_fore_paw_speed']

            self.left_step_intervals = ginf_dict['left_step_intervals']
            self.right_step_intervals = ginf_dict['right_step_intervals']

            self.left_rear_paw_xy = ginf_dict['left_rear_paw_xy']
            self.right_rear_paw_xy = ginf_dict['right_rear_paw_xy']

            self.left_rear_paw_conf = ginf_dict['left_rear_paw_conf']
            self.right_rear_paw_conf = ginf_dict['right_rear_paw_conf']
            self.base_tail_conf = ginf_dict['base_tail_conf']

            self.angular_speed = ginf_dict['angular_speed']

            self.tracks = ginf_dict['tracks']

    def _next_frame(self, *dont_care):
        self._set_frame_index(self.frame_index + self.skip_frame_count)
        self._load_frame_idle()

    def _prev_frame(self, *dont_care):
        self._set_frame_index(self.frame_index - self.skip_frame_count)
        self._load_frame_idle()

    def _next_frame_no_skip(self, *dont_care):
        self._set_frame_index(self.frame_index + 1)
        self._load_frame_idle()

    def _prev_frame_no_skip(self, *dont_care):
        self._set_frame_index(self.frame_index - 1)
        self._load_frame_idle()

    def _set_frame_index(self, frame_index):
        # clamp the requested frame index to valid values before assigning
        if frame_index < 0:
            frame_index = 0
        elif frame_index >= self.image_reader.get_length():
            frame_index = self.image_reader.get_length() - 1
        self.frame_index = frame_index

    def _load_frame_idle(self):
        if not self.load_pending:
            self.load_pending = True
            #self.after_idle(self._load_frame)
            self.after(50, self._load_frame)

    def _load_frame(self):

        self.load_pending = False

        # make sure the video was loaded before we proceed
        if self.image_reader is None:
            tk.messagebox.showinfo(
                'No Video Loaded',
                'You must load a video before you can attempt to skip frames',
                parent=self,
            )
            return

        # load the frame
        self.frame_data = self.image_reader.get_data(self.frame_index)
        frame_image_temp = Image.fromarray(self.frame_data)
        self.frame_image = ImageTk.PhotoImage(frame_image_temp)
        image_width = self.frame_image.width()
        image_height = self.frame_image.height()
        if image_width != self.image_width or image_height != self.image_height:
            self.image_width = image_width
            self.image_height = image_height

            print('width/height:', self.image_width, self.image_height)
            self.canvas.config(
                width=self.image_width,
                height=self.image_height + self.plot_height,
            )
            self.canvas.pack()

        self.canvas.create_image(
            self.image_width // 2,
            self.image_height // 2,
            image=self.frame_image,
        )

        if self.metric_mode.get() == LabelingApp.paw_tail_speed_label:
            self._render_hind_paw_tail_speed()
        # elif self.metric_mode.get() == LabelingApp.lateral_point_dists_label:
        #     self._render_lateral_point_dists()
        elif self.metric_mode.get() == LabelingApp.fore_paw_tail_speed_label:
            self._render_fore_paw_tail_speed()

    def _render_frame_and_overlay(self):
        data_group = self._get_data_group()
        if data_group and self.image_reader is not None:

            frame_count = len(self.base_tail_speed)
            plot_range_start = max(0, self.frame_index - self.plot_window_size // 2)
            plot_range_stop = min(frame_count, self.frame_index + self.plot_window_size // 2)

            # point_dists = ginf.point_distances(
            #     data_group['points'][plot_range_start : plot_range_stop, ...])
            # lat_dist = point_dists['lat_dist']
            # paw_dist = point_dists['paw_dist']

            rrp_xy_pos = data_group['points'][plot_range_start : plot_range_stop, ginf.RIGHT_REAR_PAW_INDEX, :]
            rrp_coords = []
            for x_pos, y_pos in rrp_xy_pos:
                rrp_coords.append(y_pos)
                rrp_coords.append(x_pos)
            self.canvas.create_line(rrp_coords, fill='#1f77b4', width=2)

            lrp_xy_pos = data_group['points'][plot_range_start : plot_range_stop, ginf.LEFT_REAR_PAW_INDEX, :]
            lrp_coords = []
            for x_pos, y_pos in lrp_xy_pos:
                lrp_coords.append(y_pos)
                lrp_coords.append(x_pos)
            self.canvas.create_line(lrp_coords, fill='#ff7f0e', width=2)

            curr_frame_points = data_group['points'][self.frame_index, :, :]
            self.canvas.create_oval(
                curr_frame_points[ginf.NOSE_INDEX, 1] - 2,
                curr_frame_points[ginf.NOSE_INDEX, 0] - 2,
                curr_frame_points[ginf.NOSE_INDEX, 1] + 2,
                curr_frame_points[ginf.NOSE_INDEX, 0] + 2,
                outline='red',
                fill='white',
            )
            self.canvas.create_oval(
                curr_frame_points[ginf.BASE_NECK_INDEX, 1] - 2,
                curr_frame_points[ginf.BASE_NECK_INDEX, 0] - 2,
                curr_frame_points[ginf.BASE_NECK_INDEX, 1] + 2,
                curr_frame_points[ginf.BASE_NECK_INDEX, 0] + 2,
                outline='red',
                fill='white',
            )
            self.canvas.create_oval(
                curr_frame_points[ginf.CENTER_SPINE_INDEX, 1] - 2,
                curr_frame_points[ginf.CENTER_SPINE_INDEX, 0] - 2,
                curr_frame_points[ginf.CENTER_SPINE_INDEX, 1] + 2,
                curr_frame_points[ginf.CENTER_SPINE_INDEX, 0] + 2,
                outline='red',
                fill='white',
            )
            self.canvas.create_oval(
                curr_frame_points[ginf.BASE_TAIL_INDEX, 1] - 2,
                curr_frame_points[ginf.BASE_TAIL_INDEX, 0] - 2,
                curr_frame_points[ginf.BASE_TAIL_INDEX, 1] + 2,
                curr_frame_points[ginf.BASE_TAIL_INDEX, 0] + 2,
                outline='red',
                fill='white',
            )
            self.canvas.create_oval(
                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 1] - 2,
                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 0] - 2,
                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 1] + 2,
                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 0] + 2,
                outline='red',
                fill='white',
            )
            self.canvas.create_oval(
                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 1] - 2,
                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 0] - 2,
                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 1] + 2,
                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 0] + 2,
                outline='red',
                fill='white',
            )
            # self.canvas.create_oval(
            #     curr_frame_points[ginf.RIGHT_FRONT_PAW_INDEX, 1] - 2,
            #     curr_frame_points[ginf.RIGHT_FRONT_PAW_INDEX, 0] - 2,
            #     curr_frame_points[ginf.RIGHT_FRONT_PAW_INDEX, 1] + 2,
            #     curr_frame_points[ginf.RIGHT_FRONT_PAW_INDEX, 0] + 2,
            #     outline='red',
            #     fill='white',
            # )
            # self.canvas.create_oval(
            #     curr_frame_points[ginf.LEFT_FRONT_PAW_INDEX, 1] - 2,
            #     curr_frame_points[ginf.LEFT_FRONT_PAW_INDEX, 0] - 2,
            #     curr_frame_points[ginf.LEFT_FRONT_PAW_INDEX, 1] + 2,
            #     curr_frame_points[ginf.LEFT_FRONT_PAW_INDEX, 0] + 2,
            #     outline='red',
            #     fill='white',
            # )

    # def _render_lateral_point_dists(self):

    #     self._render_frame_and_overlay()

    #     data_group = self._get_data_group()
    #     if data_group and self.image_reader is not None:

    #         frame_count = len(self.base_tail_speed)
    #         plot_range_start = max(0, self.frame_index - self.plot_window_size // 2)
    #         plot_range_stop = min(frame_count, self.frame_index + self.plot_window_size // 2)

    #         point_dists = ginf.point_distances(
    #             data_group['points'][plot_range_start : plot_range_stop, ...])
    #         lat_dist = point_dists['lat_dist']

    #         # Create the figure we want to add to an existing canvas
    #         fig = mpl.figure.Figure(
    #             figsize=(self.image_width / 100, self.plot_height / 100),
    #             dpi=100)

    #         ax = fig.add_subplot(411)
    #         ax.set_xlim([plot_range_start, plot_range_stop])
    #         ax.plot(
    #             np.arange(plot_range_start, plot_range_stop),
    #             lat_dist[0, :],
    #         )
    #         ax.axvline(x=self.frame_index, color='r')
    #         ax.axhline(y=0, color='k')
    #         ax.tick_params(
    #             axis='x',          # changes apply to the x-axis
    #             which='both',      # both major and minor ticks are affected
    #             bottom=False,      # ticks along the bottom edge are off
    #             top=False,         # ticks along the top edge are off
    #             labelbottom=False) # labels along the bottom edge are off

    #         ax = fig.add_subplot(412)
    #         ax.set_xlim([plot_range_start, plot_range_stop])
    #         ax.plot(
    #             np.arange(plot_range_start, plot_range_stop),
    #             lat_dist[1, :],
    #         )
    #         ax.axvline(x=self.frame_index, color='r')
    #         ax.axhline(y=0, color='k')
    #         ax.tick_params(
    #             axis='x',          # changes apply to the x-axis
    #             which='both',      # both major and minor ticks are affected
    #             bottom=False,      # ticks along the bottom edge are off
    #             top=False,         # ticks along the top edge are off
    #             labelbottom=False) # labels along the bottom edge are off

    #         ax = fig.add_subplot(413)
    #         ax.set_xlim([plot_range_start, plot_range_stop])
    #         ax.plot(
    #             np.arange(plot_range_start, plot_range_stop),
    #             lat_dist[2, :],
    #         )
    #         ax.axvline(x=self.frame_index, color='r')
    #         ax.axhline(y=0, color='k')
    #         ax.tick_params(
    #             axis='x',          # changes apply to the x-axis
    #             which='both',      # both major and minor ticks are affected
    #             bottom=False,      # ticks along the bottom edge are off
    #             top=False,         # ticks along the top edge are off
    #             labelbottom=False) # labels along the bottom edge are off

    #         ax = fig.add_subplot(414)
    #         ax.set_xlim([plot_range_start, plot_range_stop])
    #         ax.plot(
    #             np.arange(plot_range_start, plot_range_stop),
    #             lat_dist[3, :],
    #         )
    #         ax.axvline(x=self.frame_index, color='r')
    #         ax.axhline(y=0, color='k')

    #         # Keep this handle alive, or else figure will disappear
    #         ax.set_visible(True)
    #         fig_x, fig_y = 0, self.image_height
    #         self.fig_photo = draw_figure(self.canvas, fig, loc=(fig_x, fig_y))

    #         # # draw a vertical line on the current frame
    #         # ax.axvline(x=self.frame_index, color='r')

    #     # set title
    #     self.title('{}: frame {} of {}'.format(
    #         self.video_name,
    #         self.frame_index,
    #         self.image_reader.get_length(),
    #     ))

    def _render_paw_tail_speed(self, use_hind_paws):

        self._render_frame_and_overlay()

        data_group = self._get_data_group()
        if data_group and self.image_reader is not None:

            frame_count = len(self.base_tail_speed)
            plot_range_start = max(0, self.frame_index - self.plot_window_size // 2)
            plot_range_stop = min(frame_count, self.frame_index + self.plot_window_size // 2)

            # Create the figure we want to add to an existing canvas
            fig = mpl.figure.Figure(
                figsize=(self.image_width / 100, self.plot_height / 100),
                dpi=100)

            ax = fig.add_subplot(211)
            ax.set_xlim([plot_range_start, plot_range_stop])
            ax.plot(
                np.arange(plot_range_start, plot_range_stop),
                self.angular_speed[plot_range_start:plot_range_stop],
            )
            ax.axvline(x=self.frame_index, color='r')
            ax.axhline(y=0, color='k')
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

            ax = fig.add_subplot(212)
            ax.set_xlim([plot_range_start, plot_range_stop])
            if use_hind_paws:
                ax.plot(
                    np.arange(plot_range_start, plot_range_stop),
                    np.transpose(np.stack((
                        self.right_rear_paw_speed[plot_range_start:plot_range_stop],
                        self.left_rear_paw_speed[plot_range_start:plot_range_stop],
                        self.base_tail_speed[plot_range_start:plot_range_stop],
                    )))
                )
            else:
                ax.plot(
                    np.arange(plot_range_start, plot_range_stop),
                    np.transpose(np.stack((
                        self.right_fore_paw_speed[plot_range_start:plot_range_stop],
                        self.left_fore_paw_speed[plot_range_start:plot_range_stop],
                        self.base_tail_speed[plot_range_start:plot_range_stop],
                    )))
                )
            ax.axvline(x=self.frame_index, color='r')

            for track in self.tracks:
                if track.stop_frame_exclu > plot_range_start and track.start_frame < plot_range_stop:
                    # ax.axvspan(xmin=track.start_frame, xmax=track.stop_frame_exclu, color='0.8')

                    for stride in track.good_strides:
                        ax.axvspan(
                            xmin=stride.start_frame,
                            xmax=stride.stop_frame_exclu - 1,
                            color='#c8f4d4')
                        ax.axvline(x=stride.start_frame, color='black')
                        ax.axvline(x=stride.stop_frame_exclu - 1, color='black')

                    for step in track.lrp_steps:
                        ax.axvspan(
                            xmin=step.start_frame, xmax=step.stop_frame_exclu,
                            # ymin=0.51, ymax=0.75,
                            ymin=0.0, ymax=0.05,
                            color='k')

                    for step in track.rrp_steps:
                        ax.axvspan(
                            xmin=step.start_frame, xmax=step.stop_frame_exclu,
                            # ymin=0.25, ymax=0.49,
                            ymin=0.05, ymax=0.1,
                            color='k')

                    for stances in track.lrp_stances:
                        if (stances.start_frame >= plot_range_start
                                and stances.start_frame < plot_range_stop):

                            curr_frame_points = data_group['points'][stances.start_frame, :, :]
                            self.canvas.create_oval(
                                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 1] - 2,
                                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 0] - 2,
                                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 1] + 2,
                                curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 0] + 2,
                                outline='red',
                                fill='white',
                            )
                    for stances in track.rrp_stances:
                        if (stances.start_frame >= plot_range_start
                                and stances.start_frame < plot_range_stop):

                            curr_frame_points = data_group['points'][stances.start_frame, :, :]
                            self.canvas.create_oval(
                                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 1] - 2,
                                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 0] - 2,
                                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 1] + 2,
                                curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 0] + 2,
                                outline='red',
                                fill='white',
                            )

            # Keep this handle alive, or else figure will disappear
            ax.set_visible(True)
            fig_x, fig_y = 0, self.image_height
            self.fig_photo = draw_figure(self.canvas, fig, loc=(fig_x, fig_y))

            # draw a vertical line on the current frame
            ax.axvline(x=self.frame_index, color='r')

        # set title
        # self.title('{}: frame {} of {}'.format(
        #     self.video_name,
        #     self.frame_index,
        #     self.image_reader.get_length(),
        # ))
        self.title('{}: frame {}'.format(
            self.video_name,
            self.frame_index,
        ))

    def _render_fore_paw_tail_speed(self):
        self._render_paw_tail_speed(False)

    def _render_hind_paw_tail_speed(self):
        self._render_paw_tail_speed(True)

    def _get_data_group(self):
        if self.data_file:
            for group in self.data_file.values():
                return group

        return None

    def _update_frame_skip_count(self, *dont_care):
        skip_frame_count = simpledialog.askinteger(
            'Skip Frame Count',
            'How many frames should Next Frame progress?',
            parent=self,
        )

        if self._is_strictly_pos(skip_frame_count):
            self.skip_frame_count = skip_frame_count
            self.vid_ctrl_menu.entryconfigure(
                2,
                label='Set Frame Skip Count ({})...'.format(self.skip_frame_count),
            )
        else:
            tk.messagebox.showwarning(
                'Invalid Value',
                'The skip frame count must be a strictly positive integer',
                parent=self,
            )

    def _is_strictly_pos(self, x):
        return x is not None and math.isfinite(x) and x >= 1

    def _close(self):
        if self.data_file is not None:
            self.data_file.close()
            self.data_file = None
        if self.image_reader is not None:
            self.image_reader.close()
            self.image_reader = None
        self.destroy()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--video-file',
        help='video file to open (you can also just open from the file menu)',
        required=False,
    )
    parser.add_argument(
        '--data-file',
        help='the data to plot',
        required=False,
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        help="the minimum confidence threshold for strides",
        # default=75.0,
        default=0.3,
    )

    args = parser.parse_args()
    ginf.MIN_CONF_THRESH = args.conf_thresh

    app = LabelingApp(args.video_file, args.data_file)
    app.mainloop()


if __name__ == '__main__':
    main()
