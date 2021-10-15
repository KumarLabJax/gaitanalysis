import argparse
import imageio
import numpy as np


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('frame')
    parser.add_argument('vids', nargs='+')

    args = parser.parse_args()

    frame_data = imageio.imread(args.frame)
    # print('frame_data.shape:', frame_data.shape)
    # print('frame_data.dtype:', frame_data.dtype)

    best_vid = None
    best_frame = None
    best_diff = None
    for vid_path in args.vids:
        print('## WORKING ON', vid_path, '##')
        with imageio.get_reader(vid_path) as video_reader:
            for frame_index, vid_frame in enumerate(video_reader):
                abs_diff = np.abs(vid_frame - frame_data).mean()
                if best_vid == None or abs_diff < best_diff:
                    best_vid = vid_path
                    best_frame = frame_index
                    best_diff = abs_diff

                    print('======================')
                    print('best_vid:', best_vid)
                    print('best_frame:', best_frame)
                    print('best_diff:', best_diff)


if __name__ == '__main__':
    main()
