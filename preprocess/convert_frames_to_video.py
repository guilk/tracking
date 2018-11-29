import os
import cv2
import numpy as np


def convert_frames_to_video(frames_folder, dst_root, fps):
    files = [f for f in os.listdir(frames_folder) if
             os.path.exists(os.path.join(frames_folder, f))]
    # print files
    img_files = sorted(files)
    video_path = os.path.join(dst_root, 'tracking.mp4')
    # print img_files
    img_path = os.path.join(frames_folder, img_files[0])
    img = cv2.imread(img_path)
    height, width, layers = img.shape
    size = (width, height)
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mpeg'), fps, size)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    for index, img_file in enumerate(img_files):
        print 'Write {}th of {} images'.format(index, len(img_files))
        img_path = os.path.join(frames_folder, img_file)
        img = cv2.imread(img_path)
        out.write(img)
    out.release()


if __name__ == '__main__':
    fps = 30.0

    img_root = '../VIRAT_S_040103_00_000000_000120/results/'
    dst_root = '../VIRAT_S_040103_00_000000_000120/tracking_videos/'

    convert_frames_to_video(img_root, dst_root, fps)