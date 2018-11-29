import os
import cv2
import numpy as np


def convert_frames_to_video(frames_folder, dst_root, fps):
    files = [f for f in os.listdir(frames_folder) if
             os.path.exists(os.path.join(frames_folder, f))]
    # print files
    img_files = sorted(files)
    video_path = os.path.join(dst_root, 'tracking.avi')
    # print img_files
    img_path = os.path.join(frames_folder, img_files[0])
    img = cv2.imread(img_path)
    height, width, layers = img.shape
    size = (width, height)
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mpeg'), fps, size)
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    out = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, size)

    for index, img_file in enumerate(img_files):
        # if index>100:
        #     break
        print 'Write {}th of {} images'.format(index, len(img_files))
        img_path = os.path.join(frames_folder, img_file)
        img = cv2.imread(img_path)
        out.write(img)

    out.release()


if __name__ == '__main__':
    fps = 30.0

    img_root = '../../VIRAT_S_040003_04_000758_001118/results/'
    dst_root = '../../VIRAT_S_040003_04_000758_001118/'
    # dst_root = './'

    convert_frames_to_video(img_root, dst_root, fps)