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

def generate_compared_results(video_folder):
    left_root = '../videos/tracking_videos'
    right_root = '../videos/tracking_results'
    video_path = '../videos/{}.avi'.format(video_folder)

    left_folder_path = os.path.join(left_root, video_folder)
    right_folder_path = os.path.join(right_root, video_folder)

    image_names = os.listdir(left_folder_path)
    image_names = [image_name for image_name in image_names
                   if os.path.exists(os.path.join(left_folder_path, image_name)) and (not image_name.startswith('._'))]
    image_names.sort()
    img_path = os.path.join(left_folder_path, image_names[0])
    img = cv2.imread(img_path)
    height, width, layers =img.shape
    size = (width, height / 2)
    fps = 30.0
    out = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, size)

    for index, img_file in enumerate(image_names):
        print 'Write {}th of {} images'.format(index, len(image_names))
        left_img = cv2.imread(os.path.join(left_folder_path, img_file))
        left_img = cv2.resize(left_img, (width / 2, height / 2))
        right_img = cv2.imread(os.path.join(right_folder_path, img_file))
        right_img = cv2.resize(right_img, (width / 2, height / 2))
        img = np.concatenate((left_img, right_img), axis=1)
        out.write(img)
    out.release()


if __name__ == '__main__':
    # fps = 30.0
    #
    # img_root = '../../VIRAT_S_040003_04_000758_001118/results/'
    # dst_root = '../../VIRAT_S_040003_04_000758_001118/'
    # convert_frames_to_video(img_root, dst_root, fps)

    video_folder = 'VIRAT_S_040103_00_000000_000120'
    generate_compared_results(video_folder)