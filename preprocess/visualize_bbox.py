import os
import json
import numpy as np
import cv2
import random
from sklearn import preprocessing
import argparse

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def generate_colors(n=1):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

def visualize_gt(bbox_infos, frame):
    '''
    :param bbox_infos: [dict] : [u'segmentation', u'category_id', u'score', u'cat_name', u'bbox']
    :param img: current frame
    :return:
    '''
    color_dict = {}
    for bbox_info in bbox_infos:
        if bbox_info['cat_name'] not in ['Person', 'Vehicle', 'Bike'] or bbox_info['score'] < 0.5:
            continue
        if bbox_info['cat_name'] not in color_dict:
            color_dict[bbox_info['cat_name']] = generate_colors()[0]
        x,y,w,h = int(bbox_info['bbox'][0]), int(bbox_info['bbox'][1]), \
                  int(bbox_info['bbox'][2]), int(bbox_info['bbox'][3])

        bbox_color = color_dict[bbox_info['cat_name']]
        cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

def filter_bbox(frame_index, bbox_infos, bbox_feats):
    '''
    :param bbox_infos: [dict] : [u'segmentation', u'category_id', u'score', u'cat_name', u'bbox']
    :param bbox_feats: corresponding bbox feature
    :return:
    '''
    frame_bbox_infos = []
    for index, bbox_info in enumerate(bbox_infos):
        if bbox_info['cat_name'] not in ['Person', 'Vehicle', 'Bike'] or bbox_info['score'] < 0.3:
            continue
        x,y,w,h = int(bbox_info['bbox'][0]), int(bbox_info['bbox'][1]), \
                  int(bbox_info['bbox'][2]), int(bbox_info['bbox'][3])
        avg_feat = np.mean(np.mean(bbox_feats[index], axis=1), axis=1)
        norm_feat = avg_feat / np.linalg.norm(avg_feat)
        list_feat = norm_feat.tolist()
        bbox_data = [frame_index, x, y, w, h, bbox_info['score']] + list_feat
        frame_bbox_infos.append(bbox_data)
    return frame_bbox_infos

def create_detection_file(frame_root, bbox_info_root, bbox_feat_root):
    frames = os.listdir(frame_root)
    frames = [frame for frame in frames
              if os.path.exists(os.path.join(frame_root, frame)) and (not frame.startswith('._'))]
    frames.sort()
    video_bbox_infos = []
    for frame in frames:
        frame_index = int(frame.split('.')[0]) - 1
        print 'Process {}th frame'.format(frame_index)
        bbox_info_path = os.path.join(bbox_info_root, '{}.json'.format(frame_index))
        bbox_feat_path = os.path.join(bbox_feat_root, '{}.npy'.format(frame_index))

        if not os.path.exists(bbox_info_path):
            continue
        bbox_infos = load_json(bbox_info_path)
        bbox_feats = np.load(bbox_feat_path)
        frame_bbox_infos = filter_bbox(frame_index, bbox_infos, bbox_feats)
        video_bbox_infos += frame_bbox_infos

    return video_bbox_infos

def parse_args():
    parser = argparse.ArgumentParser(description="Create input files to tracking")
    parser.add_argument(
        "--video_name", help="the input video name", default=None,
        required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # video_folder = 'VIRAT_S_040003_04_000758_001118'
    video_folder = args.video_name
    dst_root = os.path.join('/tmp/', video_folder)
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    frame_root = '/tmp/{}/frames'.format(video_folder)
    bbox_info_root = '/tmp/bbox_infos/{}/'.format(video_folder)
    bbox_feat_root = '/tmp/bbox_feats/{}/'.format(video_folder)

    video_bbox_infos = create_detection_file(frame_root, bbox_info_root, bbox_feat_root)

    video_bbox_data = np.asarray(video_bbox_infos)
    np.save('/tmp/{}/{}.npy'.format(video_folder, video_folder.split('.')[0]), video_bbox_data)
    #
    # print video_bbox_data.shape
    #
    # frame_index = 1000
    #
    # frame_path = os.path.join(frame_root, '{}.jpg'.format(str(frame_index).zfill(6)))
    # img = cv2.imread(frame_path)
    # bbox_info_path = os.path.join(box_info_root, '{}.json'.format(frame_index))
    # bbox_feat_path = os.path.join(box_feat_root, '{}.npy'.format(frame_index))
    #
    # bbox_info = load_json(bbox_info_path)
    # bbox_feat = np.load(bbox_feat_path)
    #
    # visualize_gt(bbox_info, img)
    # avg_feat = np.mean(np.mean(bbox_feat[0],axis=1), axis=1)
    # norm_feat = avg_feat / np.linalg.norm(avg_feat)
    # print norm_feat.tolist()
    #
    # bbox_feats = os.listdir(box_feat_root)
    # for bbox_feat_file in bbox_feats:
    #     data = np.load(os.path.join(box_feat_root, bbox_feat_file))
    #     print data.shape