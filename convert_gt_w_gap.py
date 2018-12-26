import os
import numpy as np

if __name__ == '__main__':
    # data_file = './VIRAT_S_040000_04_000532_000622.txt'
    data_file = './gt.txt'
    gt_data = np.loadtxt(data_file, delimiter=' ')
    frame_gap = 8

    # min_frame_idx = int(gt_data[:,0].min())
    min_frame_idx = 1
    max_frame_idx = int(gt_data[:,0].max())
    detected_frames = set(range(1, max_frame_idx+frame_gap, frame_gap))

    gt_data_list = gt_data.tolist()
    detected_list = [tracklet_sample for tracklet_sample in gt_data_list
                     if int(tracklet_sample[0]) in detected_frames]

    detected_list = sorted(detected_list, key=lambda x: (x[0], x[1]))

    sampled_gt_data = np.asarray(detected_list)







    # obj_inds = set(gt_data[:,1].tolist())
    # print obj_inds
    # min_frame_idx = int(gt_data[:,1].min())
    # max_frame_idx = int(gt_data[:,1].max())
    # print min_frame_idx, max_frame_idx
    # print gt_data.shape
    np.savetxt('./new_text.txt', sampled_gt_data, delimiter=',', fmt='%i,%i,%.2f,%.2f,%.2f,%.2f,%i,%i,%i,%i')
    # pass