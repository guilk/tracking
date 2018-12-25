import numpy as np
from bisect import bisect


def linear_inter_bbox(tracking_data, frame_gap):
    print tracking_data.shape
    obj_indices = tracking_data[:,1].astype(np.int)
    obj_ids = set(obj_indices.tolist())
    tracking_data_list = tracking_data.tolist()

    for obj_index in obj_ids:
        mask = obj_indices == obj_index
        tracked_frames = tracking_data[mask][:,0].tolist()
        min_frame_idx = int(min(tracked_frames))
        max_frame_idx = int(max(tracked_frames))
        whole_frames = range(min_frame_idx, max_frame_idx + frame_gap, frame_gap)
        missing_frames = list(set(whole_frames).difference(tracked_frames))
        if len(missing_frames) == 0:
            continue
        for missing_frame in missing_frames:
            insert_index = bisect(tracked_frames, missing_frame)
            if insert_index == 0 or insert_index == len(whole_frames):
                continue
            selected_data = tracking_data[mask]
            prev_frame = selected_data[insert_index-1,0]
            next_frame = selected_data[insert_index,0]

            prev_data = selected_data[insert_index-1,2:6]
            next_data = selected_data[insert_index,2:6]

            ratio = 1.0 * (missing_frame - prev_frame) / (next_frame - prev_frame)
            cur_data = prev_data + (next_data - prev_data) * ratio
            cur_data = np.around(cur_data, decimals=2)
            missing_data = [missing_frame, obj_index] + cur_data.tolist() + [1,-1,-1,-1]
            tracking_data_list.append(missing_data)
            # print missing_data
    tracking_data_list = sorted(tracking_data_list, key=lambda x:(x[0], x[1]))
    tracking_data = np.asarray(tracking_data_list)
    return tracking_data
    # print tracking_data.shape

if __name__ == '__main__':
    data_file = './VIRAT_S_000200_00_000100_000171.txt'
    tracking_data = np.loadtxt(data_file, delimiter=',')
    frame_gap = 1

    tracking_data = linear_inter_bbox(tracking_data, frame_gap)
    np.savetxt('./new_text.txt', tracking_data, delimiter=',', fmt='%i,%i,%.2f,%.2f,%.2f,%.2f,%i,%i,%i,%i')
    # list1 = [1, 2, 3, 4, 5, 6]
    # list2 = [4, 5, 6, 7, 8]
    # values = set(list1).difference(list2)
    # print list(values)
    # min_frame_idx = int(tracking_data[:,0].min())
    # max_frame_idx = int(tracking_data[:,0].max())
    #
    # min_obj_id = int(tracking_data[:,1].min())
    # max_obj_id = int(tracking_data[:,1].max())
    #
    # frame_gap = 1
    #
    # tracked_frames = range(min_frame_idx, max_frame_idx+frame_gap, frame_gap)
    # # print tracked_frames
    #
    # print min_frame_idx,max_frame_idx,min_obj_id,max_obj_id
    #
    #
    # print tracking_data.shape
    #
    # a = 20
    # b = [0, 10, 30, 60, 100, 150, 210, 280, 340, 480, 530]
    # print len(b)
    # print(bisect(b, a))
