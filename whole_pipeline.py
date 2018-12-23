import os

if __name__ == '__main__':
    file_list = './preprocess/validation_split.txt'
    video_list = []
    with open(file_list, 'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            video_list.append(line.rstrip('\r\n'))

    # video_list = ['VIRAT_S_000204_04_000738_000977.mp4']
    for index, video_name in enumerate(video_list):
        print 'Process {}th of {} videos'.format(index, len(video_list))
        if os.path.exists(os.path.join('../tracking_results/', video_name.replace('.mp4', '.txt'))):
            continue
        with open('/tmp/video_list.txt', 'wb') as fw:
            fw.write(video_name)
        cmd = 'python /app/obj_detect.py --video_dir /tmp/videos --video_lst_file /tmp/video_list.txt ' \
              '--out_dir /tmp/bbox_infos --get_box_feat --box_feat_path /tmp/bbox_feats --frame_gap 1 ' \
              '--threshold_conf 0.0001'
        os.system(cmd)

        os.makedirs('/tmp/{}/frames'.format(video_name))

        cmd = 'ffmpeg -i /tmp/videos/{} /tmp/{}/frames/%06d.jpg'.format(video_name, video_name)
        os.system(cmd)

        cmd = 'python /tmp/tracking/preprocess/visualize_bbox.py --video_name {}'.format(video_name)
        os.system(cmd)

        cmd = 'python /tmp/tracking/deep_sort_app.py --sequence_dir=/tmp/{}/frames ' \
              '--detection_file=/tmp/{}/{}.npy ' \
              '--output_file=/tmp/tracking_results/{}.txt ' \
              '--min_confidence=0.85 ' \
              '--nn_budget=5 ' \
              '--display=False'.format(video_name, video_name, video_name.split('.')[0], video_name.split('.')[0])
        os.system(cmd)

        cmd = 'rm -rf /tmp/bbox_feats/{}'.format(video_name)
        os.system(cmd)
        cmd = 'rm -rf /tmp/bbox_infos/{}'.format(video_name)
        os.system(cmd)
        cmd = 'rm -rf /tmp/{}'.format(video_name)
        os.system(cmd)
        cmd = 'rm -rf /tmp/video_list.txt'
        os.system(cmd)



