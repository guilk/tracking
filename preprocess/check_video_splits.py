import os


if __name__ == '__main__':
    split_root = '../../../datasets/virat/splits/annotation/actev_split/valid/'
    video_root = '../../../datasets/virat/videos/'
    files = os.listdir(split_root)
    files_set = set()
    for file_name in files:
        if file_name.startswith('VIRAT_S') and file_name.endswith('-index.json'):
            files_set.add(file_name.replace('_file-index.json', '.mp4'))
    print len(files_set)
    # for video_name in files_set:
    #     video_path = os.path.join(video_root, video_name)
    #     if not os.path.exists(video_path):
    #         print video_path


