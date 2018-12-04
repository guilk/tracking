#!/usr/bin/env bash
#python deep_sort_app.py \
#    --sequence_dir=./MOT16/test/MOT16-06 \
#    --detection_file=./MOT16_POI_test/MOT16-06.npy \
#    --min_confidence=0.3 \
#    --nn_budget=100 \
#    --display=True

python /tmp/tracking/deep_sort_app.py \
    --sequence_dir=/tmp/VIRAT_S_000204_04_000738_000977.mp4/frames \
    --detection_file=/tmp/VIRAT_S_000204_04_000738_000977.mp4/VIRAT_S_000204_04_000738_000977.npy \
    --output_file=/tmp/tracking_results/VIRAT_S_000204_04_000738_000977.txt \
    --min_confidence=0.85 \
    --nn_budget=5 \
    --display=False


