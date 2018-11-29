#!/usr/bin/env bash
#python deep_sort_app.py \
#    --sequence_dir=./MOT16/test/MOT16-06 \
#    --detection_file=./MOT16_POI_test/MOT16-06.npy \
#    --min_confidence=0.3 \
#    --nn_budget=100 \
#    --display=True

python deep_sort_app.py \
    --sequence_dir=../VIRAT_S_040003_04_000758_001118/frames \
    --detection_file=../VIRAT_S_040003_04_000758_001118/VIRAT_S_040003_04_000758_001118.npy \
    --min_confidence=0.85 \
    --nn_budget=5 \
    --display=True


