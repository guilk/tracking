import os
import numpy as np
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing

if __name__ == '__main__':
    nn_budget = 5
    min_confidence = 0.85
    max_cosine_distance = 0.4
    min_detection_height = 0
    nms_max_overlap = 1.0

    metric = metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    bbox = np.asarray([0,0,0,0])
    confidence = 0.5
    feature = np.asarray([0])
    frame_idx = 0

    detections = []
    detections.append(Detection(bbox, confidence, feature))
    detections = [d for d in detections if d.confidence >= min_confidence]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    for tracker in tracker.tracks:
        if not tracker.is_confirmed() or tracker.time_since_update > 1:
            continue
        bbox = tracker.to_tlwh()
        results.append([frame_idx, tracker.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])


    # store results
    output_file = ''
    with open(output_file, 'wb') as fw:
        for row in results:
            line = '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (row[0], row[1], row[2], row[3], row[4], row[5])
            fw.write(line+'\n')