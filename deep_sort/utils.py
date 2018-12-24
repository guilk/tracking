import numpy as np
from deep_sort.detection import Detection

def create_obj_infos(cur_frame, final_boxes, final_probs, final_labels, box_feats, targetid2class, tracking_objs, min_confidence, min_detection_height, scale):
    obj_infos = []
    tracking_boxes = final_boxes / scale
    for j, (box, prob, label) in enumerate(zip(tracking_boxes, final_probs, final_labels)):
        cat_name = targetid2class[label]
        confidence_socre = float(round(prob, 7))
        if cat_name not in tracking_objs or confidence_socre < min_confidence:
            continue
        box[2] -= box[0]
        box[3] -= box[1]
        avg_feat = np.mean(np.mean(box_feats[j], axis=1), axis=1)
        norm_feat = avg_feat / np.linalg.norm(avg_feat)
        list_feat = norm_feat.tolist()
        bbox_data = [cur_frame, box[0], box[1], box[2], box[3], confidence_socre] + list_feat
        obj_infos.append(bbox_data)
    detections = []
    for row in obj_infos:
        bbox, confidence, feature = row[1:5], row[5], row[6:]
        if bbox[3] < min_detection_height:
            continue
        detections.append(Detection(bbox, confidence, feature))
    return detections