# coding=utf-8
# run script

import sys, os, argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf

import cv2

from models import get_model, resizeImage

import math, time, json, random, operator
import cPickle as pickle
import pycocotools.mask as cocomask
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from deep_sort.utils import create_obj_infos

from utils import Dataset, Summary, get_op_tensor_name

from class_ids import targetClass2id_mergeProp

targetClass2id = targetClass2id_mergeProp

targetid2class = {targetClass2id[one]: one for one in targetClass2id}


def get_args():
    global targetClass2id, targetid2class
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--video_lst_file", default=None, help="video_file_path = os.path.join(video_dir, $line)")

    parser.add_argument("--out_dir", default=None, help="out_dir/$videoname.mp4/%%d.json, start from 0 index")

    parser.add_argument("--frame_gap", default=8, type=int)

    parser.add_argument("--threshold_conf", default=0.0001, type=float)

    # ------ for box feature extraction
    parser.add_argument("--get_box_feat", action="store_true",
                        help="this will generate (num_box, 256, 7, 7) tensor for each frame")
    parser.add_argument("--box_feat_path", default=None,
                        help="output will be out_dir/$videoname.mp4/%%d.npy, start from 0 index")

    # ---- gpu params
    parser.add_argument("--gpu", default=1, type=int, help="number of gpu")
    parser.add_argument("--gpuid_start", default=0, type=int, help="start of gpu id")
    parser.add_argument('--im_batch_size', type=int, default=1)
    parser.add_argument("--use_all_mem", action="store_true")

    # --- for internal visualization
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_path", default=None)
    parser.add_argument("--vis_thres", default=0.7, type=float)

    # ----------- model params
    parser.add_argument("--num_class", type=int, default=17, help="num catagory + 1 background")

    parser.add_argument("--model_path", default="/app/object_detection_model")

    parser.add_argument("--rpn_batch_size", type=int, default=256, help="num roi per image for RPN  training")
    parser.add_argument("--frcnn_batch_size", type=int, default=512, help="num roi per image for fastRCNN training")

    parser.add_argument("--rpn_test_post_nms_topk", type=int, default=2000, help="test post nms, input to fast rcnn")

    parser.add_argument("--max_size", type=int, default=1920, help="num roi per image for RPN and fastRCNN training")
    parser.add_argument("--short_edge_size", type=int, default=1080,
                        help="num roi per image for RPN and fastRCNN training")

    # ----------- tracking params
    parser.add_argument("--get_tracking", action="store_true",
                        help="this will generate tracking results for each frame")
    parser.add_argument("--tracking_dir", default="/tmp",
                        help="output will be out_dir/$videoname.txt, start from 0 index")
    parser.add_argument("--tracking_objs", default="Person,Vehicle",
                        help="Objects to be tracked, default are Person and Vehicle")
    parser.add_argument("--min_confidence", default=0.85, type=float,
                        help="Detection confidence threshold. Disregard all detections "
                             "that have a confidence lower than this value.")
    parser.add_argument("--min_detection_height", default=0, type=int,
                        help="Threshold on the detection bounding box height. Detections "
                             "with height smaller than this value are disregarded")
    parser.add_argument("--nms_max_overlap", default=1.0, type=float,
                        help="Non-maxima suppression threshold: Maximum detection overlap.")
    parser.add_argument("--max_cosine_distance", type=float, default=0.4,
                        help="Gating threshold for cosine distance metric (object appearance).")
    parser.add_argument("--nn_budget", type=int, default=1,
                        help="Maximum size of the appearance descriptors gallery. If None, no budget is enforced.")

    # --------------- exp junk
    parser.add_argument("--obj_v2", action="store_true")
    parser.add_argument("--obj_v3", action="store_true")
    parser.add_argument("--tall_box", action="store_true")
    parser.add_argument("--wide_box", action="store_true")
    parser.add_argument("--wide_v2", action="store_true")
    parser.add_argument("--add_act", action="store_true", help="add activitiy model")
    parser.add_argument("--finer_resolution", action="store_true", help="fpn use finer resolution conv")
    parser.add_argument("--fix_fpn_model", action="store_true",
                        help="for finetuneing a fpn model, whether to fix the lateral and poshoc weights")
    parser.add_argument("--is_cascade_rcnn", action="store_true", help="cascade rcnn on top of fpn")
    parser.add_argument("--add_relation_nn", action="store_true", help="add relation network feature")

    args = parser.parse_args()

    assert args.gpu == args.im_batch_size  # one gpu one image

    args.controller = "/cpu:0"  # parameter server

    targetid2class = targetid2class
    targetClass2id = targetClass2id

    assert len(targetClass2id) == args.num_class

    # ---------------more defautls
    args.freeze = 2
    args.no_obj_detect = False
    args.diva_class = True
    args.add_mask = False
    args.is_fpn = True
    args.new_tensorpack_model = True
    args.mrcnn_head_dim = 256
    args.is_train = False

    args.rpn_min_size = 0
    args.rpn_proposal_nms_thres = 0.7
    args.anchor_strides = (4, 8, 16, 32, 64)

    args.fpn_resolution_requirement = float(args.anchor_strides[3])  # [3] is 32, since we build FPN with r2,3,4,5?

    args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) * args.fpn_resolution_requirement

    args.fpn_num_channel = 256

    args.fpn_frcnn_fc_head_dim = 1024

    # ---- all the mask rcnn config

    args.resnet_num_block = [3, 4, 23, 3]  # resnet 101

    args.anchor_stride = 16  # has to be 16 to match the image feature total stride
    args.anchor_sizes = (32, 64, 128, 256, 512)

    args.anchor_ratios = (0.5, 1, 2)

    args.num_anchors = len(args.anchor_sizes) * len(args.anchor_ratios)
    # iou thres to determine anchor label
    # args.positive_anchor_thres = 0.7
    # args.negative_anchor_thres = 0.3

    # when getting region proposal, avoid getting too large boxes
    args.bbox_decode_clip = np.log(args.max_size / 16.0)

    # fastrcnn
    args.fastrcnn_batch_per_im = args.frcnn_batch_size
    args.fastrcnn_bbox_reg_weights = np.array([10, 10, 5, 5], dtype='float32')

    args.fastrcnn_fg_thres = 0.5  # iou thres
    # args.fastrcnn_fg_ratio = 0.25 # 1:3 -> pos:neg

    # testing
    args.rpn_test_pre_nms_topk = 6000

    args.fastrcnn_nms_iou_thres = 0.5

    args.result_score_thres = args.threshold_conf
    args.result_per_im = 200

    return args


def initialize(config, sess):
    tf.global_variables_initializer().run()
    allvars = tf.global_variables()
    allvars = [var for var in allvars if "global_step" not in var.name]
    restore_vars = allvars
    opts = ["Adam", "beta1_power", "beta2_power", "Adam_1", "Adadelta_1", "Adadelta", "Momentum"]
    restore_vars = [var for var in restore_vars if var.name.split(":")[0].split("/")[-1] not in opts]

    saver = tf.train.Saver(restore_vars, max_to_keep=5)

    load_from = config.model_path
    ckpt = tf.train.get_checkpoint_state(load_from)
    if ckpt and ckpt.model_checkpoint_path:
        loadpath = ckpt.model_checkpoint_path
        saver.restore(sess, loadpath)
    else:
        raise Exception("Model not exists")


# check argument
def check_args(args):
    assert args.video_dir is not None
    assert args.video_lst_file is not None
    assert args.frame_gap >= 1
    if args.get_box_feat:
        assert args.box_feat_path is not None
        if not os.path.exists(args.box_feat_path):
            os.makedirs(args.box_feat_path)
    print "cv2 version %s" % (cv2.__version__)


# not used, not implemented yet
def load_models(config):
    models = []
    for i in xrange(config.gpuid_start, config.gpuid_start + config.gpu):
        models.append(get_model(config, i, controller=config.controller))
    model_final_boxes = [model.final_boxes for model in models]
    # [R]
    model_final_labels = [model.final_labels for model in models]
    model_final_probs = [model.final_probs for model in models]

    return models


if __name__ == "__main__":
    args = get_args()

    check_args(args)

    videolst = [os.path.join(args.video_dir, one.strip()) for one in open(args.video_lst_file).readlines()]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.visualize:
        from viz import draw_boxes

        vis_path = args.vis_path
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

    # 1. load the object detection model

    # models = load_models(args)
    model = get_model(args, args.gpuid_start, controller=args.controller)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    if not args.use_all_mem:
        tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.visible_device_list = "%s" % (
    ",".join(["%s" % i for i in range(args.gpuid_start, args.gpuid_start + args.gpu)]))

    with tf.Session(config=tfconfig) as sess:

        initialize(config=args, sess=sess)

        for videofile in tqdm(videolst, ascii=True):
            # 2. read the video file
            try:
                vcap = cv2.VideoCapture(videofile)
                if not vcap.isOpened():
                    raise Exception("cannot open %s" % videofile)
            except Exception as e:
                raise e

            # initialize tracking module
            if args.get_tracking:
                tracking_objs = args.tracking_objs.split(',')
                metric = metric = nn_matching.NearestNeighborDistanceMetric(
                    "cosine", args.max_cosine_distance, args.nn_budget)
                tracker = Tracker(metric)
                tracking_results = []

            # videoname = os.path.splitext(os.path.basename(videofile))[0]
            videoname = os.path.basename(videofile)
            video_out_path = os.path.join(args.out_dir, videoname)
            if not os.path.exists(video_out_path):
                os.makedirs(video_out_path)
            # for box feature
            if args.get_box_feat:
                feat_out_path = os.path.join(args.box_feat_path, videoname)
                if not os.path.exists(feat_out_path):
                    os.makedirs(feat_out_path)
            # frame_width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
            # frame_height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
            # fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
            # opencv 2
            frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            # opencv 3
            # frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)


            # 3. read frame one by one
            cur_frame = 0
            vis_count = 0
            frame_stack = []
            while cur_frame < frame_count:
                suc, frame = vcap.read()
                if not suc:
                    cur_frame += 1
                    tqdm.write("warning, %s frame of %s failed" % (cur_frame, videoname))
                    continue

                # skip some frame if frame_gap >1
                if cur_frame % args.frame_gap != 0:
                    cur_frame += 1
                    continue

                # 4. run detection on the frame stack if there is enough

                im = frame.astype("float32")

                resized_image = resizeImage(im, args.short_edge_size, args.max_size)

                scale = (resized_image.shape[0] * 1.0 / im.shape[0] + resized_image.shape[1] * 1.0 / im.shape[1]) / 2.0

                feed_dict = model.get_feed_dict_forward(resized_image)

                if args.get_box_feat:
                    sess_input = [model.final_boxes, model.final_labels, model.final_probs, model.fpn_box_feat]

                    final_boxes, final_labels, final_probs, box_feats = sess.run(sess_input, feed_dict=feed_dict)
                    assert len(box_feats) == len(final_boxes)
                    # save the box feature first

                    featfile = os.path.join(feat_out_path, "%d.npy" % (cur_frame))
                    np.save(featfile, box_feats)
                elif args.get_tracking:
                    sess_input = [model.final_boxes, model.final_labels, model.final_probs, model.fpn_box_feat]
                    final_boxes, final_labels, final_probs, box_feats = sess.run(sess_input, feed_dict=feed_dict)
                    assert len(box_feats) == len(final_boxes)
                    # tracking_boxes = final_boxes / scale
                    #
                    # obj_infos = []
                    # for j, (box, prob, label) in enumerate(zip(tracking_boxes, final_probs, final_labels)):
                    #     cat_name = targetid2class[label]
                    #     confidence_socre = float(round(prob,7))
                    #     if cat_name not in tracking_objs or confidence_socre < args.min_confidence:
                    #         continue
                    #     box[2] -= box[0]
                    #     box[3] -= box[1]
                    #     avg_feat = np.mean(np.mean(box_feats[j], axis=1), axis=1)
                    #     norm_feat = avg_feat/np.linalg.norm(avg_feat)
                    #     list_feat = norm_feat.tolist()
                    #     bbox_data = [cur_frame, box[0], box[1], box[2], box[3], confidence_socre] + list_feat
                    #     obj_infos.append(bbox_data)
                    # detections = []
                    # for row in obj_infos:
                    #     bbox, confidence, feature = row[1:5], row[5], row[6:]
                    #     if bbox[3] < args.min_detection_height:
                    #         continue
                    #     detections.append(Detection(bbox, confidence, feature))
                    detections = create_obj_infos(cur_frame, final_boxes, final_probs, final_labels, box_feats,
                                                  targetid2class,tracking_objs, args.min_confidence, args.min_detection_height, scale)
                    # Run non-maxima suppression.
                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    indices = preprocessing.non_max_suppression(
                        boxes, args.nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]

                    # tracking
                    tracker.predict()
                    tracker.update(detections)

                    # Store results
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        bbox = track.to_tlwh()
                        tracking_results.append([
                            cur_frame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                else:
                    sess_input = [model.final_boxes, model.final_labels, model.final_probs]
                    final_boxes, final_labels, final_probs = sess.run(sess_input, feed_dict=feed_dict)
                # print "sess run done"
                # scale back the box to original image size
                final_boxes = final_boxes / scale

                # save as json
                pred = []

                for j, (box, prob, label) in enumerate(zip(final_boxes, final_probs, final_labels)):
                    box[2] -= box[0]
                    box[3] -= box[1]  # produce x,y,w,h output

                    cat_id = label
                    cat_name = targetid2class[cat_id]

                    # encode mask
                    rle = None

                    res = {
                        "category_id": cat_id,
                        "cat_name": cat_name,  # [0-80]
                        "score": float(round(prob, 7)),
                        "bbox": list(map(lambda x: float(round(x, 2)), box)),
                        "segmentation": rle,
                    }

                    pred.append(res)

                # predfile = os.path.join(args.out_dir, "%s_F_%08d.json"%(videoname, cur_frame))
                predfile = os.path.join(video_out_path, "%d.json" % (cur_frame))
                with open(predfile, "w") as f:
                    json.dump(pred, f)

                # for visualization
                if args.visualize:
                    good_ids = [i for i in xrange(len(final_boxes)) if final_probs[i] >= args.vis_thres]
                    final_boxes, final_labels, final_probs = final_boxes[good_ids], final_labels[good_ids], final_probs[
                        good_ids]
                    vis_boxes = np.asarray([[box[0], box[1], box[2] + box[0], box[3] + box[1]] for box in final_boxes])
                    vis_labels = ["%s_%.2f" % (targetid2class[cat_id], prob) for cat_id, prob in
                                  zip(final_labels, final_probs)]
                    newim = draw_boxes(im, vis_boxes, vis_labels, color=np.array([255, 0, 0]), font_scale=0.5,
                                       thickness=2)

                    vis_file = os.path.join(vis_path, "%s_F_%08d.jpg" % (videoname, vis_count))
                    cv2.imwrite(vis_file, newim)
                    vis_count += 1

                cur_frame += 1
            if args.get_tracking:
                output_file = os.path.join(args.tracking_dir, '{}.txt'.format(videoname.split('.')[0]))
                with open(output_file, 'wb') as fw:
                    for row in tracking_results:
                        line = '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (row[0], row[1], row[2], row[3], row[4], row[5])
                        fw.write(line + '\n')