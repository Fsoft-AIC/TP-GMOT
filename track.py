import argparse
import numpy as np
from numpy import random
from pathlib import Path

import sys
import os
import cv2

import matplotlib.pyplot as plt

from ultralytics.utils.checks import print_args

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from groundingdino.util.inference import Model
from utils import FilterTools, nms, contains_bbox

import torch

from boxmot.tracker_zoo import create_tracker

GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swinb_cogcoor.pth"

# Init models
device = "cuda:0" if torch.cuda.is_available() else "cpu"

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Define funcs
def delete_by_index(tensor, indices):
    mask = torch.ones(len(tensor), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def process_bboxes(detections, phrases, sub_parts, negative_parts):
    rm_list = []
    for box_id in range(len(detections.xyxy)):
        #Check if detected box is the main object
        if phrases[box_id] in sub_parts:
            rm_list.append(box_id)
            continue
    phrases = np.delete(phrases, rm_list, axis=0)
    detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
    detections.confidence = np.delete(detections.confidence, rm_list, axis=0)

    rm_list = []
    for box_id in range(len(detections.xyxy)):
        if negative_parts != '' and negative_parts in phrases[box_id]:
            rm_list.append(box_id)
            continue

        # Remove overlapped boxes
        cnt = 0
        for id, box in enumerate(detections.xyxy):
            if box_id != id and contains_bbox(detections.xyxy[box_id], box):
                if negative_parts != '' and negative_parts in phrases[id]:
                    cnt = 2
                else:
                    cnt += 1
                if cnt > 1:
                    break
        if cnt > 1:
            rm_list.append(box_id)

    phrases = np.delete(phrases, rm_list, axis=0)
    detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
    detections.confidence = np.delete(detections.confidence, rm_list, axis=0)
    detections.xyxy, detections.confidence = nms(detections.xyxy, detections.confidence, 0.45)

    return detections, phrases

def generate_color_map(num_ids):
    """Generate a color map for given IDs with distinct colors."""
    colormap = plt.get_cmap('jet')
    colors = [colormap(i) for i in np.linspace(0, 1, 13)]
    base_colors = [(int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)) for color in colors]

    color_map = {}

    # Use base colors for the first few IDs
    for i in range(min(num_ids, len(base_colors))):
        color_map[i] = base_colors[i]  # IDs start from 0

    # Repeat base colors for additional IDs
    for i in range(len(base_colors), num_ids):
        color_map[i] = base_colors[i % len(base_colors)]  # Repeat base colors

    return color_map

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3.5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3.5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

@torch.no_grad()
def run(args):

    folder_name = args.source.split('/')[-1].split('.')[-2]
    dest_folder = os.path.join(args.save_dir, folder_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Create text path and clear content in text file
    parent_txt_path = args.save_dir
    if not os.path.exists(parent_txt_path):
        os.makedirs(parent_txt_path)
        
    txt_path = os.path.join(parent_txt_path, folder_name + '.txt')
    open(txt_path, 'w').close()

    # Init Tracker
    tracking_config = \
        ROOT /\
        'boxmot' /\
        opt.tracking_method /\
        'configs' /\
        (opt.tracking_method + '.yaml')

    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        device,
      )

    detections = None
    image = None
    frame_idx = 0

    cap = cv2.VideoCapture(args.source)
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    tools = FilterTools(args.short_mems, args.long_mems)

    color_map = generate_color_map(1000)

    # Construct text prompt
    text_prompt = args.main_object
    if args.sub_part != '':
        text_prompt = f'{text_prompt} has {args.sub_part}'
    if args.negative_part != '':
        text_prompt = f'{text_prompt}. {args.negative_part}'

    while True:
        ret, image = cap.read()
        if not ret:
            break
    
        #Option for tracking on a snippet
        if frame_idx < args.start_frame:
            frame_idx += 1
            continue

        if (frame_idx > args.end_frame) and (args.end_frame != 0):
            break

        # load image
        print('processing frame:', frame_idx)

        # detect objects
        detections, phrases, feature = grounding_dino_model.predict_with_caption(
            image=image, 
            caption=text_prompt, 
            box_threshold=0.2, 
            text_threshold=0.2
        )

        detections, phrases = process_bboxes(detections, phrases, args.sub_part, args.negative_part)
        max_idx = detections.confidence.argmax()

        # Sim score by GDino
        sims, best_sims, cropped_sims, embs = tools.feature_sim_from_gdino(detections.xyxy, feature, max_idx,
                                                                  detections.confidence[max_idx])

        # Adaptive Threshold
        if args.short_mems:
            target_conf = np.mean(detections.confidence) - 1.29*np.std(detections.confidence)
            num_k = sum(map(lambda x : x >= target_conf, detections.confidence)) - 1
            target_sim_1 = torch.mean(torch.sort(sims.detach().clone(), descending=True)[0][1:num_k])

            # Two-level filter
            rm_list = []
            for idx, conf in enumerate(detections.confidence):
                if conf < target_conf:
                    # Level 2 is optional, sometimes it is better with only one level
                    if sims[idx] < target_sim_1:
                    
                        if args.long_mems:
                            target_sim_2 = torch.mean(torch.sort(best_sims.detach().clone(), 
                                                        descending=True)[0][1:num_k])
                            if best_sims[idx] < target_sim_2:
                            
                                if args.cropped_mems:
                                    target_sim_3 = torch.mean(torch.sort(cropped_sims.detach()
                                                        .clone(), descending=True)[0][1:num_k])
                                    if cropped_sims[idx] < target_sim_3:
                                        rm_list.append(idx)
                                else:
                                      rm_list.append(idx)

                        else:
                            rm_list.append(idx)

            # Delete filtered objects
            detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
            detections.confidence = np.delete(detections.confidence, rm_list, axis=0)
            embs = delete_by_index(embs, rm_list)
            sims = delete_by_index(sims, rm_list)
            max_idx = detections.confidence.argmax()

        #Feed data into tracker
        mean_vector = torch.mean(embs, dim=0)
        measures = []
        for idx, emb in enumerate(embs):
            sim = torch.nn.functional.cosine_similarity(mean_vector, emb, dim=-1).cpu()
            measures.append(sim)

        outputs = tracker.update(detections, measures, embs.cpu(), image)

        if len(outputs) > 0:
            for j, output in enumerate(outputs):
    
                bboxes = output[0:4]
                id = output[4]
                conf = output[5]
                cls = output[6]
                sim = output[7]

                if args.save_txt:
                    # to MOT format
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]
                    # Write MOT compliant results to file
                    with open(txt_path, 'a') as f:
                        f.write(('%g,' * 9 + '%g' + '\n') % (frame_idx, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, conf, -1, -1, -1))

                c = int(cls)  # integer class
                id = int(id)  # integer id
                label = (f'{args.main_object} {id} conf:{conf:.2f}')
                plot_one_box(bboxes, image, label=label, color=color_map[id], line_thickness=2)

        img_path = os.path.join(dest_folder, f'{frame_idx:4d}.jpg')
        cv2.imwrite(img_path, image)
        frame_idx += 1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-method', type=str, default='macsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--save-txt', action='store_false', help='save tracking results in a txt file')
    parser.add_argument('--save-dir', type=str, default='./outputs/')
    parser.add_argument('--main-object', type=str, default='')
    parser.add_argument('--sub-part', type=str, default='')
    parser.add_argument('--negative-part', type=str, default='')
    parser.add_argument('--short-mems', type=int, default=3, help='re-filter with best embedding')
    parser.add_argument('--long-mems', type=int, default=9, help='re-filter with best embedding')
    parser.add_argument('--cropped-mems', action='store_true', help='re-filter for occluded objects') #experimental setting, unused
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=0)
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)