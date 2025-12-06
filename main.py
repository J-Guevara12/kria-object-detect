#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import json
import shutil
import numpy as np
import cv2
import random
import colorsys
from datetime import datetime
from pynq_dpu import DpuOverlay
from pynq.lib import AxiGPIO

# ***********************************************************************
# Configuration & Arguments
# ***********************************************************************

# Argument Parsing
parser = argparse.ArgumentParser(description="YOLOX DPU Application")
parser.add_argument("--headless", action="store_true", help="Run without display")
args = parser.parse_args()

# Environment Variables Configuration
HEADLESS = os.environ.get("HEADLESS", "false").lower() == "true" or args.headless
LOG_ENTRY_DELAY_MS = int(os.environ.get("LOG_ENTRY_DELAY_MS", 500))
LOG_EXIT_DELAY_MS = int(os.environ.get("LOG_EXIT_DELAY_MS", 500))
CAPTURES_DIR = os.environ.get("CAPTURES_DIR", "./captures")

# Ensure captures directory exists
os.makedirs(CAPTURES_DIR, exist_ok=True)

print(f"Configuration Loaded:")
print(f"  Headless Mode: {HEADLESS}")
print(f"  Entry Delay: {LOG_ENTRY_DELAY_MS}ms")
print(f"  Exit Delay: {LOG_EXIT_DELAY_MS}ms")
print(f"  Captures Dir: {CAPTURES_DIR}")

# ***********************************************************************
# Input file names
# ***********************************************************************
dpu_model = os.path.abspath("dpu.bit")
cnn_xmodel = os.path.join("./", "yolox_nano_pt.xmodel")
labels_file = os.path.join("./img", "coco2017_classes.txt")

# ***********************************************************************
# Prepare the Overlay
# ***********************************************************************
overlay = DpuOverlay(dpu_model)
overlay.load_model(cnn_xmodel)
ol = overlay

# ***********************************************************************
# Class: Scene Monitor (Event Logic)
# ***********************************************************************
class SceneMonitor:
    def __init__(self, entry_delay_ms, exit_delay_ms, save_dir):
        self.entry_delay = entry_delay_ms / 1000.0
        self.exit_delay = exit_delay_ms / 1000.0
        self.save_dir = save_dir
        
        # Structure: { 'label': { 'first_seen': time, 'last_seen': time, 'last_frame': img, 'state': 'candidate'|'active' } }
        self.tracked_objects = {}

    def update(self, detected_labels, current_frame):
        current_time = time.time()
        events = []
        
        # 1. Process Detections
        # We use a set to handle multiple detections of same class as one "presence"
        unique_labels = set(detected_labels)

        for label in unique_labels:
            if label not in self.tracked_objects:
                # New candidate
                self.tracked_objects[label] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_frame': current_frame.copy(), # Cache frame for exit event
                    'state': 'candidate'
                }
            else:
                # Update existing
                obj = self.tracked_objects[label]
                obj['last_seen'] = current_time
                obj['last_frame'] = current_frame.copy() # Update cache
                
                # Check for ENTRY confirmation
                if obj['state'] == 'candidate':
                    if (current_time - obj['first_seen']) >= self.entry_delay:
                        obj['state'] = 'active'
                        events.append(self._create_event("entry", label, current_frame))

        # 2. Process Missing Objects (Exits)
        labels_to_remove = []
        for label, obj in self.tracked_objects.items():
            if label not in unique_labels:
                # Object is missing
                time_since_last = current_time - obj['last_seen']
                
                if time_since_last >= self.exit_delay:
                    # Confirm EXIT only if it was active
                    if obj['state'] == 'active':
                        # Use the cached last_frame for the exit image
                        events.append(self._create_event("exit", label, obj['last_frame']))
                    
                    labels_to_remove.append(label)
        
        # Cleanup removed objects
        for label in labels_to_remove:
            del self.tracked_objects[label]
            
        return events

    def _create_event(self, event_type, label, image):
        timestamp_iso = datetime.now().isoformat()
        filename = f"{label}_{event_type}_{int(time.time())}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        
        event_data = {
            "type": "scene_event",
            "event": event_type, # "entry" or "exit"
            "object_class": label,
            "timestamp": timestamp_iso,
            "image_filename": filename,
            "image_path_local": filepath
        }
        return event_data

# ***********************************************************************
# Utility Functions (Unchanged mostly)
# ***********************************************************************
def preprocess(image, input_size, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_image = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_image = np.ones(input_size, dtype=np.uint8) * 114

    ratio = min(input_size[0] / image.shape[0],
                input_size[1] / image.shape[1])
    resized_image = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_image = resized_image.astype(np.uint8)

    padded_image[:int(image.shape[0] * ratio), :int(image.shape[1] *
                                                    ratio)] = resized_image
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, ratio

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def postprocess(outputs, img_size, ratio, nms_th, nms_score_th, max_width, max_height, p6=False):
    grids = []
    expanded_strides = []
    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    predictions = outputs[0]
    boxes = predictions[:, :4]
    scores = sigmoid(predictions[:, 4:5]) * softmax(predictions[:, 5:])
    
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_th, score_thr=nms_score_th)

    bboxes, scores, class_ids = [], [], []
    if dets is not None:
        bboxes, scores, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]
        for bbox in bboxes:
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(bbox[2], max_width)
            bbox[3] = min(bbox[3], max_height)
    return bboxes, scores, class_ids

def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    dets = None
    if keep:
        dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)
    return dets

def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

class_names = get_class(labels_file)
num_classes = len(class_names)

# Colors for bounding boxes
hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
random.seed(0)
random.shuffle(colors)
random.seed(None)

def draw_bbox(image, bboxes, classes):
    image_h, image_w, _ = image.shape
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(1.8 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    return image

# ***********************************************************************
# Use VART APIs Setup
# ***********************************************************************
dpu = overlay.runner
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()
shapeIn = tuple(inputTensors[0].dims)

shapeOut0 = (tuple(outputTensors[0].dims))
shapeOut1 = (tuple(outputTensors[1].dims))
shapeOut2 = (tuple(outputTensors[2].dims))

input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data = [np.empty(shapeOut0, dtype=np.float32, order="C"),
               np.empty(shapeOut1, dtype=np.float32, order="C"),
               np.empty(shapeOut2, dtype=np.float32, order="C")]
image = input_data[0]

def run(input_image):
    """
    Executes DPU inference on the frame.
    Returns: list of detections (bboxes with scores and class_ids) and detected label names
    """
    input_shape = (416, 416)
    nms_th = 0.45
    nms_score_th = 0.1

    # Pre-processing
    image_height, image_width = input_image.shape[0], input_image.shape[1]
    image_data, ratio = preprocess(input_image, input_shape)

    # Fetch data to DPU and trigger it
    image[0, ...] = image_data.reshape(shapeIn[1:])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    # Decode output from YOLOX-nano
    outputs = np.concatenate([output.reshape(1, -1, output.shape[-1]) for output in output_data], axis=1)
    bboxes, scores, class_ids = postprocess(
        outputs, input_shape, ratio, nms_th, nms_score_th, image_width, image_height,
    )

    detected_labels = []
    detections = []

    for i in range(len(bboxes)):
        bbox = bboxes[i].tolist() + [scores[i], class_ids[i]]
        detections.append(bbox)
        label_name = class_names[int(class_ids[i])]
        detected_labels.append(label_name)

    return detections, detected_labels

# ***********************************************************************
# Main Loop
# ***********************************************************************

# Initialize Scene Monitor
monitor = SceneMonitor(LOG_ENTRY_DELAY_MS, LOG_EXIT_DELAY_MS, CAPTURES_DIR)

# GStreamer Pipeline
pipeline = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open the camera.")
    exit()
else:
    print("Camera opened successfully.")

# GPIO Setup
gpio_0_ip = ol.ip_dict['axi_gpio_0']
gpio_out = AxiGPIO(gpio_0_ip).channel1
mask = 0xffffffff

frame_count = 0
avg_start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        start_time = time.time()

        # Run Inference
        detections, detected_labels = run(frame)
        
        # --- LOGIC UPGRADE: Event Based Logging ---
        events = monitor.update(detected_labels, frame)
        
        for event in events:
            # Output format suitable for MQTT/Data pipelines
            json_output = json.dumps(event)
            print(json_output)
            
            # TODO: Add MQTT Publish here
            # client.publish("kria/events", json_output)
            
            # TODO: Add GCP Upload here
            # upload_blob(bucket_name, event['image_path_local'], event['image_filename'])
        
        # --- HEADLESS CHECK ---
        if not HEADLESS:
            # Draw boxes only if display is enabled to save CPU
            if len(detections) > 0:
                bboxes_with_scores = np.array(detections)
                draw_bbox(frame, bboxes_with_scores, class_names)
            
            cv2.imshow(f"YOLOX-Nano", frame)
            
            # Check for Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # In headless mode, we still need a small sleep to prevent
            # busy-looping 100% CPU if processing is too fast (unlikely on DPU but good practice)
            # or to allow other threads/interrupts to handle gracefully.
            pass

        # Performance Metrics
        end_time = time.time()
        frame_count += 1

        if frame_count % 100 == 0:
            avg_end_time = time.time()
            elapsed_time = avg_end_time - avg_start_time
            fps = frame_count / elapsed_time
            if not HEADLESS:
                print(f"Avg_FPS: {fps:.2f}")
            frame_count = 0
            avg_start_time = time.time()

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Cleanup
    gpio_out.write(0x00, mask) # Turn off LEDs
    cap.release()
    cv2.destroyAllWindows()
    del overlay
    del dpu
    print("Cleaned up resources.")
