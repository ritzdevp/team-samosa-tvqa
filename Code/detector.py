import os, sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import torch
import torchvision
import matplotlib.patches as patches
import cv2

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

#https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
def detect_objects(model, img_filename):
  img = Image.open(img_filename)
  img_np = np.asarray(img)
  img_t = torch.from_numpy(img_np)
  img_t_2 = torch.permute(img_t, (2, 0, 1))
  img_t_2 = img_t_2/255
  x = [img_t_2]
  predictions = model(x)
  return predictions

def make_bbox(img_filename, predictions):
  image = cv2.imread(img_filename)
  for i in range(20):
    label = predictions[0]['labels'][i]
    bbox = predictions[0]['boxes'][i]
    cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),1) # add rectangle to image
    cv2.putText(image, COCO_INSTANCE_CATEGORY_NAMES[label], (bbox[0], bbox[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 246, 55), 1)
    cv2.imwrite('out_detect.jpg', image)
  return
