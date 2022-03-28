import os, sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import normalize
import matplotlib.patches as patches
from torchvision import transforms
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Extracts RESNET 101 Imagenet features of an image file
def extract_resnet_feature(filename):
  resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
  resnet101.eval()


  modules=list(resnet101.children())[:-1] #last block, 2048 pooled features
  resnet101_seq = nn.Sequential(*modules)

  input_image = Image.open(filename)
  preprocess = transforms.Compose([
      transforms.Resize(256),
      # transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to(device)
      resnet101_seq.to(device)

  with torch.no_grad():
      output = resnet101_seq(input_batch)
  
  return output


def get_feature_stack():
  frames = os.listdir('/content/frames_buffer')
  feature_stack = None
  for frame in frames:
    frame_link = '/content/frames_buffer/' + frame
    feature = extract_resnet_feature(frame_link)
    feature_norm = normalize(feature[0].flatten(), p=2,dim=0).detach() #detach because no gradients need to be computed for this, L2 norm
    if (feature_stack is None):
      feature_stack = feature_norm
    else:
      feature_stack = torch.vstack((feature_stack, feature_norm))
  return feature_stack
