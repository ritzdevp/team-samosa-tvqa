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

glove_dir = '/content/'

embedding_index = {}

f = open(os.path.join(glove_dir,'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embedding_index[word] = coefs
f.close()

# print('found word vecs: ',len(embedding_index))

#checks if detected labels are in embedding index or not
def process_list(detected_list):
  detected_list_processed = []
  for i in range(len(detected_list)):
    w = detected_list[i]
    if (w in embedding_index):
      detected_list_processed.append(w)
    else:
      arr = w.split()
      temp = "".join(arr)
      if (temp in embedding_index):
        detected_list_processed.append(temp)
        continue
      for t in arr:
        if (t in embedding_index):
          detected_list_processed.append(t)
  return detected_list_processed


def get_vis_concepts(detected_list):
  detected_list_processed = process_list(detected_list)
  vis_concept = None
  for w in detected_list_processed:
    vc_of_w = torch.as_tensor(embedding_index[w])
    if (vis_concept is None):
      vis_concept = vc_of_w
    else:
      vis_concept = torch.vstack((vis_concept, vc_of_w))
  return vis_concept
