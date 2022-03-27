#NOTE: '/content/frames_buffer/' should exist

import os, sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request

#for bbt, the episodes don't have bbt in frames folder name
shows_prefix_link = {
    'bbt':'https://cardibmmml.s3.us-east-1.amazonaws.com/frames_hq/',
    'house':'https://tvqammml.s3.us-east-1.amazonaws.com/',
    'grey':'https://tvqammml.s3.us-east-1.amazonaws.com/',
    'castle':'https://cardibmmml.s3.us-east-1.amazonaws.com/frames_hq/',
    'met':'https://tvqammml.s3.us-east-1.amazonaws.com/',
    'friends':'https://cardibmmml.s3.us-east-1.amazonaws.com/frames_hq/',
}

def get_s3_name(link):
  start = link.index('://') + 3
  end = link.index('.s3')
  bucket_name = link[start:end]
  return bucket_name

def clear_frames_buffer():
  while (len(os.listdir('/content/frames_buffer')) != 0):
    os.system('rm -rf /content/frames_buffer/*')
  return

def get_frames(vid_name, skip=1):
  #skip=number of frames to be skipped during retrieval

  clear_frames_buffer()
  filename_dummy = '00000'
  _index = vid_name.index('_')
  show_name = vid_name[:_index]
  if (show_name not in shows_prefix_link):
    show_name = 'bbt'
  frames_folder = show_name + '_' + 'frames'
  print(show_name)
  prefix_link = ''
  if (get_s3_name(shows_prefix_link[show_name]) == 'tvqammml'):
    prefix_link = shows_prefix_link[show_name] 
  else:
    prefix_link = shows_prefix_link[show_name] + frames_folder + '/'
  for i in range(1,300,skip):
    filename_temp = (filename_dummy + str(i))[-5:] + '.jpg'
    frame_link = prefix_link + vid_name + '/' + filename_temp
    try:
      urllib.request.urlretrieve(frame_link, '/content/frames_buffer/'+filename_temp)
    except:
      print("Reached end.")
      if (len(os.listdir('/content/frames_buffer')) == 0):
        print("Frames not found.")
      break
