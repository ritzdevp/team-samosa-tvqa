import pickle
import json
import os
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwordslist = stopwords.words('english')
stopwordslist.append("?")
exceptlist = ["what", "who", "whom", "how", "where", "why"]
for w in exceptlist:
  stopwordslist.remove(w)

vis_concepts = pickle.load(open('det_visual_concepts_hq.pickle', 'rb'))

embedding_index = {}
f = open('/content/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embedding_index[word] = coefs
f.close()

def get_detected_objects(episode, index):
  x = vis_concepts[episode][index].strip().split(",")
  y = []
  for w in x:
    y.append(w.strip())
  return y

def phrase_splitter(arr):
  new_arr = []
  for w in arr:
    temp = w.split()
    for t in temp:
      if (t not in new_arr):
        new_arr.append(t)
  return new_arr

def get_glove_wordlist_embedding(arr_of_words):
  glove_wordlist_embedding = []
  for w in arr_of_words:
    if (w in embedding_index):
      glove_wordlist_embedding.append(embedding_index[w])
  glove_wordlist_embedding = np.array(glove_wordlist_embedding)
  return glove_wordlist_embedding


def tokenize_question(question):
  words = nltk.word_tokenize(question)
  filtered_words = [word.lower() for word in words if word.lower() not in stopwordslist ]
  qarr = []
  for w in filtered_words:
    if w in embedding_index:
      qarr.append(w)
  return qarr

def relevance_score(episode, frame_index, question):
  #function to get relevance score between question and detected objects in a frame
  detected_objects_list = get_detected_objects(episode, frame_index)[:10]
  words = nltk.word_tokenize(question)

  #to split words like "pink shirt" to "pink" and "shirt"
  detected_objects_list = phrase_splitter(detected_objects_list)

  #num_det is number of words in detected objects list
  #num_q is number of words in question
  det_objs_embedding_matrix = get_glove_wordlist_embedding(detected_objects_list) #num_det x 50
  question_embedding_matrix = get_glove_wordlist_embedding(tokenize_question(question)) #num_q x 50

  score_matrix = det_objs_embedding_matrix.dot(question_embedding_matrix.T)  #num_det x num_q
  score = np.sum(score_matrix)

  return score_matrix, score

### returns maximum sum subarray of given window size

def get_maxsum_subarray(myarr, window_size):
  start = 0
  sum = 0
  currlen = 0
  ws = window_size
  maxsum = 0
  maxindexend = -1
  for i in range(len(myarr)-ws):
    sum += myarr[i]
    currlen += 1
    if (currlen > ws):
      sum -= myarr[i - ws]
      currlen -= 1
    
    if (sum > maxsum):
      maxsum = sum
      maxindexend = i
  
  maxstartindex = maxindexend - ws + 1
  return maxstartindex, maxsum


### returns relevance scores of all frames for the given question
def get_relevance_scorelist(vid_name, question):
  num_frames = len(vis_concepts[vid_name])
  score_list = np.zeros(num_frames)

  max_score_index = -1
  max_score = 0
  min_score_index = -1
  min_score = 10000

  for frame_index in range(num_frames):
    score_matrix, score = relevance_score(vid_name, frame_index, question)
    score_list[frame_index] = score
    if (score > max_score):
      max_score = score
      max_score_index = frame_index
    if (score < min_score):
      min_score = score
      min_score_index = frame_index

  return score_list, max_score_index, min_score_index

def make_link(vid_name, frame_index):
  dummy = "https://tvqammml.s3.us-east-1.amazonaws.com/"
  temp = "00000"
  temp = (temp + str(frame_index))[-5:]
  url = dummy + vid_name + '/' + temp + '.jpg'
  return url

def plot_image_from_url(url):
  image = io.imread(url)
  plt.imshow(image)
  plt.show()

def plot_max_relevance_frames(vid_name, start_index, window_size):
  dummy = "https://tvqammml.s3.us-east-1.amazonaws.com/"
  fig = plt.figure(figsize=(10, 20))
  columns = 2
  rows = window_size/2
  j = start_index
  for i in range(1, window_size +1):
      temp = "00000"
      temp = (temp + str(j))[-5:]
      url = dummy + vid_name + '/' + temp + '.jpg'
      img = io.imread(url)
      fig.add_subplot(rows, columns, i)
      plt.imshow(img)
      j += 1
  plt.show()


"""
Returns the unique objects detected in an episode from
a start frame index, given the window size
"""

def get_unique_objs_in_chunk(episode, start_index, window_size):
  num_frames = len(vis_concepts[episode])
  temp = set()
  for i in range(start_index, start_index + window_size, 1):
    arr = get_detected_objects(episode, i)
    arr = phrase_splitter(arr)
    for w in arr:
      temp.add(w)
  return list(temp)
