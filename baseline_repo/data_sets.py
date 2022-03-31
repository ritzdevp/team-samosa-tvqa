from CONSTANTS import BASE_PATH

import torch
import json
import pysrt
import torch.nn as nn
import torch.optim as optim

##############################
# TVQA and TVQA+ dataloaders #
##############################
class TVQA(torch.utils.data.Dataset):

    SUBTITLE_PATH = BASE_PATH + '/tvqa_subtitles/'

    def __init__(self, dataset='train'):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        TVQA_TRAIN_DICT_JSON = BASE_PATH + "/tvqa_qa_release/tvqa_train.jsonl"
        TVQA_VAL_DICT_JSON = BASE_PATH + "/tvqa_qa_release/tvqa_val.jsonl"
        TVQA_TEST_DICT_JSON = BASE_PATH + '/tvqa_qa_release/tvqa_test_public.jsonl'

        self.dataset = dataset

        anno_tvqa_val = []
        anno_tvqa_test_public = []
        anno_tvqa_train = []

        with open(TVQA_VAL_DICT_JSON,'r',encoding='utf-8') as j:
          for line in j:
            anno_tvqa_val.append(json.loads(line))

        with open(TVQA_TRAIN_DICT_JSON,'r',encoding='utf-8') as j:
          for line in j:
            anno_tvqa_train.append(json.loads(line))

        with open(TVQA_TEST_DICT_JSON,'r',encoding='utf-8') as j:
          for line in j:
            anno_tvqa_test_public.append(json.loads(line))


        self.target_dict = {}
        if self.dataset == 'train':
          self.target_dict = anno_tvqa_train
        elif self.dataset == 'val':
          self.target_dict = anno_tvqa_val
        else:
          raise Exception(f"Invalid dataset passed {self.dataset}")

    @staticmethod
    def read_tvqa_subtitles(srt_path: str):
      subs = pysrt.open(srt_path)
      
      texts = []
      
      for sub in subs:
        line_text = sub.text.replace('(', '')
        line_text = line_text.replace(':)', ': ')
        line_text = line_text.replace('\n', '')
        texts.append(line_text)

      return " [SEP] ".join(texts)
      
    def __len__(self):
        return len(self.target_dict)
        
    def __getitem__(self, i):

      question = self.target_dict[i]['q']
    
      a0       = self.target_dict[i]['a0']
      a1       = self.target_dict[i]['a1']
      a2       = self.target_dict[i]['a2']
      a3       = self.target_dict[i]['a3']
      a4       = self.target_dict[i]['a4']

      answer_idx = int(self.target_dict[i]['answer_idx'])      
      subtitle_path = TVQA.SUBTITLE_PATH + self.target_dict[i]["vid_name"] + ".srt"
      video_name = self.target_dict[i]['vid_name']

      subt_text = TVQA.read_tvqa_subtitles(subtitle_path)
      return question, subt_text, a0, a1, a2, a3, a4, video_name, answer_idx


class TVQAPlus(torch.utils.data.Dataset):
    def __init__(self, dataset='train'):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        TRAIN_DICT_JSON = BASE_PATH + "/tvqa_plus/tvqa_plus_train.json"
        VAL_DICT_JSON = BASE_PATH + '/tvqa_plus/tvqa_plus_val.json'
        SUBTITLES_DICT_JSON = BASE_PATH + '/tvqa_plus/tvqa_plus_subtitles.json'

        self.dataset = dataset

        train_dict = {}
        val_dict = {}
        subtitles_dict = {}

        with open(TRAIN_DICT_JSON) as f:
          train_dict = json.load(f)

        with open(VAL_DICT_JSON) as f:
          val_dict = json.load(f)

        with open(SUBTITLES_DICT_JSON) as f:
          self.subtitles_dict = json.load(f)

        self.target_dict = {}
        if self.dataset == 'train':
          self.target_dict = train_dict
        elif self.dataset == 'val':
          self.target_dict = val_dict
        else:
          raise Exception(f"Invalid dataset passed {self.dataset}")
      
    def __len__(self):
        return len(self.target_dict)
        
    def __getitem__(self, i):

      question = self.target_dict[i]['q']
    
      a0       = self.target_dict[i]['a0']
      a1       = self.target_dict[i]['a1']
      a2       = self.target_dict[i]['a2']
      a3       = self.target_dict[i]['a3']
      a4       = self.target_dict[i]['a4']

      answer_idx = int(self.target_dict[i]['answer_idx'])
      
      video_name = self.target_dict[i]['vid_name']
      subtitle = self.subtitles_dict[video_name]
      subt_text = subtitle['sub_text']
      subt_text = subt_text.replace('<eos>', '[SEP]')

      return question, subt_text, a0, a1, a2, a3, a4, video_name, answer_idx
