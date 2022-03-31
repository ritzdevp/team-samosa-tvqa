import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score
import json
import os
import pysrt
from tqdm import tqdm
import h5py
from transformers import BertTokenizer, BertModel



BASE_PATH="/home/ubuntu/MML"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",  output_hidden_states = True)
bert = BertModel.from_pretrained("bert-base-uncased",  output_hidden_states = True)

##############################
# Bert Embedding Models      #
##############################

def get_bert_embeddings(model, texts):
    """Get embeddings from an embedding model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    texts = ["[CLS] " + text + " [SEP]" for text in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model(**inputs)

    # https://stackoverflow.com/questions/67703260/xlm-bert-sequence-outputs-to-pooled-output-with-weighted-average-pooling
    # For Sentence embedding, [batch_size, seq_len, dim] --> use 'cls0' token or use 'pooler_output;. We use pooler_output
    hidden_states = output.last_hidden_state
    return hidden_states

##############################
# TVQA and TVQA+ dataloaders #
##############################
# "vid_name"
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

      subt_text = TVQA.read_tvqa_subtitles(subtitle_path)
      return question, subt_text, a0, a1, a2, a3, a4, answer_idx


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
      
      subtitle_key = self.target_dict[i]['vid_name']
      subtitle = self.subtitles_dict[subtitle_key]
      subt_text = subtitle['sub_text']
      subt_text = subt_text.replace('<eos>', '[SEP]')

      return question, subt_text, a0, a1, a2, a3, a4, answer_idx




class TVQAPlusVideos(torch.utils.data.Dataset):
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



#####################################
# Model Definition - Self-Attention #
#####################################

class TVQAQAModel(torch.nn.Module):

  def __init__(self,
               q_dim: int=768,
               a_dim: int=768,
               subt_dim: int=768,
               num_ans: int=5,
               att_dim: int=64,
               ):
    super(TVQAQAModel, self).__init__()

    hidden_proj_dim = 256

    quest_proj = [nn.Linear(q_dim, hidden_proj_dim),
                  nn.GELU()]

    ans_proj = [nn.Linear(a_dim, hidden_proj_dim),
                nn.GELU()]

    subt_proj = [nn.Linear(subt_dim, hidden_proj_dim),
                 nn.GELU()]
                 


    cls_layer = [nn.Linear(att_dim, 1)]

    self.quest_proj = nn.Sequential(*quest_proj)
    self.ans_proj   = nn.Sequential(*ans_proj)
    self.subt_proj  = nn.Sequential(*subt_proj)

    self.query_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.value_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.key_proj   = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))

    self.attention = nn.MultiheadAttention(embed_dim=att_dim, num_heads=1, batch_first=True)
    self.cls_layer = nn.Sequential(*cls_layer)


  def forward(self, question, a1, a2, a3, a4, a5, subt):

    q_fwd = self.quest_proj(question)
    subt_fwd = self.subt_proj(subt)

    ans_one_fwd = self.ans_proj(a1)
    ans_two_fwd = self.ans_proj(a2)
    ans_three_fwd = self.ans_proj(a3)
    ans_four_fwd  = self.ans_proj(a4)
    ans_five_fwd = self.ans_proj(a5)

    q_ans_one_subt_concat = torch.cat((q_fwd, ans_one_fwd, subt_fwd), dim=1)
    q_ans_two_subt_concat = torch.cat((q_fwd, ans_two_fwd, subt_fwd), dim=1)
    q_ans_three_subt_concat = torch.cat((q_fwd, ans_three_fwd, subt_fwd), dim=1)
    q_ans_four_subt_concat  = torch.cat((q_fwd, ans_four_fwd, subt_fwd), dim=1)
    q_ans_five_subt_concat  = torch.cat((q_fwd, ans_five_fwd, subt_fwd), dim=1)

    att_one_query = self.query_proj(q_ans_one_subt_concat)
    att_one_key   = self.key_proj(q_ans_one_subt_concat)
    att_one_val   = self.value_proj(q_ans_one_subt_concat)

    att_two_query = self.query_proj(q_ans_two_subt_concat)
    att_two_key   = self.key_proj(q_ans_two_subt_concat)
    att_two_val   = self.value_proj(q_ans_two_subt_concat)

    att_three_query = self.query_proj(q_ans_three_subt_concat)
    att_three_key   = self.key_proj(q_ans_three_subt_concat)
    att_three_val    = self.value_proj(q_ans_three_subt_concat)

    att_four_query  = self.query_proj(q_ans_four_subt_concat)
    att_four_key    = self.key_proj(q_ans_four_subt_concat)
    att_four_val    = self.value_proj(q_ans_four_subt_concat)

    att_five_query  = self.query_proj(q_ans_five_subt_concat)
    att_five_key    = self.key_proj(q_ans_five_subt_concat)
    att_five_val    = self.value_proj(q_ans_five_subt_concat)

    att_one_fwd, att_one_weight = self.attention(query=att_one_query, key=att_one_key, value=att_one_val)
    att_one_fwd_pool = torch.max(att_one_fwd, dim=1).values
    score_one        = self.cls_layer(att_one_fwd_pool)

    att_two_fwd, att_two_weight = self.attention(query=att_two_query, key=att_two_key, value=att_two_val)
    att_two_fwd_pool = torch.max(att_two_fwd, dim=1).values
    score_two        = self.cls_layer(att_two_fwd_pool)


    att_three_fwd, att_three_weight = self.attention(query=att_three_query, key=att_three_key, value=att_three_val)
    att_three_fwd_pool = torch.max(att_three_fwd, dim=1).values
    score_three      = self.cls_layer(att_three_fwd_pool)


    att_four_fwd, att_four_weight = self.attention(query=att_four_query, key=att_four_key, value=att_four_val)
    att_four_fwd_pool  = torch.max(att_four_fwd, dim=1).values
    score_four         = self.cls_layer(att_four_fwd_pool)

    att_five_fwd, att_five_weight = self.attention(query=att_five_query, key=att_five_key, value=att_five_val)
    att_five_fwd_pool  = torch.max(att_five_fwd, dim=1).values
    score_five         = self.cls_layer(att_five_fwd_pool)


    logits = torch.cat((score_one, score_two, score_three, score_four, score_five), dim=1)

    return logits


########################
# Model - Transformer  #
########################

class TVQAQAModelTransformer(torch.nn.Module):

  def __init__(self,
               q_dim: int=768,
               a_dim: int=768,
               subt_dim: int=768,
               num_ans: int=5,
               att_dim: int=256,
               n_heads: int=8,
               enc_layers: int=6
               ):
    super(TVQAQAModelTransformer, self).__init__()

    quest_proj = [nn.Linear(q_dim, att_dim),
                  nn.GELU()]

    ans_proj = [nn.Linear(a_dim, att_dim),
                nn.GELU()]

    subt_proj = [nn.Linear(subt_dim, att_dim),
                 nn.GELU()]
    self.quest_proj = nn.Sequential(*quest_proj)
    self.ans_proj   = nn.Sequential(*ans_proj)
    self.subt_proj  = nn.Sequential(*subt_proj)


    encoder_layer = nn.TransformerEncoderLayer(d_model=att_dim, nhead=n_heads, dim_feedforward=att_dim, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

    cls_layer = [nn.Linear(att_dim, 1)]
    self.cls_layer = nn.Sequential(*cls_layer)


  def forward(self, question, a1, a2, a3, a4, a5, subt):

    q_fwd = self.quest_proj(question)
    subt_fwd = self.subt_proj(subt)

    ans_one_fwd = self.ans_proj(a1)
    ans_two_fwd = self.ans_proj(a2)
    ans_three_fwd = self.ans_proj(a3)
    ans_four_fwd  = self.ans_proj(a4)
    ans_five_fwd = self.ans_proj(a5)

    q_ans_one_subt_concat = torch.cat((subt_fwd, q_fwd, ans_one_fwd), dim=1)
    q_ans_two_subt_concat = torch.cat((subt_fwd, q_fwd, ans_two_fwd), dim=1)
    q_ans_three_subt_concat = torch.cat((subt_fwd, q_fwd, ans_three_fwd), dim=1)
    q_ans_four_subt_concat  = torch.cat((subt_fwd, q_fwd, ans_four_fwd), dim=1)
    q_ans_five_subt_concat  = torch.cat((subt_fwd, q_fwd, ans_five_fwd), dim=1)

    att_one_fwd = self.transformer_encoder(q_ans_one_subt_concat)
    att_one_fwd_pool = torch.max(att_one_fwd, dim=1).values
    score_one        = self.cls_layer(att_one_fwd_pool)

    att_two_fwd = self.transformer_encoder(q_ans_two_subt_concat)
    att_two_fwd_pool = torch.max(att_two_fwd, dim=1).values
    score_two        = self.cls_layer(att_two_fwd_pool)

    att_three_fwd = self.transformer_encoder(q_ans_three_subt_concat)
    att_three_fwd_pool = torch.max(att_three_fwd, dim=1).values
    score_three      = self.cls_layer(att_three_fwd_pool)

    att_four_fwd = self.transformer_encoder(q_ans_four_subt_concat)
    att_four_fwd_pool  = torch.max(att_four_fwd, dim=1).values
    score_four         = self.cls_layer(att_four_fwd_pool)

    att_five_fwd = self.transformer_encoder(q_ans_five_subt_concat)
    att_five_fwd_pool  = torch.max(att_five_fwd, dim=1).values
    score_five         = self.cls_layer(att_five_fwd_pool)

    logits = torch.cat((score_one, score_two, score_three, score_four, score_five), dim=1)

    return logits



class TVQAQAModelVideo(torch.nn.Module):

  def __init__(self,
               q_dim: int=768,
               a_dim: int=768,
               subt_dim: int=768,
               vid_dim: int=2048,
               num_ans: int=5,
               att_dim: int=64,
               ):
    super(TVQAQAModelVideo, self).__init__()

    hidden_proj_dim = 256

    quest_proj = [nn.Linear(q_dim, hidden_proj_dim),
                  nn.GELU()]

    ans_proj = [nn.Linear(a_dim, hidden_proj_dim),
                nn.GELU()]

    subt_proj = [nn.Linear(subt_dim, hidden_proj_dim),
                 nn.GELU()]

    vid_proj = [nn.Linear(vid_dim, hidden_proj_dim),
                nn.GELU()]
                 


    cls_layer = [nn.Linear(att_dim, 1)]

    self.quest_proj = nn.Sequential(*quest_proj)
    self.ans_proj   = nn.Sequential(*ans_proj)
    self.subt_proj  = nn.Sequential(*subt_proj)
    self.vid_proj   = nn.Sequential(*vid_proj)

    self.query_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.value_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.key_proj   = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))

    self.attention = nn.MultiheadAttention(embed_dim=att_dim, num_heads=1, batch_first=True)
    self.cls_layer = nn.Sequential(*cls_layer)


  def forward(self, question, a1, a2, a3, a4, a5, subt, vid):

    q_fwd = self.quest_proj(question)
    subt_fwd = self.subt_proj(subt)
    vid_fwd = self.vid_proj(vid)

    ans_one_fwd = self.ans_proj(a1)
    ans_two_fwd = self.ans_proj(a2)
    ans_three_fwd = self.ans_proj(a3)
    ans_four_fwd  = self.ans_proj(a4)
    ans_five_fwd = self.ans_proj(a5)

    q_ans_one_subt_vid_concat = torch.cat((q_fwd, ans_one_fwd, subt_fwd, vid_fwd), dim=1)
    q_ans_two_subt_vid_concat = torch.cat((q_fwd, ans_two_fwd, subt_fwd, vid_fwd), dim=1)
    q_ans_three_subt_vid_concat = torch.cat((q_fwd, ans_three_fwd, subt_fwd, vid_fwd), dim=1)
    q_ans_four_subt_vid_concat  = torch.cat((q_fwd, ans_four_fwd, subt_fwd, vid_fwd), dim=1)
    q_ans_five_subt_vid_concat  = torch.cat((q_fwd, ans_five_fwd, subt_fwd, vid_fwd), dim=1)

    att_one_query = self.query_proj(q_ans_one_subt_vid_concat)
    att_one_key   = self.key_proj(q_ans_one_subt_vid_concat)
    att_one_val   = self.value_proj(q_ans_one_subt_vid_concat)

    att_two_query = self.query_proj(q_ans_two_subt_vid_concat)
    att_two_key   = self.key_proj(q_ans_two_subt_vid_concat)
    att_two_val   = self.value_proj(q_ans_two_subt_vid_concat)

    att_three_query = self.query_proj(q_ans_three_subt_vid_concat)
    att_three_key   = self.key_proj(q_ans_three_subt_vid_concat)
    att_three_val    = self.value_proj(q_ans_three_subt_vid_concat)

    att_four_query  = self.query_proj(q_ans_four_subt_vid_concat)
    att_four_key    = self.key_proj(q_ans_four_subt_vid_concat)
    att_four_val    = self.value_proj(q_ans_four_subt_vid_concat)

    att_five_query  = self.query_proj(q_ans_five_subt_vid_concat)
    att_five_key    = self.key_proj(q_ans_five_subt_vid_concat)
    att_five_val    = self.value_proj(q_ans_five_subt_vid_concat)

    att_one_fwd, att_one_weight = self.attention(query=att_one_query, key=att_one_key, value=att_one_val)
    att_one_fwd_pool = torch.max(att_one_fwd, dim=1).values
    score_one        = self.cls_layer(att_one_fwd_pool)

    att_two_fwd, att_two_weight = self.attention(query=att_two_query, key=att_two_key, value=att_two_val)
    att_two_fwd_pool = torch.max(att_two_fwd, dim=1).values
    score_two        = self.cls_layer(att_two_fwd_pool)


    att_three_fwd, att_three_weight = self.attention(query=att_three_query, key=att_three_key, value=att_three_val)
    att_three_fwd_pool = torch.max(att_three_fwd, dim=1).values
    score_three      = self.cls_layer(att_three_fwd_pool)


    att_four_fwd, att_four_weight = self.attention(query=att_four_query, key=att_four_key, value=att_four_val)
    att_four_fwd_pool  = torch.max(att_four_fwd, dim=1).values
    score_four         = self.cls_layer(att_four_fwd_pool)

    att_five_fwd, att_five_weight = self.attention(query=att_five_query, key=att_five_key, value=att_five_val)
    att_five_fwd_pool  = torch.max(att_five_fwd, dim=1).values
    score_five         = self.cls_layer(att_five_fwd_pool)


    logits = torch.cat((score_one, score_two, score_three, score_four, score_five), dim=1)

    return logits



###################################
# Load the Dev and Train Loader   #  
###################################
# dev_items = TVQAPlus(dataset='val')
# batch_size_dev = 4
# dev_loader = torch.utils.data.DataLoader(dev_items, batch_size=batch_size_dev, shuffle=False)

# train_items = TVQAPlus(dataset='train')
# batch_size = 16
# train_loader = torch.utils.data.DataLoader(train_items, batch_size=batch_size, shuffle=True)

dev_items_video = TVQAPlusVideos(dataset='val')
batch_size_dev = 4
dev_loader_video = torch.utils.data.DataLoader(dev_items_video, batch_size=batch_size_dev, shuffle=False)

train_items_video = TVQAPlusVideos(dataset='train')
batch_size = 16
train_loader_video = torch.utils.data.DataLoader(train_items_video, batch_size=batch_size, shuffle=True)




def val_acc(model):
  model.eval()
  num_correct = 0
  for batch_idx, (question, subt_text, a0, a1, a2, a3, a4, ans_ohe) in enumerate(dev_loader):
    ans_ohe = ans_ohe.cuda()
    quest_embed = get_bert_embeddings(model=bert, texts=question)
    subt_text_embed = get_bert_embeddings(model=bert, texts=subt_text)
    a0_embed = get_bert_embeddings(model=bert, texts=a0)
    a1_embed = get_bert_embeddings(model=bert, texts=a1)
    a2_embed = get_bert_embeddings(model=bert, texts=a2)
    a3_embed = get_bert_embeddings(model=bert, texts=a3)
    a4_embed = get_bert_embeddings(model=bert, texts=a4)

    with torch.no_grad():
      logits = model.forward(question=quest_embed, 
                                  a1=a0_embed, 
                                  a2=a1_embed, 
                                  a3=a2_embed,
                                  a4=a3_embed, 
                                  a5=a4_embed,
                                  subt=subt_text_embed)
      
    num_correct += int((torch.argmax(logits, axis=1) == ans_ohe).sum())
    acc = 100 * num_correct / ((batch_idx + 1) * batch_size_dev)

  dev_acc = 100 * num_correct / (len(dev_loader) * batch_size_dev)

  model.train()
  return dev_acc


def val_acc_video(model):
  model.eval()
  num_correct = 0
        for batch_idx, (question, subt_text, a0, a1, a2, a3, a4, video_name, ans_ohe) in enumerate(train_loader_video):
            ans_ohe = ans_ohe.cuda()

            quest_embed = get_bert_embeddings(model=bert, texts=question)
            subt_text_embed = get_bert_embeddings(model=bert, texts=subt_text)
            a0_embed = get_bert_embeddings(model=bert, texts=a0)
            a1_embed = get_bert_embeddings(model=bert, texts=a1)
            a2_embed = get_bert_embeddings(model=bert, texts=a2)
            a3_embed = get_bert_embeddings(model=bert, texts=a3)
            a4_embed = get_bert_embeddings(model=bert, texts=a4)

            # print("video_resnet_feat", video_resnet_feat.shape)
            video_resnet_feat = []
            for video in video_name:
                # print("video", video, vid_h5[video])
                video_resnet_feat.append(torch.tensor(vid_h5[video], device="cuda"))

            video_resnet_feat =  pad_sequence(video_resnet_feat, batch_first=True)

    with torch.no_grad():
      logits = model.forward(question=quest_embed, 
                                  a1=a0_embed, 
                                  a2=a1_embed, 
                                  a3=a2_embed,
                                  a4=a3_embed, 
                                  a5=a4_embed,
                                  subt=subt_text_embed,
                                  vid=video_resnet_feat)
      
    num_correct += int((torch.argmax(logits, axis=1) == ans_ohe).sum())
    acc = 100 * num_correct / ((batch_idx + 1) * batch_size_dev)

  dev_acc = 100 * num_correct / (len(dev_loader) * batch_size_dev)

  model.train()
  return dev_acc


def train_loop():

    tvqa_model = TVQAQAModel()
    tvqa_model.cuda()

    optimizer = optim.Adam(tvqa_model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print('tvqa_model', tvqa_model)
    model_version='model_attention_tvqa_v1.pt'
    epoch = 0
    best_dev_acc = 0
    while epoch < 100:

        loss_epoch = 0
        num_correct = 0
        optimizer.zero_grad()
        tvqa_model.train()

        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        if os.path.exists(f'{BASE_PATH}/MultiModalExp/{model_version}'):
            # model.load_state_dict(torch.load(f'{SAVE_PATH}{EXP_TAG}/model_saved_epoch{epoch-1}.pt')) 

            checkpoint = torch.load(f'{BASE_PATH}/MultiModalExp/{model_version}')
            tvqa_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            
        for batch_idx, (question, subt_text, a0, a1, a2, a3, a4, ans_ohe) in enumerate(train_loader):
            ans_ohe = ans_ohe.cuda()

            quest_embed = get_bert_embeddings(model=bert, texts=question)
            subt_text_embed = get_bert_embeddings(model=bert, texts=subt_text)
            a0_embed = get_bert_embeddings(model=bert, texts=a0)
            a1_embed = get_bert_embeddings(model=bert, texts=a1)
            a2_embed = get_bert_embeddings(model=bert, texts=a2)
            a3_embed = get_bert_embeddings(model=bert, texts=a3)
            a4_embed = get_bert_embeddings(model=bert, texts=a4)

            logits = tvqa_model.forward(question=quest_embed, 
                                        a1=a0_embed, 
                                        a2=a1_embed, 
                                        a3=a2_embed,
                                        a4=a3_embed, 
                                        a5=a4_embed,
                                        subt=subt_text_embed)
            
            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((batch_idx + 1) * batch_size)),
                loss="{:.04f}".format(float(loss_epoch / (batch_idx + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
            

            loss = criterion(logits, ans_ohe)
            num_correct += int((torch.argmax(logits, axis=1) == ans_ohe).sum())

            loss.backward()
            optimizer.step()
            loss_epoch += float(loss)
            optimizer.zero_grad()

            batch_bar.update() # Update tqdm bar



        batch_bar.close() # You need this to close the tqdm bar
        torch.save({
                'epoch': epoch,
                'model_state_dict': tvqa_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                },  f'{BASE_PATH}/MultiModalExp/{model_version}')

    train_acc = 100 * num_correct / (len(train_loader) * batch_size)
    dev_acc = val_acc(tvqa_model)

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save({
                'epoch': epoch,
                'model_state_dict': tvqa_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_acc': dev_acc,
                'train_acc': train_acc,
                'loss': loss,
                },  f'{BASE_PATH}/MultiModalExp/best_dev_acc_{best_dev_acc}_{model_version}')

    print(f'Epoch {epoch} Loss {loss_epoch} train_acc {train_acc}, devacc {dev_acc}')
    epoch += 1

    scheduler.step()

def train_loop_video():

    tvqa_model = TVQAQAModelVideo()
    tvqa_model.cuda()

    optimizer = optim.Adam(tvqa_model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print('tvqa_model', tvqa_model)
    model_version='model_video_attention_tvqaplus_v1.pt'

    h5driver=None
    vid_feat_path="/home/ubuntu/MML/tvqa_imagenet_pool5_hq.h5"
    vid_h5 = h5py.File(vid_feat_path, "r", driver=h5driver)

    epoch = 0
    best_dev_acc = 0
    while epoch < 100:

        loss_epoch = 0
        num_correct = 0
        optimizer.zero_grad()
        tvqa_model.train()

        batch_bar = tqdm(total=len(train_loader_video), dynamic_ncols=True, leave=False, position=0, desc='Train')

        if os.path.exists(f'{BASE_PATH}/MultiModalExp/{model_version}'):
            # model.load_state_dict(torch.load(f'{SAVE_PATH}{EXP_TAG}/model_saved_epoch{epoch-1}.pt')) 

            checkpoint = torch.load(f'{BASE_PATH}/MultiModalExp/{model_version}')
            tvqa_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            
        for batch_idx, (question, subt_text, a0, a1, a2, a3, a4, video_name, ans_ohe) in enumerate(train_loader_video):
            ans_ohe = ans_ohe.cuda()

            quest_embed = get_bert_embeddings(model=bert, texts=question)
            subt_text_embed = get_bert_embeddings(model=bert, texts=subt_text)
            a0_embed = get_bert_embeddings(model=bert, texts=a0)
            a1_embed = get_bert_embeddings(model=bert, texts=a1)
            a2_embed = get_bert_embeddings(model=bert, texts=a2)
            a3_embed = get_bert_embeddings(model=bert, texts=a3)
            a4_embed = get_bert_embeddings(model=bert, texts=a4)

            # print("video_resnet_feat", video_resnet_feat.shape)
            video_resnet_feat = []
            for video in video_name:
                # print("video", video, vid_h5[video])
                video_resnet_feat.append(torch.tensor(vid_h5[video], device="cuda"))

            video_resnet_feat =  pad_sequence(video_resnet_feat, batch_first=True)

            # print("video_resnet_feat shape", video_resnet_feat.shape)
                
            logits = tvqa_model.forward(question=quest_embed, 
                                        a1=a0_embed, 
                                        a2=a1_embed, 
                                        a3=a2_embed,
                                        a4=a3_embed, 
                                        a5=a4_embed,
                                        subt=subt_text_embed,
                                        vid=video_resnet_feat)
            
            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((batch_idx + 1) * batch_size)),
                loss="{:.04f}".format(float(loss_epoch / (batch_idx + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
            

            loss = criterion(logits, ans_ohe)
            num_correct += int((torch.argmax(logits, axis=1) == ans_ohe).sum())

            loss.backward()
            optimizer.step()
            loss_epoch += float(loss)
            optimizer.zero_grad()

            batch_bar.update() # Update tqdm bar



        batch_bar.close() # You need this to close the tqdm bar
        torch.save({
                'epoch': epoch,
                'model_state_dict': tvqa_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                },  f'{BASE_PATH}/MultiModalExp/{model_version}')

        train_acc = 100 * num_correct / (len(train_loader) * batch_size)
        dev_acc = val_acc(tvqa_model)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': tvqa_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_acc': dev_acc,
                    'train_acc': train_acc,
                    'loss': loss,
                    },  f'{BASE_PATH}/MultiModalExp/best_dev_acc_{best_dev_acc}_{model_version}')

        print(f'Epoch {epoch} Loss {loss_epoch} train_acc {train_acc}, devacc {dev_acc}')
        epoch += 1

        scheduler.step()


if __name__ == "__main__":
    train_loop_video()