import torch
import torch.nn as nn
import torch.optim as optim

from CONSTANTS import BASE_PATH

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


