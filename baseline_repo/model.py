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



class TVQAMultiModalModel(torch.nn.Module):

  def __init__(self,
               q_dim: int=768,
               a_dim: int=768,
               subt_dim: int=768,
               vid_dim: int=2048,
               num_ans: int=5,
               att_dim: int=64,
               ):
    super(TVQAMultiModalModel, self).__init__()

    hidden_proj_dim = 256

    quest_proj = [nn.Linear(q_dim, hidden_proj_dim),
                  nn.GELU()]

    ans_proj = [nn.Linear(a_dim, hidden_proj_dim),
                nn.GELU()]

    subt_proj = [nn.Linear(subt_dim, hidden_proj_dim),
                 nn.GELU()]

    vid_proj = [nn.Linear(vid_dim, hidden_proj_dim),
                nn.GELU()]
                 
    

    cls_layer_sub = [nn.Linear(64, 64),
                 nn.GELU(),
                 nn.Linear(64, 1)]

    cls_layer_vid = [nn.Linear(64, 64),
                 nn.GELU(),
                 nn.Linear(64, 1)]

    self.quest_proj = nn.Sequential(*quest_proj)
    self.ans_proj   = nn.Sequential(*ans_proj)
    self.subt_proj  = nn.Sequential(*subt_proj)
    self.vid_proj   = nn.Sequential(*vid_proj)

    self.query_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.value_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.key_proj   = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))

    self.attention_vid = nn.MultiheadAttention(embed_dim=att_dim, num_heads=1, batch_first=True)
    self.attention_sub = nn.MultiheadAttention(embed_dim=att_dim, num_heads=1, batch_first=True)
    self.cls_layer_sub = nn.Sequential(*cls_layer_sub)
    self.cls_layer_vid = nn.Sequential(*cls_layer_vid)


  def forward(self, question, a1, a2, a3, a4, a5, subt, vid):

    q_fwd = self.quest_proj(question)
    subt_fwd = self.subt_proj(subt)
    vid_fwd = self.vid_proj(vid)
    #Creating list of answer projections
    ans_fwd = [self.ans_proj(a) for a in [a1,a2,a3,a4,a5]]

    #Concatenating Q A_i and S for each answer choice
    q_a_s = [torch.cat((q_fwd, ans_i_fwd, subt_fwd), dim = 1)
             for ans_i_fwd in ans_fwd]

    #Concatenating Q A_i and V for each answer choice
    q_a_v = [torch.cat((q_fwd, ans_i_fwd, vid_fwd), dim = 1)
             for ans_i_fwd in ans_fwd] 


    #Computing Attention Q,K,V values for Subtitles and Videos 
    #for each answer choice a_i
   
    att_subt = [{ "q":self.query_proj(q_a_i_s),
                  "k":self.key_proj(q_a_i_s),
                  "v":self.value_proj(q_a_i_s)}
                for q_a_i_s in q_a_s
                ]

    att_vid = [{ "q":self.query_proj(q_a_i_v),
                  "k":self.key_proj(q_a_i_v),
                  "v":self.value_proj(q_a_i_v)}
                for q_a_i_v in q_a_v
                ]
    

    #Computing attention weighted q,a,s for each answer choice
    q_a_s_att = [torch.max(self.attention_sub(query = att_subt_a_i["q"],
                                key = att_subt_a_i["k"],
                                value = att_subt_a_i["v"])[0],
                            dim = 1 ).values
                 for att_subt_a_i in att_subt]

    #Computing attention weighted q,a,v for each answer choice
    q_a_v_att =  [torch.max(self.attention_vid(query = att_vid_a_i["q"],
                                key = att_vid_a_i["k"],
                                value = att_vid_a_i["v"])[0],
                              dim = 1).values
                  for att_vid_a_i in att_vid]
    


    score_sub = [self.cls_layer_sub(q_a_i_s_att)
                 for q_a_i_s_att in q_a_s_att]
    
    score_vid = [self.cls_layer_vid(q_a_i_v_att)
                 for q_a_i_v_att in q_a_v_att]
    
    score_all = [0.8*score_sub[i] + 0.2*score_vid[i] 
                 for i in range(5)]
    
    logits = torch.cat(score_all, dim=1)


    return logits

  
  
class TVQAMMLinear(torch.nn.Module):

  def __init__(self,
               q_dim: int=768,
               a_dim: int=768,
               subt_dim: int=768,
               vid_dim: int=2048,
               num_ans: int=5,
               att_dim: int=64,
               ):
    super(TVQAMMLinear, self).__init__()

    hidden_proj_dim = 256

    quest_proj = [nn.Linear(q_dim, hidden_proj_dim),
                  nn.GELU()]

    ans_proj = [nn.Linear(a_dim, hidden_proj_dim),
                nn.GELU()]

    subt_proj = [nn.Linear(subt_dim, hidden_proj_dim),
                 nn.GELU()]

    vid_proj = [nn.Linear(vid_dim, hidden_proj_dim),
                nn.GELU()]

    linear = [nn.Linear(hidden_proj_dim, att_dim*2),
              nn.GELU(),
              nn.Linear(att_dim*2, att_dim),
              nn.GELU()]
              
    cls_layer = [nn.Linear(att_dim*2, att_dim),
                 nn.GELU(),
                 nn.Linear(att_dim, 1)]

    self.quest_proj = nn.Sequential(*quest_proj)
    self.ans_proj   = nn.Sequential(*ans_proj)
    self.subt_proj  = nn.Sequential(*subt_proj)
    self.vid_proj   = nn.Sequential(*vid_proj)

    self.linear = nn.Sequential(*linear)
    self.cls_layer = nn.Sequential(*cls_layer)


  def forward(self, question, a1, a2, a3, a4, a5, subt, vid):

    q_fwd = self.quest_proj(question)
    subt_fwd = self.subt_proj(subt)
    vid_fwd = self.vid_proj(vid)

    ans_fwd = [self.ans_proj(ans) for ans in [a1, a2, a3, a4, a5]]
    
    q_a_s = [torch.cat((q_fwd, ans_i_fwd, subt_fwd), dim = 1)
             for ans_i_fwd in ans_fwd]
    q_a_v = [torch.cat((q_fwd, ans_i_fwd, vid_fwd), dim = 1)
             for ans_i_fwd in ans_fwd] 

    q_a_s_lin = [torch.max(self.linear(q_a_s_i), dim=1).values
             for q_a_s_i in q_a_s]
    q_a_v_lin = [torch.max(self.linear(q_a_v_i), dim=1).values
             for q_a_v_i in q_a_v]

    
    score_one = self.cls_layer(torch.cat((q_a_s_lin[0], q_a_v_lin[0]), dim=1))
    score_two = self.cls_layer(torch.cat((q_a_s_lin[1], q_a_v_lin[1]), dim=1))
    score_three = self.cls_layer(torch.cat((q_a_s_lin[2], q_a_v_lin[2]), dim=1))
    score_four = self.cls_layer(torch.cat((q_a_s_lin[3], q_a_v_lin[3]), dim=1))
    score_five = self.cls_layer(torch.cat((q_a_s_lin[4], q_a_v_lin[4]), dim=1))

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
	
	vid_proj = [nn.Linear(vid_dim, hidden_proj_dim),
                nn.GELU()]

    cls_layer = [nn.Linear(att_dim, 1)]

    self.quest_proj = nn.Sequential(*quest_proj)
    self.ans_proj   = nn.Sequential(*ans_proj)
    self.vid_proj   = nn.Sequential(*vid_proj)

    self.query_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.value_proj = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))
    self.key_proj   = nn.Sequential(nn.Linear(hidden_proj_dim, att_dim))

    self.attention = nn.MultiheadAttention(embed_dim=att_dim, num_heads=1, batch_first=True)
    self.cls_layer = nn.Sequential(*cls_layer)


  def forward(self, question, a1, a2, a3, a4, a5, vid):

    q_fwd = self.quest_proj(question)
    vid_fwd = self.vid_proj(vid)

    ans_one_fwd = self.ans_proj(a1)
    ans_two_fwd = self.ans_proj(a2)
    ans_three_fwd = self.ans_proj(a3)
    ans_four_fwd  = self.ans_proj(a4)
    ans_five_fwd = self.ans_proj(a5)

    q_ans_one_vid_concat = torch.cat((q_fwd, ans_one_fwd, vid_fwd), dim=1)
    q_ans_two_vid_concat = torch.cat((q_fwd, ans_two_fwd, vid_fwd), dim=1)
    q_ans_three_vid_concat = torch.cat((q_fwd, ans_three_fwd, vid_fwd), dim=1)
    q_ans_four_vid_concat  = torch.cat((q_fwd, ans_four_fwd, vid_fwd), dim=1)
    q_ans_five_vid_concat  = torch.cat((q_fwd, ans_five_fwd, vid_fwd), dim=1)

    att_one_query = self.query_proj(q_ans_one_vid_concat)
    att_one_key   = self.key_proj(q_ans_one_vid_concat)
    att_one_val   = self.value_proj(q_ans_one_vid_concat)

    att_two_query = self.query_proj(q_ans_two_vid_concat)
    att_two_key   = self.key_proj(q_ans_two_vid_concat)
    att_two_val   = self.value_proj(q_ans_two_vid_concat)

    att_three_query = self.query_proj(q_ans_three_vid_concat)
    att_three_key   = self.key_proj(q_ans_three_vid_concat)
    att_three_val    = self.value_proj(q_ans_three_vid_concat)

    att_four_query  = self.query_proj(q_ans_four_vid_concat)
    att_four_key    = self.key_proj(q_ans_four_vid_concat)
    att_four_val    = self.value_proj(q_ans_four_vid_concat)

    att_five_query  = self.query_proj(q_ans_five_vid_concat)
    att_five_key    = self.key_proj(q_ans_five_vid_concat)
    att_five_val    = self.value_proj(q_ans_five_vid_concat)

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
