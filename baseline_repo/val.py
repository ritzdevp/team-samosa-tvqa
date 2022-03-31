
import torch
import torch.nn as nn
from bert import get_bert_embeddings
from resnet_extract import read_resnet_feats


def val_acc(model, dev_loader, batch_size_dev):
  model.eval()
  num_correct = 0
  for batch_idx, batch in enumerate(dev_loader):
    question, subt_text, a0, a1, a2, a3, a4, video_name, ans_ohe = batch
    ans_ohe = ans_ohe.cuda()
    quest_embed = get_bert_embeddings(texts=question)
    subt_text_embed = get_bert_embeddings(texts=subt_text)
    a0_embed = get_bert_embeddings(texts=a0)
    a1_embed = get_bert_embeddings(texts=a1)
    a2_embed = get_bert_embeddings(texts=a2)
    a3_embed = get_bert_embeddings(texts=a3)
    a4_embed = get_bert_embeddings(texts=a4)

    with torch.no_grad():
      # # IF MODEL does not TAKES VIDEO INPUT

      # logits = model.forward(question=quest_embed, 
      #                             a1=a0_embed, 
      #                             a2=a1_embed, 
      #                             a3=a2_embed,
      #                             a4=a3_embed, 
      #                             a5=a4_embed,
      #                             subt=subt_text_embed)

    # # IF MODEL TAKES VIDEO INPUT
    with torch.no_grad():
      logits = model.forward(question=quest_embed, 
                              a1=a0_embed, 
                              a2=a1_embed, 
                              a3=a2_embed,
                              a4=a3_embed, 
                              a5=a4_embed,
                              vid=video_resnet_feat)
      
    num_correct += int((torch.argmax(logits, axis=1) == ans_ohe).sum())
    acc = 100 * num_correct / ((batch_idx + 1) * batch_size_dev)

  dev_acc = 100 * num_correct / (len(dev_loader) * batch_size_dev)

  model.train()
  return dev_acc
