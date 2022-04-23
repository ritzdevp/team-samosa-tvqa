from CONSTANTS import BASE_PATH

import torch
from transformers import BertTokenizer, VisualBertForQuestionAnswering, VisualBertModel


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",  output_hidden_states = True)

global vis_bert_model
# vis_bert_model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa",  output_hidden_states = True)
vis_bert_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

def get_visbert_embeddings_new(texts, visual_embeds):
    """Get embeddings from an embedding model
    """
    global vis_bert_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=vis_bert_model.to(device)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=True,
        return_attention_mask=True, truncation=True).to(device)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)
    inputs.update(
        {
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        }
    )
    output = vis_bert_model(**inputs)
    # print("out: ", output)

    # https://stackoverflow.com/questions/67703260/xlm-bert-sequence-outputs-to-pooled-output-with-weighted-average-pooling
    # For Sentence embedding, [batch_size, seq_len, dim] --> use 'cls0' token or use 'pooler_output;. We use pooler_output
    
    hidden_state = output['last_hidden_state']
    cls_tokens = hidden_state[:,0,:]

    # hidden_states = output.last_hidden_state
    # return hidden_states
    return cls_tokens
