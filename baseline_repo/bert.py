from CONSTANTS import BASE_PATH

import torch
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",  output_hidden_states = True)

global model
model = BertModel.from_pretrained("bert-base-uncased",  output_hidden_states = True)

##############################
# Bert Embedding Models      #
##############################

def get_bert_embeddings(texts):
    """Get embeddings from an embedding model
    """
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    texts = ["[CLS] " + text + " [SEP]" for text in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model(**inputs)

    # https://stackoverflow.com/questions/67703260/xlm-bert-sequence-outputs-to-pooled-output-with-weighted-average-pooling
    # For Sentence embedding, [batch_size, seq_len, dim] --> use 'cls0' token or use 'pooler_output;. We use pooler_output
    hidden_states = output.last_hidden_state
    return hidden_states
