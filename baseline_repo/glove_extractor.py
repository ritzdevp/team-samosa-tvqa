from CONSTANTS import BASE_PATH
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence

f = open(os.path.join(BASE_PATH,'glove.6B.50d.txt'))
embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = values[1:]
    embedding_index[word] = coefs
f.close()

def get_glove_embeddings(texts):

    glove_embed = []
    for sentence in list(texts):
        temp = [embedding_index[word] for word in sentence  if word in embedding_index.keys()]
        if temp:
            temp = np.array(temp, dtype='float32')
        else:
            temp = np.zeros((1,50), dtype='float32')
        
        glove_embed.append(torch.tensor(temp, device="cuda"))

    glove_embed = pad_sequence(glove_embed, batch_first=True)

    return glove_embed





   