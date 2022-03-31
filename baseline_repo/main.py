import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from data_sets import TVQA, TVQAPlus

from model import TVQAQAModel

from train import train

if __name__ == "__main__":

    tvqa_model = TVQAQAModel()

    
    batch_size=16
    batch_size_dev=4

    train_dataset = TVQA(dataset="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TVQA(dataset="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_dev, shuffle=False)


    optimizer = optim.Adam(tvqa_model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model_version = "tvqa_subt_v1.pt"
    train(tvqa_model, optimizer, criterion, scheduler, model_version, train_loader, val_loader, batch_size, batch_size_dev)