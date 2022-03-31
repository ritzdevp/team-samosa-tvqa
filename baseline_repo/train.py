import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.optim as optim


from CONSTANTS import BASE_PATH, RESNET_FEATURES
from val import val_acc
from bert import get_bert_embeddings
from resnet_extract import read_resnet_feats


def train(tvqa_model, optimizer, criterion, scheduler, model_version, train_loader, val_loader,
          batch_size, batch_size_dev):

    tvqa_model.cuda()

    if not os.path.exists(f'{BASE_PATH}/MultiModalExp/'):
        raise Exception("Please Create a Path MultiModalExp/ to store the checkpoints!! ")

    print('tvqa_model', tvqa_model)
    print('optimizer', optimizer)
    print('scheduler', scheduler)
    print('model version', model_version)

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
            
        for batch_idx, (question, subt_text, a0, a1, a2, a3, a4, video_name, ans_ohe) in enumerate(train_loader):
            ans_ohe = ans_ohe.cuda()

            quest_embed = get_bert_embeddings(texts=question)
            subt_text_embed = get_bert_embeddings(texts=subt_text)
            a0_embed = get_bert_embeddings(texts=a0)
            a1_embed = get_bert_embeddings(texts=a1)
            a2_embed = get_bert_embeddings(texts=a2)
            a3_embed = get_bert_embeddings(texts=a3)
            a4_embed = get_bert_embeddings(texts=a4)

            video_resnet_feat = read_resnet_feats(video_name)

            # IF MODEL TAKES VIDEO AS INPUT
            # logits = tvqa_model.forward(question=quest_embed, 
            #                             a1=a0_embed, 
            #                             a2=a1_embed, 
            #                             a3=a2_embed,
            #                             a4=a3_embed, 
            #                             a5=a4_embed,
            #                             subt=subt_text_embed,
            #                             vid=video_resnet_feat)

            logits = tvqa_model.forward(question=quest_embed, 
                                        a1=a0_embed, 
                                        a2=a1_embed, 
                                        a3=a2_embed,
                                        a4=a3_embed, 
                                        a5=a4_embed,
                                        subt=subt_text_embed)
            

            loss = criterion(logits, ans_ohe)
            num_correct += int((torch.argmax(logits, axis=1) == ans_ohe).sum())

            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((batch_idx + 1) * batch_size)),
                loss="{:.04f}".format(float(loss_epoch / (batch_idx + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

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
        dev_acc = val_acc(tvqa_model, val_loader, batch_size_dev)

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