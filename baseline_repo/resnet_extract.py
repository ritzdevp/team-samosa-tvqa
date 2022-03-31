import h5py
import torch
from torch.nn.utils.rnn import pad_sequence

from CONSTANTS import RESNET_FEATURES


h5driver=None
vid_feat_path=RESNET_FEATURES
vid_h5 = h5py.File(vid_feat_path, "r", driver=h5driver)


def read_resnet_feats(video_names):

    video_resnet_feat = []
    for video in video_names:
        video_resnet_feat.append(torch.tensor(vid_h5[video], device="cuda"))

    video_resnet_feat =  pad_sequence(video_resnet_feat, batch_first=True)
    return video_resnet_feat