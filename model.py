import sys, os
import time
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
import torchvision

from utils import sec2str, weight_init


# L2 normalize a batched tensor (bs, ft)
def l2normalize(ten):
    norm = torch.norm(ten, dim=1, keepdim=True)
    return ten / norm

class ImageEncoder(nn.Module):
    def __init__(self, out_size=256, cnn_type="resnet18", pretrained=True):
        super(ImageEncoder, self).__init__()
        self.cnn = getattr(torchvision.models, cnn_type)(pretrained)
        # replace final fc layer
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.model.classifier._modules['6'].in_features,
                                out_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.fc.in_features, out_size)
            self.cnn.fc = nn.Sequential()
        if not pretrained:
            self.cnn.apply(weight_init)
        self.fc.apply(weight_init)


    def forward(self, x):
        resout = self.cnn(x)
        out = self.fc(resout)
        normed_out = l2normalize(out)
        return normed_out

class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size=256, out_size=256, rnn_type="LSTM", pad_idx=0):
        super(CaptionEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = getattr(nn, rnn_type)(emb_size, out_size, batch_first=True)

        self.emb.apply(weight_init)
        self.rnn.apply(weight_init)

    # x: (bs, seq)
    # lengths: (bs)
    def forward(self, x, lengths):
        emb = self.emb(x)
        # packed: PackedSequence of (bs, seq, emb_size)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.rnn(packed)
        # hn: (bs, seq, emb_size)
        normed_out = l2normalize(hn[0])
        return normed_out

if __name__ == '__main__':
    ten = torch.randn((16, 3, 224, 224))

    cnn = ImageEncoder()
    out = cnn(ten)
    print(out)
    print(out.size())


    cap = CaptionEncoder(vocab_size=100)
    seq = torch.randint(100, (16, 30), dtype=torch.long)
    len = torch.randint(1, 31, (16,), dtype=torch.long)
    out = cap(seq, len)
    print(out)
    print(out.size())

