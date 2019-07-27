import sys, os
sys.path.append(os.pardir)
import time
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision

from utils import sec2str, weight_init



class ImageEncoder(nn.Module):
    def __init__(self, cnn_type="resnet18", embed_size=256, pretrained=True):
        super(ImageEncoder, self).__init__()
        self.model = getattr(torchvision.models, cnn_type)(pretrained)
        # replace final fc layer
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.model.classifier._modules['6'].in_features,
                                embed_size)
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.model.fc.in_features, embed_size)
            self.model.fc = nn.Sequential()
        if not pretrained:
            self.model.apply(weight_init)
        self.fc.apply(weight_init)


    def forward(self, x):
        resout = self.model(x)
        out = self.fc(resout)
        return out

class CaptionEncoder(nn.Module):
    def __init__(self, rnn_type="LSTM", embed_size=256, pretrained=True):
        super(CaptionEncoder, self).__init__()
        self.model = getattr(torchvision.models, cnn_type)(pretrained)
        # replace final fc layer
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.model.classifier._modules['6'].in_features,
                                embed_size)
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.model.fc.in_features, embed_size)
            self.model.fc = nn.Sequential()

        self.model.apply(weight_init)

    def forward(self, x):
        resout = self.model(x)
        out = self.fc(resout)
        return out

if __name__ == '__main__':
    ten = torch.randn((16, 3, 224, 224))

    model = ImageEncoder()
    out = model(ten)
    print(out.size())


