import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

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
        if cnn_type.startswith("vgg"):
            self.fc = nn.Linear(self.model.classifier._modules["6"].in_features, out_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith("resnet"):
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
    def __init__(self, vocab_size, emb_size=256, out_size=256, rnn_type="LSTM", padidx=0):
        super(CaptionEncoder, self).__init__()
        self.out_size = out_size
        self.padidx = padidx
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
        output, _ = self.rnn(packed)
        # output: (bs, seq, out_size)
        output = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)[0]
        # lengths: (bs, 1, out_size)
        lengths = lengths.view(-1, 1, 1).expand(-1, -1, self.out_size) - 1
        # out: (bs, out_size)
        out = torch.gather(output, 1, lengths).squeeze(1)
        normed_out = l2normalize(out)
        return normed_out


class VSE(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        out_size,
        max_seqlen,
        cnn_type,
        rnn_type,
        pretrained=True,
        dropout_prob=0.1,
        ss_prob=0.0,
        pad_idx=0,
        bos_idx=1,
    ):
        super().__init__()
        self.im_enc = ImageEncoder(out_size, cnn_type, pretrained)
        self.cap_enc = CaptionEncoder(vocab_size, emb_size, out_size, rnn_type, pad_idx)
        self.cap_dec = SimpleDecoder(
            out_size, emb_size, out_size, vocab_size, max_seqlen, dropout_prob, ss_prob, bos_idx,
        )

    def forward(self, image, caption, lengths):
        im_emb = self.im_enc(image) if image is not None else None
        cap_emb = self.cap_enc(caption, lengths) if caption is not None else None
        return im_emb, cap_emb


class SimpleDecoder(nn.Module):
    """
    RNN decoder for captioning, Google NIC
    Args:
        feature_dim:    dimension of image feature
        emb_dim:        dimension of word embeddings
        memory_dim:     dimension of LSTM memory
        vocab_size:     vocabulary size
        max_seqlen:     max sequence size
        dropout_p:      dropout probability for LSTM memory
        ss_prob:        scheduled sampling rate, 0 for teacher forcing and 1 for free running
    """

    def __init__(
        self, feature_dim, emb_dim, memory_dim, vocab_size, max_seqlen, dropout_p, ss_prob, bos_idx,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.ss_prob = ss_prob
        self.bos_idx = bos_idx

        self.init_h = nn.Linear(feature_dim, memory_dim)
        self.init_c = nn.Linear(feature_dim, memory_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTMCell(emb_dim, memory_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(memory_dim, vocab_size)

        self.init_h.apply(weight_init)
        self.init_c.apply(weight_init)
        self.rnn.apply(weight_init)
        self.emb.apply(weight_init)
        self.linear.apply(weight_init)

    def forward(self, feature, caption, length):
        """
        Args:
            torch.Tensor feature:       (bs x feature_dim), torch.float
            torch.Tensor caption:       (bs x max_seqlen), torch.long
            torch.Tensor length:        (bs), torch.long
        Returns:
            torch.Tensor out:           (bs x vocab_size x max_seqlen-1), contains logits
        """
        bs = caption.size(0)
        scale = bs // feature.size(0)
        print(feature.size(), caption.size(), length.size())
        if scale > 1:
            feature = torch.repeat_interleave(feature, scale, dim=0)
        print(feature.size(), caption.size(), length.size())
        print((feature[0] == feature[1]).all())
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature)
        cn = self.init_c(feature)
        # caption: (bs x max_seqlen x emb_dim)
        caption = self.emb(caption)
        xn = caption[:, 0, :]
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.empty((bs, self.vocab_size, self.max_seqlen - 1), device=feature.device)
        for step in range(self.max_seqlen - 1):
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(xn, (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out[:, :, step] = on
            # xn: (bs x emb_dim)
            xn = (
                self.emb(on.argmax(dim=1))
                if np.random.uniform() < self.ss_prob
                else caption[:, step + 1, :]
            )
        return out

    def sample(self, feature):
        bs = feature.size(0)
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature)
        cn = self.init_c(feature)
        # xn: (bs x emb_dim)
        xn = self.emb(torch.full((bs,), self.bos_idx, dtype=torch.long, device=feature.device))
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.empty((bs, self.vocab_size, self.max_seqlen - 1), device=feature.device)
        for step in range(self.max_seqlen - 1):
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(xn, (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out[:, :, step] = on
            # xn: (bs x emb_dim)
            xn = self.emb(on.argmax(dim=1))
        return out


class SPVSE(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        out_size,
        max_seqlen,
        cnn_type,
        rnn_type,
        pretrained=True,
        dropout_prob=0.1,
        ss_prob=0.0,
        pad_idx=0,
        bos_idx=1,
    ):
        super().__init__()
        self.im_enc = ImageEncoder(out_size, cnn_type, pretrained)
        self.cap_enc = CaptionEncoder(vocab_size, emb_size, out_size, rnn_type, pad_idx)
        self.cap_gen = SimpleDecoder(
            out_size, emb_size, out_size, vocab_size, max_seqlen, dropout_prob, ss_prob, bos_idx,
        )
        self.cap_rec = SimpleDecoder(
            out_size, emb_size, out_size, vocab_size, max_seqlen, dropout_prob, ss_prob, bos_idx,
        )

    def forward(self, image, caption, lengths):
        if image is not None:
            im_emb = self.im_enc(image)
            gen = self.cap_gen(im_emb, caption, lengths)
        else:
            im_emb, gen = None, None
        if caption is not None:
            cap_emb = self.cap_enc(caption, lengths)
            rec = self.cap_rec(cap_emb, caption, lengths)
        else:
            cap_emb, rec = None, None
        return im_emb, cap_emb, gen, rec


if __name__ == "__main__":
    ten = torch.randn((16, 3, 224, 224))

    cnn = ImageEncoder()
    out = cnn(ten)
    print(out.size())

    cap = CaptionEncoder(vocab_size=100)
    seq = torch.randint(100, (16, 30), dtype=torch.long)
    len = torch.randint(1, 31, (16,), dtype=torch.long)
    out = cap(seq, len)
    print(out.size())
