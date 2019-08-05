import sys, os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import faiss

from dataset import CocoDataset, EmbedDataset
from utils import sec2str, PairwiseRankingLoss
from model import ImageEncoder, CaptionEncoder
from vocab import Vocabulary


def train(epoch, loader, imenc, capenc, optimizer, lossfunc, vocab, args):
    begin = time.time()
    maxit = int(len(loader.dataset) / args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cumloss = 0
    for it, data in enumerate(loader):
        """image, target, index, img_id"""
        image = data["image"]
        caption = data["caption"]
        index = data["index"]
        id = data["id"]
        target = vocab.return_idx(caption)
        lengths = target.ne(vocab.padidx).sum(dim=1)

        optimizer.zero_grad()

        image = image.to(device)
        target = target.to(device)

        im_emb = imenc(image)
        cap_emb = capenc(target, lengths)
        lossval = lossfunc(im_emb, cap_emb)
        lossval.backward()
        optimizer.step()
        cumloss += lossval.item()
        if it % args.log_every == args.log_every-1:
            print("epoch {} | {} | {:06d}/{:06d} iterations | loss: {:.04f}".format(epoch, sec2str(time.time()-begin), it+1, maxit, lossval))
    return cumloss / maxit

def validate(epoch, loader, imenc, capenc, vocab, args):
    begin = time.time()
    print("begin validation for epoch {}".format(epoch), flush=True)
    dset = EmbedDataset(loader, imenc, capenc, vocab, args)
    print("val dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    im = dset.embedded["image"]
    cap = dset.embedded["caption"]
    nd = len(dset)
    nq = len(dset)
    d = im.shape[1]
    cpu_index = faiss.IndexFlatIP(d)

    # im2cap
    cpu_index.add(cap)
    print("db size: {}, dimension: {}".format(cpu_index.ntotal, cpu_index.d), flush=True)
    D, I = cpu_index.search(im, 10)
    r1 = np.sum(D[:, :1] == np.arange(nq)) / nq
    print("recall@1: {}".format(r1))

    # cap2im
    return r1

def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    if args.dataset == 'coco':
        train_dset = CocoDataset(root=args.root_path, transform=transform)
        val_dset = CocoDataset(root=args.root_path, imgdir='val2017', jsonfile='annotations/captions_val2017.json', transform=transform)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(args.vocab_path)

    imenc = ImageEncoder(args.out_size, args.cnn_type)
    capenc = CaptionEncoder(len(vocab), args.emb_size, args.out_size, args.rnn_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    imenc = imenc.to(device)
    capenc = capenc.to(device)

    optimizer = optim.SGD([
        {'params' : imenc.parameters(), 'lr' : args.lr_cnn, 'momentum' : args.mom_cnn},
        {'params' : capenc.parameters(), 'lr' : args.lr_rnn, 'momentum' : args.mom_rnn}
        ])
    lossfunc = PairwiseRankingLoss(margin=args.margin)

    for ep in range(args.max_epochs):
        #avgloss = train(ep+1, train_loader, imenc, capenc, optimizer, lossfunc, vocab, args)
        avgloss = 0
        r1 = validate(ep+1, val_loader, imenc, capenc, vocab, args)
        print("-"*50)
        print("epoch {} done, average epoch loss: {}, recall@1: {}".format(ep+1, avgloss, r1))
        print("-"*50)


def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--root_path', type=str, default='/home/seito/hdd/dsets/coco')
    parser.add_argument('--vocab_path', type=str, default='captions_train2017.txt')
    parser.add_argument('--model_path', type=str, default='../models', help='Path to read models from when training / testing')
    parser.add_argument('--model_save_path', type=str, default='../models', help='Path to save models to when training')

    # configurations of models
    parser.add_argument('--cnn_type', type=str, default="resnet18")
    parser.add_argument('--rnn_type', type=str, default="LSTM")

    # training config
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--emb_size', type=int, default=256, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=256, help="embedding size for output vectors")
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=1, help="log every x iterations")
    parser.add_argument('--no_cuda', action='store_true', help="log every x iterations")

    # hyperparams
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr_cnn', type=float, default=1e-2)
    parser.add_argument('--mom_cnn', type=float, default=0.9)
    parser.add_argument('--lr_rnn', type=float, default=1e-2)
    parser.add_argument('--mom_rnn', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
