import sys, os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import faiss

from dataset import CocoDataset, EmbedDataset
from utils import sec2str, PairwiseRankingLoss, collater
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
        img_id = data["img_id"]
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
            print("epoch {} | {} | {:06d}/{:06d} iterations | loss: {:.08f}".format(epoch, sec2str(time.time()-begin), it+1, maxit, cumloss/args.log_every), flush=True)
            cumloss = 0

def validate(epoch, loader, imenc, capenc, vocab, args):
    begin = time.time()
    print("begin validation for epoch {}".format(epoch), flush=True)
    dset = EmbedDataset(loader, imenc, capenc, vocab, args)
    print("val dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    im = dset.embedded["image"]
    cap = dset.embedded["caption"]

    # img_ids = dset.embedded["img_id"]
    # ann_ids = dset.embedded["ann_id"]
    # idx2im_id = img_ids
    # idx2cap_id = [a for ann in ann_ids for a in ann]
    # print(idx2im_id)
    # print(idx2cap_id)
    # print(len(img_ids)) # 5000
    # print(len(ann_ids)) # 25000

    nd = im.shape[0]
    nq = cap.shape[0]
    d = im.shape[1]
    cpu_index = faiss.IndexFlatIP(d)

    print("# images: {}, # captions: {}, dimension: {}".format(nd, nq, d), flush=True)
    # im2cap
    cpu_index.add(cap)
    D, I = cpu_index.search(im, nq)
    data = {}
    allrank = []

    # TODO: Make more efficient, do not hardcode 5
    for i in range(5):
        gt = (np.arange(nd) * 5).reshape(-1, 1) + i
        rank = np.where(I == gt)[1]
        allrank.append(rank)
    allrank = np.stack(allrank)
    allrank = np.amin(allrank, 0)
    for rank in [1, 5, 10, 20]:
        data["i2c_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / len(allrank)
    data["i2c_median@r"] = np.median(allrank) + 1

    # cap2im
    cpu_index.reset()
    cpu_index.add(im)
    D, I = cpu_index.search(cap, nd)
    # TODO: Make more efficient, do not hardcode 5
    gt = np.arange(nq).reshape(-1, 1) // 5
    allrank = np.where(I == gt)[1]
    for rank in [1, 5, 10, 20]:
        data["c2i_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / len(allrank)
    data["c2i_median@r"] = np.median(allrank) + 1

    print("-"*50)
    print("results of cross-modal retrieval")
    for key, val in data.items():
        print("{}: {}".format(key, val), flush=True)
    print("-"*50)
    return data

def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    if args.dataset == 'coco':
        train_dset = CocoDataset(root=args.root_path, transform=transform, mode='one')
        val_dset = CocoDataset(root=args.root_path, imgdir='val2017', jsonfile='annotations/captions_val2017.json', transform=transform, mode='all')
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, collate_fn=collater)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, collate_fn=collater)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(args.vocab_path)

    imenc = ImageEncoder(args.out_size, args.cnn_type)
    capenc = CaptionEncoder(len(vocab), args.emb_size, args.out_size, args.rnn_type, vocab.padidx)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    imenc = imenc.to(device)
    capenc = capenc.to(device)

    optimizer = optim.SGD([
        {'params' : imenc.parameters(), 'lr' : args.lr_cnn, 'momentum' : args.mom_cnn},
        {'params' : capenc.parameters(), 'lr' : args.lr_rnn, 'momentum' : args.mom_rnn}
        ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.dampen_factor, patience=args.patience, verbose=True)
    lossfunc = PairwiseRankingLoss(margin=args.margin, method=args.method, improved=args.improved, intra=args.intra)

    if args.checkpoint is not None:
        print("loading model and optimizer checkpoint from {} ...".format(args.checkpoint), flush=True)
        ckpt = torch.load(args.checkpoint)
        imenc.load_state_dict(ckpt["encoder_state"])
        capenc.load_state_dict(ckpt["decoder_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        offset = ckpt["epoch"]
    else:
        offset = 0
    imenc = nn.DataParallel(imenc)
    capenc = nn.DataParallel(capenc)

    metrics = {}
    bestscore = -1

    assert offset < args.max_epochs
    for ep in range(offset, args.max_epochs):
        #train(ep+1, train_loader, imenc, capenc, optimizer, lossfunc, vocab, args)
        data = validate(ep+1, val_loader, imenc, capenc, vocab, args)
        totalscore = 0
        for rank in [1, 5, 10, 20]:
            totalscore += int(100 * (data["i2c_recall@{}".format(rank)] + data["c2i_recall@{}".format(rank)]))
        scheduler.step(totalscore)

        # save checkpoint
        ckpt = {
                "stats": data,
                "epoch": ep+1,
                "encoder_state": imenc.module.state_dict(),
                "decoder_state": capenc.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
                }
        savedir = os.path.join("models", args.config_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        savepath = os.path.join(savedir, "epoch_{:04d}_score_{:05d}.ckpt".format(ep+1, totalscore))
        if totalscore > bestscore:
            print("score: {:05d}, saving model and optimizer checkpoint to {} ...".format(totalscore, savepath), flush=True)
            bestscore = totalscore
            torch.save(ckpt, savepath)
        else:
            print("score: {:05d}, no improvement from best score {:05d}, not saving".format(totalscore, bestscore), flush=True)
        print("done for epoch {:04d}".format(ep+1), flush=True)

        for k, v in data.items():
            if k not in metrics.keys():
                metrics[k] = [v]
            else:
                metrics[k].append(v)

    visualize(metrics, args)

def visualize(metrics, args):
    savepath = os.path.join("out", args.config_name)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Recall for lr={}, margin={}, bs={}".format(args.lr_cnn, args.margin, args.batch_size))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Recall")
    for k, v in metrics.items():
        if "median" in k:
            continue
        ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savepath, "recall.png"))

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Median ranking for lr={}, margin={}, bs={}".format(args.lr_cnn, args.margin, args.batch_size))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Median")
    for k, v in metrics.items():
        if "recall" in k:
            continue
        ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savepath, "median.png"))


def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--root_path', type=str, default='/ssd1/dsets/coco')
    parser.add_argument('--vocab_path', type=str, default='captions_train2017.txt')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint if any, will restart training from there')
    parser.add_argument('--config_name', type=str, default='default/', help='Path to save checkpoints and figures to when training')

    # configurations of models
    parser.add_argument('--cnn_type', type=str, default="resnet18", help="architecture of cnn")
    parser.add_argument('--rnn_type', type=str, default="LSTM", help="architecture of rnn")

    # training config
    parser.add_argument('--n_cpu', type=int, default=8, help="number of threads for dataloading")
    parser.add_argument('--method', type=str, default="max", help="hardest negative (max) or all negatives (sum)")
    parser.add_argument('--margin', type=float, default=0.2, help="margin for pairwise ranking loss")
    parser.add_argument('--improved', action='store_true', help="improved triplet loss")
    parser.add_argument('--intra', type=float, default=0.5, help="beta for improved triplet loss")
    parser.add_argument('--emb_size', type=int, default=512, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=512, help="embedding size for output vectors")
    parser.add_argument('--max_epochs', type=int, default=30, help="max number of epochs to train for")
    parser.add_argument('--max_len', type=int, default=30, help="max length of sentences")
    parser.add_argument('--log_every', type=int, default=10, help="log every x iterations")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")

    # hyperparams
    parser.add_argument('--imsize', type=int, default=224, help="image size to train on.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size. must be a large number for negatives")
    parser.add_argument('--lr_cnn', type=float, default=1e-2, help="learning rate of cnn")
    parser.add_argument('--mom_cnn', type=float, default=0.9, help="momentum of cnn")
    parser.add_argument('--lr_rnn', type=float, default=1e-2, help="learning rate of rnn")
    parser.add_argument('--mom_rnn', type=float, default=0.9, help="momentum of rnn")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay of all parameters")
    parser.add_argument('--patience', type=int, default=10, help="patience of learning rate scheduler")
    parser.add_argument('--dampen_factor', type=float, default=0.5, help="dampening factor for learning rate scheduler")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
