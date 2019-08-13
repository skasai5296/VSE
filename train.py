import sys, os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import faiss

from dataset import CocoDataset, EmbedDataset
from utils import sec2str, PairwiseRankingLoss, collater_eval, collater_train
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
    return imenc, capenc, optimizer

def validate(epoch, loader, imenc, capenc, vocab, args):
    begin = time.time()
    print("begin validation for epoch {}".format(epoch), flush=True)
    dset = EmbedDataset(loader, imenc, capenc, vocab, args)
    print("val dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    im = dset.embedded["image"]
    cap = dset.embedded["caption"]
    img_ids = dset.embedded["img_id"]
    ann_ids = dset.embedded["ann_id"]
    #print(len(img_ids)) # 5000
    #print(len(ann_ids)) # 25000

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

    for i in range(5):
        gt = (np.arange(nd) * 5).reshape(-1, 1) + i
        rank = np.where(I == gt)[1]
        allrank.append(rank)
    allrank = np.stack(allrank)
    allrank = np.amin(allrank, 0)
    for rank in [1, 5, 10, 20]:
        data["i2c_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / nq
    data["i2c_median@r"] = np.median(allrank) + 1

    # cap2im
    cpu_index.reset()
    cpu_index.add(im)
    D, I = cpu_index.search(cap, nd)
    gt = np.arange(nq).reshape(-1, 1) // 5
    allrank = np.where(I == gt)[1]
    for rank in [1, 5, 10, 20]:
        data["c2i_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / nq
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
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, collate_fn=collater_train)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, collate_fn=collater_eval)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, verbose=True)
    lossfunc = PairwiseRankingLoss(margin=args.margin)

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

    assert offset < args.max_epochs
    for ep in range(offset, args.max_epochs):
        #imenc, capenc, optimizer = train(ep+1, train_loader, imenc, capenc, optimizer, lossfunc, vocab, args)
        data = validate(ep+1, val_loader, imenc, capenc, vocab, args)
        totalscore = 0
        for rank in [1, 5, 10, 20]:
            totalscore += data["i2c_recall@{}".format(rank)] + data["c2i_recall@{}".format(rank)]
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
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)
        savepath = os.path.join(args.model_save_path, "epoch_{:04d}_score_{:05d}.ckpt".format(ep+1, int(100*totalscore)))
        print("saving model and optimizer checkpoint to {} ...".format(savepath), flush=True)
        torch.save(ckpt, savepath)
        print("done for epoch {}".format(ep+1), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--root_path', type=str, default='/home/seito/hdd/dsets/coco')
    parser.add_argument('--vocab_path', type=str, default='captions_train2017.txt')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint if any, will restart training from there')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Path to save models to when training')

    # configurations of models
    parser.add_argument('--cnn_type', type=str, default="resnet18")
    parser.add_argument('--rnn_type', type=str, default="LSTM")

    # training config
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--margin', type=float, default=0.05)
    parser.add_argument('--emb_size', type=int, default=512, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=512, help="embedding size for output vectors")
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=10, help="log every x iterations")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")

    # hyperparams
    parser.add_argument('--imsize', type=int, default=224, help="image size to train on.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size. must be a large number for negatives")
    parser.add_argument('--lr_cnn', type=float, default=1e-3, help="learning rate of cnn")
    parser.add_argument('--mom_cnn', type=float, default=0.9, help="momentum of cnn")
    parser.add_argument('--lr_rnn', type=float, default=1e-3, help="learning rate of rnn")
    parser.add_argument('--mom_rnn', type=float, default=0.9, help="momentum of rnn")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay of all parameters")
    parser.add_argument('--patience', type=int, default=5, help="patience of learning rate scheduler")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
