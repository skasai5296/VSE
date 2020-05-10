import argparse
import os
import sys
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np
import nvgpu
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset import CocoDataset, EmbedDataset
from model import SPVSE
from utils import SPVSELoss, collater, sec2str
from vocab import Vocabulary


def train(epoch, loader, model, optimizer, lossfunc, vocab, args):
    begin = time.time()
    maxit = int(len(loader.dataset) / args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    metrics = {}
    for it, data in enumerate(loader):
        if it == 1:
            print(nvgpu.gpu_info())
        """image, target, index, img_id"""
        image = data["image"]
        caption = data["caption"]
        caption = [i[np.random.randint(0, len(i))] for i in caption]
        target = vocab.return_idx(caption)
        lengths = target.ne(vocab.padidx).sum(dim=1)

        optimizer.zero_grad()

        image = image.to(device)
        target = target.to(device)

        # im_emb, cap_emb = model(image, target, lengths)
        im_emb, cap_emb, gen, rec = model(image, target, lengths)
        # lossval = lossfunc(im_emb, cap_emb)
        lossval, lossdict = lossfunc(im_emb, cap_emb, gen, rec, target[:, 1:])
        lossval.backward()
        if not metrics:
            metrics = lossdict
        else:
            for k, v in lossdict.items():
                metrics[k] += v
        # clip gradient norm
        if args.grad_clip > 0:
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if it % args.log_every == args.log_every - 1:
            lossstr = " | ".join([f"{k}: {v/args.log_every:.05f}" for k, v in metrics.items()])
            print(
                f"epoch {epoch} | {sec2str(time.time()-begin)} | {it+1:06d}/{maxit:06d} iterations "
                f"| {lossstr}",
                flush=True,
            )
            metrics = {}


def validate(epoch, loader, model, vocab, args):
    begin = time.time()
    print("begin validation for epoch {}".format(epoch), flush=True)
    dset = EmbedDataset(loader, model, vocab, args)
    print("val dataset created | {} ".format(sec2str(time.time() - begin)), flush=True)
    im = dset.embedded["image"]
    cap = dset.embedded["caption"]

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
    cap_per_image = 5
    for i in range(cap_per_image):
        gt = (np.arange(nd) * cap_per_image).reshape(-1, 1) + i
        rank = np.where(I == gt)[1]
        allrank.append(rank)
    allrank = np.stack(allrank)
    allrank = np.amin(allrank, 0)
    for rank in [1, 5, 10, 20]:
        data["i2c_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / len(allrank)
    data["i2c_median@r"] = np.median(allrank) + 1
    data["i2c_mean@r"] = np.mean(allrank)

    # cap2im
    cpu_index.reset()
    cpu_index.add(im)
    D, I = cpu_index.search(cap, nd)
    # TODO: Make more efficient, do not hardcode 5
    gt = np.arange(nq).reshape(-1, 1) // cap_per_image
    allrank = np.where(I == gt)[1]
    for rank in [1, 5, 10, 20]:
        data["c2i_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / len(allrank)
    data["c2i_median@r"] = np.median(allrank) + 1
    data["c2i_mean@r"] = np.mean(allrank)

    print("-" * 50)
    print("results of cross-modal retrieval")
    for key, val in data.items():
        print("{}: {}".format(key, val), flush=True)
    print("-" * 50)
    return data


def main():
    args = parse_args()
    print(args)

    train_transform = transforms.Compose(
        [
            transforms.Resize(args.imsize_pre),
            transforms.RandomCrop(args.imsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(args.imsize_pre),
            transforms.CenterCrop(args.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if args.dataset == "coco":
        train_dset = CocoDataset(root=args.root_path, transform=train_transform)
        val_dset = CocoDataset(
            root=args.root_path,
            imgdir="val2017",
            jsonfile="annotations/captions_val2017.json",
            transform=val_transform,
        )
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        collate_fn=collater,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        collate_fn=collater,
    )

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(args.vocab_path)

    # model = VSE(
    model = SPVSE(
        len(vocab),
        args.emb_size,
        args.out_size,
        args.max_len,
        args.cnn_type,
        args.rnn_type,
        pad_idx=vocab.padidx,
        bos_idx=vocab.bosidx,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = model.to(device)

    cfgs = [
        {"params": model.parameters(), "lr": args.lr_cnn},
    ]
    if args.optimizer == "SGD":
        optimizer = optim.SGD(cfgs, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(cfgs, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(cfgs, alpha=args.alpha, weight_decay=args.weight_decay)
    if args.scheduler == "Plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.dampen_factor, patience=args.patience, verbose=True
        )
    elif args.scheduler == "Step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.patience, gamma=args.dampen_factor
        )
    lossfunc = SPVSELoss(vocab.padidx, weight_rank=1.0, weight_gen=1.0, weight_rec=1.0)

    if args.checkpoint is not None:
        print(
            "loading model and optimizer checkpoint from {} ...".format(args.checkpoint), flush=True
        )
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if args.scheduler != "None":
            scheduler.load_state_dict(ckpt["scheduler_state"])
        offset = ckpt["epoch"]
        data = ckpt["stats"]
        bestscore = 0
        for rank in [1, 5, 10, 20]:
            bestscore += data["i2c_recall@{}".format(rank)] + data["c2i_recall@{}".format(rank)]
        bestscore = int(bestscore)
    else:
        offset = 0
        bestscore = -1
    model = nn.DataParallel(model)

    metrics = {}
    es_cnt = 0

    assert offset < args.max_epochs
    for ep in range(offset, args.max_epochs):
        train(ep + 1, train_loader, model, optimizer, lossfunc, vocab, args)
        data = validate(ep + 1, val_loader, model, vocab, args)
        totalscore = 0
        for rank in [1, 5, 10, 20]:
            totalscore += data["i2c_recall@{}".format(rank)] + data["c2i_recall@{}".format(rank)]
        totalscore = int(totalscore)
        if args.scheduler == "Plateau":
            scheduler.step(totalscore)
        if args.scheduler == "Step":
            scheduler.step()

        # save checkpoint
        ckpt = {
            "stats": data,
            "epoch": ep + 1,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if args.scheduler != "None":
            ckpt["scheduler_state"] = scheduler.state_dict()
        savedir = os.path.join("models", args.config_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for k, v in data.items():
            if k not in metrics.keys():
                metrics[k] = [v]
            else:
                metrics[k].append(v)

        savepath = os.path.join(
            savedir, "epoch_{:04d}_score_{:03d}.ckpt".format(ep + 1, totalscore)
        )
        if int(totalscore) > int(bestscore):
            print(
                "score: {:03d}, saving model and optimizer checkpoint to {} ...".format(
                    totalscore, savepath
                ),
                flush=True,
            )
            bestscore = totalscore
            torch.save(ckpt, savepath)
            es_cnt = 0
        else:
            print(
                "score: {:03d}, no improvement from best score of {:03d}, not saving".format(
                    totalscore, bestscore
                ),
                flush=True,
            )
            es_cnt += 1
            if es_cnt == args.es_cnt:
                print(
                    "early stopping at epoch {} because of no improvement for {} epochs".format(
                        ep + 1, args.es_cnt
                    )
                )
                break
        print("done for epoch {:04d}".format(ep + 1), flush=True)

    visualize(metrics, args)
    print("complete training")


def visualize(metrics, args):
    savedir = os.path.join("out", args.config_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title(
        "Recall for lr={}, margin={}, bs={}".format(args.lr_cnn, args.margin, args.batch_size)
    )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Recall")
    for k, v in metrics.items():
        if "recall" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.savefig(os.path.join(savedir, "recall.png"))

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title(
        "Median ranking for lr={}, margin={}, bs={}".format(
            args.lr_cnn, args.margin, args.batch_size
        )
    )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Median")
    for k, v in metrics.items():
        if "median" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.savefig(os.path.join(savedir, "median.png"))

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title(
        "Mean ranking for lr={}, margin={}, bs={}".format(args.lr_cnn, args.margin, args.batch_size)
    )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean")
    for k, v in metrics.items():
        if "mean" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.savefig(os.path.join(savedir, "mean.png"))


def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--root_path", type=str, default="/ssd1/dsets/coco")
    parser.add_argument("--vocab_path", type=str, default="captions_train2017.txt")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint if any, will restart training from there",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="default/",
        help="Path to save checkpoints and figures to when training",
    )

    # configurations of models
    parser.add_argument("--cnn_type", type=str, default="resnet101", help="architecture of cnn")
    parser.add_argument("--rnn_type", type=str, default="LSTM", help="architecture of rnn")

    # training config
    parser.add_argument("--n_cpu", type=int, default=8, help="number of threads for dataloading")
    parser.add_argument(
        "--method", type=str, default="max", help="hardest negative (max) or all negatives (sum)"
    )
    parser.add_argument(
        "--margin", type=float, default=0.2, help="margin for pairwise ranking loss"
    )
    parser.add_argument("--improved", action="store_true", help="improved triplet loss")
    parser.add_argument("--intra", type=float, default=0.5, help="beta for improved triplet loss")
    parser.add_argument(
        "--imp_weight", type=float, default=1e-2, help="weight for improved ranking loss"
    )
    parser.add_argument(
        "--freeze_ep", type=int, default=15, help="at which epoch to unfreeze the CNN encoder"
    )
    parser.add_argument("--emb_size", type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument(
        "--out_size", type=int, default=1024, help="embedding size for output vectors"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="max number of epochs to train for"
    )
    parser.add_argument("--max_len", type=int, default=30, help="max length of sentences")
    parser.add_argument("--log_every", type=int, default=10, help="log every x iterations")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")

    # hyperparams
    parser.add_argument(
        "--imsize_pre", type=int, default=256, help="to what size to crop the image"
    )
    parser.add_argument("--imsize", type=int, default=224, help="image size to train on.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size. must be a large number for negatives",
    )
    parser.add_argument("--lr_cnn", type=float, default=2e-4, help="learning rate of cnn")
    parser.add_argument("--lr_rnn", type=float, default=2e-4, help="learning rate of rnn")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--alpha", type=float, default=0.99, help="alpha for RMSprop")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for Adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer, [SGD, Adam, RMSprop]"
    )
    parser.add_argument(
        "--scheduler", type=str, default="Step", help="learning rate scheduler, [Plateau, Step]"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="patience of learning rate scheduler"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="weight decay of all parameters, unrecommended",
    )
    parser.add_argument("--grad_clip", type=float, default=2, help="gradient norm clipping")
    parser.add_argument(
        "--dampen_factor",
        type=float,
        default=0.1,
        help="dampening factor for learning rate scheduler",
    )
    parser.add_argument(
        "--weight_rank", type=float, default=1.0, help="loss weight for ranking loss",
    )
    parser.add_argument(
        "--weight_gen", type=float, default=0.0, help="loss weight for generation loss",
    )
    parser.add_argument(
        "--weight_rec", type=float, default=0.0, help="loss weight for reconstruction loss",
    )
    parser.add_argument("--es_cnt", type=int, default=30, help="threshold epoch for early stopping")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
