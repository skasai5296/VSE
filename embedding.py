import argparse
import glob
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import CocoDataset, EmbedDataset
from model import CaptionEncoder, ImageEncoder
from utils import collater, sec2str
from vocab import Vocabulary


def main():
    args = parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((args.imsize, args.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if args.dataset == "coco":
        val_dset = CocoDataset(root=args.root_path, split="val", transform=transform,)
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        collate_fn=collater,
    )

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(args.vocab_path)

    imenc = ImageEncoder(args.out_size, args.cnn_type)
    capenc = CaptionEncoder(len(vocab), args.emb_size, args.out_size, args.rnn_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    imenc = imenc.to(device)
    capenc = capenc.to(device)

    assert args.checkpoint is not None
    print("loading model and optimizer checkpoint from {} ...".format(args.checkpoint), flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    imenc.load_state_dict(ckpt["encoder_state"])
    capenc.load_state_dict(ckpt["decoder_state"])

    begin = time.time()
    dset = EmbedDataset(val_loader, imenc, capenc, vocab, args)
    print("database created | {} ".format(sec2str(time.time() - begin)), flush=True)

    savedir = os.path.join("out", args.config_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir, 0o777)

    image = dset.embedded["image"]
    caption = dset.embedded["caption"]
    n_i = image.shape[0]
    n_c = caption.shape[0]
    all = np.concatenate([image, caption], axis=0)

    emb_file = os.path.join(savedir, "embedding_{}.npy".format(n_i))
    save_file = os.path.join(savedir, "{}.npy".format(args.method))
    vis_file = os.path.join(savedir, "{}.png".format(args.method))
    np.save(emb_file, all)
    print("saved embeddings to {}".format(emb_file), flush=True)
    dimension_reduction(emb_file, save_file, method=args.method)
    plot_embeddings(save_file, n_i, vis_file, method=args.method)


def dimension_reduction(numpyfile, dstfile, method="PCA"):
    all = np.load(numpyfile)
    begin = time.time()
    print("conducting {} on data...".format(method), flush=True)
    if method == "T-SNE":
        all = TSNE(n_components=2).fit_transform(all)
    elif method == "PCA":
        all = PCA(n_components=2).fit_transform(all)
    else:
        raise NotImplementedError()
    print("done | {} ".format(sec2str(time.time() - begin)), flush=True)
    np.save(dstfile, all)
    print("saved {} embeddings to {}".format(method, dstfile), flush=True)


def plot_embeddings(numpyfile, n_v, out_file, method="PCA"):
    all = np.load(numpyfile)
    assert all.shape[1] == 2
    fig = plt.figure(clear=True)
    fig.suptitle("visualization of embeddings using {}".format(method))
    plt.scatter(all[:n_v, 0], all[:n_v, 1], s=2, c="red", label="image")
    plt.scatter(all[n_v::5, 0], all[n_v::5, 1], s=2, c="blue", label="caption")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(out_file)
    print("saved {} plot to {}".format(method, out_file), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--root_path", type=str, default="/groups1/gaa50131/datasets/MSCOCO")
    parser.add_argument("--vocab_path", type=str, default="captions_train2017.txt")
    parser.add_argument(
        "--method",
        type=str,
        default="PCA",
        help="Name of dimensionality reduction method, should be {T-SNE | PCA}",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="embedding",
        help="name of config, filename where to save",
    )

    # configurations of models
    parser.add_argument("--cnn_type", type=str, default="resnet152")
    parser.add_argument("--rnn_type", type=str, default="GRU")

    # training config
    parser.add_argument("--n_cpu", type=int, default=8)
    parser.add_argument("--emb_size", type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument(
        "--out_size", type=int, default=1024, help="embedding size for output vectors"
    )
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--no_cuda", action="store_true", help="disable gpu training")

    # hyperparams
    parser.add_argument(
        "--imsize_pre", type=int, default=256, help="to what size to crop the image"
    )
    parser.add_argument("--imsize", type=int, default=224, help="image size to resize on")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size. irrelevant")

    # retrieval config
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, will load model from there",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
