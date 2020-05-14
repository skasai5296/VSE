import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import CocoDataset
from model import SPVSE
from train import validate
from utils import collater
from vocab import Vocabulary


def main():
    args = parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize(args.imsize_pre),
            transforms.CenterCrop(args.imsize),
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

    assert args.checkpoint is not None
    print("loading model and optimizer checkpoint from {} ...".format(args.checkpoint), flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _ = validate(1000, val_loader, model, vocab, args)


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
    parser.add_argument("--cnn_type", type=str, default="resnet152", help="architecture of cnn")
    parser.add_argument("--rnn_type", type=str, default="GRU", help="architecture of rnn")

    # training config
    parser.add_argument("--n_cpu", type=int, default=8, help="number of threads for dataloading")
    parser.add_argument("--emb_size", type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument(
        "--out_size", type=int, default=1024, help="embedding size for output vectors"
    )
    parser.add_argument("--max_len", type=int, default=30, help="max length of sentences")
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
