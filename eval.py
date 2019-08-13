import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import faiss
import skimage.io as io

from dataset import CocoDataset, EmbedDataset
from utils import sec2str, collater_eval
from model import ImageEncoder, CaptionEncoder
from vocab import Vocabulary


def retrieve_i2c(dset, v_dset, imenc, vocab, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    begin = time.time()
    print("-"*50)
    print("retrieving nearest caption to: '{}'".format(args.image_path), flush=True)
    im = Image.open(args.image_path)
    transform = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    im = transform(im).unsqueeze(0)
    with torch.no_grad():
        im = im.to(device)
        im = imenc(im)
    im = im.cpu().numpy()
    cap = dset.embedded["caption"]
    nd = cap.shape[0]
    d = cap.shape[1]
    cpu_index = faiss.IndexFlatIP(d)
    print("# captions: {}, dimension: {}".format(nd, d), flush=True)

    # im2cap
    cpu_index.add(cap)
    D, I = cpu_index.search(im, 5)
    nnidx = I[0, 0]
    nnann_id = dset.embedded["ann_id"][nnidx]
    anns = v_dset.coco.loadAnns(nnann_id)
    print("retrieval time {}".format(sec2str(time.time()-begin)), flush=True)
    v_dset.coco.showAnns(anns)
    print("-"*50)
    return


def retrieve_c2i(dset, v_dset, capenc, vocab, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    begin = time.time()
    print("-"*50)
    print("retrieving nearest image to: '{}'".format(args.caption), flush=True)
    cap = vocab.return_idx([args.caption])
    length = [torch.sum(torch.ne(cap, vocab.padidx)).item()]
    with torch.no_grad():
        cap = cap.to(device)
        cap = capenc(cap, length)
    cap = cap.cpu().numpy()
    im = dset.embedded["image"]
    nd = im.shape[0]
    d = im.shape[1]
    cpu_index = faiss.IndexFlatIP(d)
    print("# images: {}, dimension: {}".format(nd, d), flush=True)

    # cap2im
    cpu_index.add(im)
    D, I = cpu_index.search(cap, 5)
    nnidx = I[0, 0]
    nnim_id = dset.embedded["img_id"][nnidx]
    img = v_dset.coco.loadImgs(nnim_id)[0]
    nnim = io.imread(img['coco_url'])
    plt.title("nearest neighbor of '{}'".format(args.caption))
    plt.axis('off')
    plt.imshow(nnim)
    plt.show(block=False)
    print("retrieval time {}".format(sec2str(time.time()-begin)), flush=True)
    print("-"*50)
    plt.show()
    return


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    if args.dataset == 'coco':
        val_dset = CocoDataset(root=args.root_path, imgdir='val2017', jsonfile='annotations/captions_val2017.json', transform=transform, mode='all')
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, collate_fn=collater_eval)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(args.vocab_path)

    imenc = ImageEncoder(args.out_size, args.cnn_type)
    capenc = CaptionEncoder(len(vocab), args.emb_size, args.out_size, args.rnn_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    imenc = imenc.to(device)
    capenc = capenc.to(device)

    assert args.checkpoint is not None
    print("loading model and optimizer checkpoint from {} ...".format(args.checkpoint), flush=True)
    ckpt = torch.load(args.checkpoint)
    imenc.load_state_dict(ckpt["encoder_state"])
    capenc.load_state_dict(ckpt["decoder_state"])

    begin = time.time()
    dset = EmbedDataset(val_loader, imenc, capenc, vocab, args)
    print("database created | {} ".format(sec2str(time.time()-begin)), flush=True)

    retrieve_i2c(dset, val_dset, imenc, vocab, args)
    retrieve_c2i(dset, val_dset, capenc, vocab, args)


def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--root_path', type=str, default='/ssd1/dsets/coco')
    parser.add_argument('--vocab_path', type=str, default='captions_train2017.txt')

    # configurations of models
    parser.add_argument('--cnn_type', type=str, default="resnet18")
    parser.add_argument('--rnn_type', type=str, default="LSTM")

    # training config
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--emb_size', type=int, default=512, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=512, help="embedding size for output vectors")
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--no_cuda', action='store_true', help="disable gpu training")

    # hyperparams
    parser.add_argument('--imsize', type=int, default=224, help="image size to resize on")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size. irrelevant")

    # retrieval config
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint, will load model from there')
    parser.add_argument('--image_path', type=str, default='samples/sample1.jpg')
    parser.add_argument('--caption', type=str, default='the cat is walking on the street')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
