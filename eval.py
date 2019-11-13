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
from utils import sec2str, collater
from model import ImageEncoder, CaptionEncoder
from vocab import Vocabulary


def retrieve_i2c(dset, v_dset, impath, imenc, transform, k=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im = Image.open(impath)
    print("-"*50)
    plt.title("source image")
    plt.imshow(np.asarray(im))
    plt.axis('off')
    plt.show(block=False)
    plt.show()
    im = transform(im).unsqueeze(0)
    begin = time.time()
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
    D, I = cpu_index.search(im, k)
    nnann = []
    for i in range(k):
        nnidx = I[0, i]
        ann_ids = [a for ids in dset.embedded["ann_id"] for a in ids]
        nnann_id = ann_ids[nnidx]
        nnann.append(nnann_id)
    anns = v_dset.coco.loadAnns(nnann)
    print("retrieval time {}".format(sec2str(time.time()-begin)), flush=True)
    print("-"*50)
    print("{} nearest neighbors of image:".format(k))
    v_dset.coco.showAnns(anns)
    print("-"*50)


def retrieve_c2i(dset, v_dset, caption, capenc, vocab, k=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    begin = time.time()
    print("-"*50)
    print("source caption: '{}'".format(caption), flush=True)
    cap = vocab.return_idx([caption])
    length = torch.tensor([torch.sum(torch.ne(cap, vocab.padidx)).item()]).to(device, dtype=torch.long)
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
    D, I = cpu_index.search(cap, k)
    print("retrieval time {}".format(sec2str(time.time()-begin)), flush=True)
    nnimid = []
    for i in range(k):
        nnidx = I[0, i]
        nnim_id = dset.embedded["img_id"][nnidx]
        nnimid.append(nnim_id)
    img = v_dset.coco.loadImgs(nnimid)
    print("-"*50)
    print("{} nearest neighbors of '{}'".format(k, caption))
    if k == 1:
        plt.figure(figsize=(8, 10))
        nnim = io.imread(img[0]['coco_url'])
        plt.imshow(nnim)
        plt.axis('off')
    elif k > 1:
        fig, axs = plt.subplots(1, k, figsize=(8*k, 10))
        fig.suptitle("retrieved {} nearest neighbors of '{}'".format(k, caption))
        for i in range(k):
            nnim = io.imread(img[i]['coco_url'])
            axs[i].imshow(nnim)
            axs[i].axis('off')
    else:
        raise
    plt.show(block=False)
    print("-"*50)
    plt.show()


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
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, collate_fn=collater)

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

    retrieve_i2c(dset, val_dset, args.image_path, imenc, transform)
    retrieve_c2i(dset, val_dset, args.caption, capenc, vocab)


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
    parser.add_argument('--emb_size', type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=1024, help="embedding size for output vectors")
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--no_cuda', action='store_true', help="disable gpu training")

    # hyperparams
    parser.add_argument('--imsize_pre', type=int, default=256, help="to what size to crop the image")
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
