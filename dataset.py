import os
import json as jsonmod

import numpy as np
import torch
import torchtext
import spacy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import nltk
from PIL import Image
from pycocotools.coco import COCO

from vocab import Vocabulary

sp = spacy.load("en_core_web_sm")


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, imgdir='train2017', jsonfile='annotations/captions_train2017.json', transform=None):
        """
        Args:
            root: root directory.
            json: coco annotation file path.
            transform: transformer for image.
        /home/seito/hdd/dsets/coco/train2017 """
        self.coco = COCO(os.path.join(root, jsonfile))
        self.img_dir = os.path.join(root, imgdir)

        self.imgids = self.coco.getImgIds()
        self.annids = [self.coco.getAnnIds(id) for id in self.imgids]
        self.transform = transform

    def __getitem__(self, index):

        img_id = self.imgids[index]
        ann_id = self.annids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        caption = [obj['caption'] for obj in self.coco.loadAnns(ann_id)]

        # restrict to 5 captions
        caption = caption[:5]
        ann_id = ann_id[:5]

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        return {"image": image, "caption": caption, "index": index, "img_id": img_id, "ann_id": ann_id}

    def __len__(self):
        return len(self.imgids)

class FlickrDataset(Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, jsonfile, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = json.load(open(jsonfile, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)

class EmbedDataset(Dataset):
    """Dataset to create when evaluating model"""

    def __init__(self, loader, image_model, caption_model, vocab, args):
        """
        Args:
            loader: DataLoader for validation images and captions
            model: trained model to evaluate
        """
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.embedded = {"image": [], "caption": [], "img_id": [], "ann_id": []}
        for data in loader:
            im = data["image"]
            caption = data["caption"]
            caption = [c for cap in caption for c in cap]
            cap = vocab.return_idx(caption)
            lengths = cap.ne(vocab.padidx).sum(dim=1)
            im = im.to(device)
            cap = cap.to(device)
            with torch.no_grad():
                emb_im = image_model(im)
                emb_cap = caption_model(cap, lengths)
            self.embedded["image"].append(emb_im.cpu().numpy())
            self.embedded["caption"].append(emb_cap.cpu().numpy())
            self.embedded["img_id"].extend(data["img_id"])
            self.embedded["ann_id"].extend(data["ann_id"])
        self.embedded["image"] = np.concatenate(self.embedded["image"], axis=0)
        self.embedded["caption"] = np.concatenate(self.embedded["caption"], axis=0)


    def __len__(self):
        return len(self.embedded["img_id"])

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    cocodset = CocoDataset(root="/ssd1/dsets/coco", transform=transform)
    print(cocodset[1])

