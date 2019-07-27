import sys, os
sys.path.append(os.pardir)
import time
import json
import pickle

import torch
import torchtext
import spacy

from utils import sec2str

sp = spacy.load('en_core_web_sm')


class Vocabulary():
    def __init__(self, min_freq=5, max_len=30):
        self.min_freq = min_freq
        self.max_len = max_len

    """
    build vocabulary from textfile.
    """
    def load_vocab(self, textfile):
        before = time.time()
        print("building vocabulary...", flush=True)
        self.text_proc = torchtext.data.Field(sequential=True, init_token="<bos>", eos_token="<eos>", lower=True, fix_length=self.max_len, tokenize="spacy", batch_first=True)
        with open(textfile, 'r') as f:
            sentences = f.readlines()
        sent_proc = list(map(self.text_proc.preprocess, sentences))
        self.text_proc.build_vocab(sent_proc, min_freq=self.min_freq)
        self.len = len(self.text_proc.vocab)
        print("done building vocabulary, minimum frequency is {} times".format(self.min_freq), flush=True)
        print("{} | # of words in vocab: {}".format(sec2str(time.time() - before), self.len), flush=True)

    # sentence_batch: list of str
    # return indexes of sentence batch as torch.LongTensor
    def return_idx(self, sentence_batch):
        out = []
        preprocessed = list(map(self.text_proc.preprocess, sentence_batch))
        out = self.text_proc.process(preprocessed)
        return out

    # return sentence batch from indexes from torch.LongTensor
    def return_sentences(self, ten):
        if isinstance(ten, torch.Tensor):
            ten = ten.tolist()
        out = []
        for idxs in ten:
            tokenlist = [self.text_proc.vocab.itos[idx] for idx in idxs]
            out.append(" ".join(tokenlist))
        return out

    def __len__(self):
        return self.len

# build caption txt file from coco annotation json file
def cococap2txt(jsonfile, dst):
    sentences = []
    with open(jsonfile, 'r') as f:
        alldata = json.load(f)
    for ann in alldata["annotations"]:
        sentences.append(ann["caption"].strip())
    with open(dst, 'w+') as f:
        f.write("\n".join(sentences))

# for debugging
if __name__ == '__main__':

    file = "/home/seito/hdd/dsets/coco/annotations/captions_train2017.json"
    dest = "captions_train2017.txt"
    # first time only
    if not os.path.exists(dest):
        cococap2txt(file, dest)
    vocab = Vocabulary()
    vocab.load_vocab(dest)
    sentence = ["The cat and the hat sat on a mat."]
    ten = vocab.return_idx(sentence)
    print(ten)
    sent = vocab.return_sentences(ten)
    print(sent)
