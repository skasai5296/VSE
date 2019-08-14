# VSE: Visual Semantic Embedding in PyTorch

## Description
This repository contains the implementation of visual-semantic embedding.  
Training and evaluation is done on the MSCOCO dataset.  


## Requirements (libraries)
```
python>=3.7
numpy
matplotlib
pytorch>=1.1.0
torchvision
Pillow
faiss-cpu (for nearest neighbor search)
accimage (optional, for fast loading of images)
torchtext (for vocabulary)
spacy (for spacy tokenizer)
```

Run the below command before training.
```bash
$ python -m spacy download en
```

### For Anaconda Users
- `environment.yml` file contains environment details for Anaconda users.
- run `conda env create -f environment.yml && conda activate mse` for simple use.

## Preparation of Dataset
Go to the directory where the data should be and run `download_coco.sh`.  
This directory would be denoted `$ROOTPATH`.

## Training
```bash
$ python train.py --root_path $ROOTPATH
```

## Evaluation, Visualization
```bash
$ python eval.py --root_path $ROOTPATH --checkpoint hogehoge.ckpt --image_path $IMAGE --caption $CAPTION
```
`$IMAGE` denotes the path to reference image. Defaults to `samples/sample1.jpg`.  
`$CAPTION` denotes the reference caption. Defaults to `"the cat is walking on the street"`  
Retrieval is done on MSCOCO validation set.


## TODO
- [ ] add Flickr8k
- [ ] add Flickr30k
- [ ] clean up validation
- [ ] find optimal hyperparams
