# VSE
Visual Semantic Embedding
---

Code is still in progress.


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
```
$ python -m spacy download en
```

### For Anaconda Users
- `environment.yml` file contains environment details for Anaconda users.
- run `conda env create -f environment.yml && conda activate mse` for simple use.

## Preparation of Dataset
download the COCO dataset.


## Description
This repository contains the implementation of visual-semantic embedding.
Training and Evaluation is done on the COCO dataset.
