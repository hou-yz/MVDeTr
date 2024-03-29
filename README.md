# Multiview Detection with Shadow Transformer (and View-Coherent Data Augmentation) [[arXiv](https://arxiv.org/pdf/2108.05888.pdf)] [[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475310)]

```
@inproceedings{hou2021multiview,
  title={Multiview Detection with Shadow Transformer (and View-Coherent Data Augmentation)},
  author={Hou, Yunzhong and Zheng, Liang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia (MM ’21)},
  year={2021}
}
```


## Overview

We release the PyTorch code for **MVDeTr**, a state-of-the-art multiview pedestrian detector. Its superior performance should be credited to transformer architectures, updated loss terms, and view-coherent data augmentations. Moreover, MVDeTr is also very efficient and can be trained on a single RTX 2080TI. 
This repo also includes a simplified version of **[MVDet](https://github.com/hou-yz/MVDet)**, which also runs on a single RTX 2080TI. 

 
## Content
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Code Preparation](#code-preparation)
- [Training](#training)
    * [Architectures](#architectures)
    * [Loss terms](#loss-terms)
    * [Augmentations](#augmentations)


## MVDeTr Code
This repo is dedicated to the code for **MVDeTr**. 

<!-- ![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet") -->

## Dependencies
This code uses the following libraries
- python
- pytorch & tochvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia

## Data Preparation
By default, all datasets are in `~/Data/`. We use [MultiviewX](https://github.com/hou-yz/MultiviewX) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

Your `~/Data/` folder should look like this
```
Data
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```

## Code Preparation
Before running the code, one should go to ```multiview_detector/models/ops``` and run ```bash mask.sh``` to build the deformable transformer (forked from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)). 


## Training
In order to train classifiers, please run the following,
```shell script
python main.py -d wildtrack
python main.py -d multiviewx
``` 
This should automatically return evaluation results similar to the reported 91.5\% MODA on Wildtrack dataset and 93.7\% MODA on MultiviewX dataset. 


### Architectures
This repo supports multiple architecture variants. For MVDeTr, please specify ```--world_feat deform_trans```; for a similar fully convolutional architecture like [MVDet](https://github.com/hou-yz/MVDet), please specify ```--world_feat conv```. 

### Loss terms
This repo supports multiple loss terms. For the focal loss variant as in MVDeTr, please specify ```--use_mse 0```; for the MSE loss as in [MVDet](https://github.com/hou-yz/MVDet), please specify ```----use_mse 1```. 

### Augmentations
This repo includes support for view coherent data augmentation, which applies affine transformations onto the per-view inputs, and then invert the per-view feature maps to maintain multiview coherency. 

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRDo-mUXOWPd3Di4Q?e=monHmQ).
