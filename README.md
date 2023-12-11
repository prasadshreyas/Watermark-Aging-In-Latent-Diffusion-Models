# README

This repository contains all the code and data required to reproduce the experiments in the paper "Watermark Aging of Images in Latent Diffusion Models".

Authors:

- Shreyas Prasad (<prasad.shre@northeastern.edu>)

## Links

<h3 align="center">
  <a href="">Video</a> |
  <a href="docs/slides.pdf">Slides</a>
</h3>

## How to use this repository

This repository is built originally on the DGX A100 machine, with the V100 GPUs. The code is tested on Ubuntu 18.04.5 LTS with CUDA toolkit 11.3 and Python 3.8.10 with a VSCode as the IDE.

## Setup

We first need to install the required dependencies. We recommend using a virtual environment to avoid conflicts with other packages.

``` bash
conda create -n watermark_aging python=3.8 -y && conda activate watermark_aging

conda install -c pytorch torchvision pytorch==1.12.0 cudatoolkit==11.3 -y

pip install -r requirements.txt
```

## Usage

### Background

Before we run the experiments for aging, we first need to set-up the Stable Signature pipeline.

**Stable Signature**

It has two steps:

1. Pre-training the Watermark Extractor
   1.  Uses the HiDDeN method to train a watermark extractor that embeds messages into images, resistant to common transformations.
   2.  Applies PCA whitening to enhance message recovery and reduce output biases.

2. Fine-tuning the Generative Model
   1. Fine-tunes the decoder of a Latent Diffusion Model (LDM) to embed a predefined message into images without altering the diffusion process.
   2. Binary Cross Entropy for message loss and Watson-VGG perceptual loss for minimal perceptual distortion, optimized using AdamW.


To perform step 1, we will checkpoint from the code [repository][Stable-Signature].




### Data

Exactly like the original Stable Signature pipeline, we need to have the following data:

The COCO dataset for fine-tuning the models. The dataset can be downloaded from the [official website](https://cocodataset.org/#download) or using the following command:

``` bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```





## Acknowledgements

Most of the code in this repository is based on the following papers and their code repositories:

1. [Stable-Signature](https://github.com/facebookresearch/stable_signature)
2. [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion)
3. [Perceptual-Similarity](https://github.com/SteffenCzolbe/PerceptualSimilarity)
4. [HiDDeN](https://github.com/ando-khachatryan/HiDDeN)