# README

This repository contains all the code and data required to reproduce the experiments in the paper "Watermark Aging of Images in Latent Diffusion Models".

This repository is built originally on the DGX A100 machine, with the V100 GPUs. The code is tested on Ubuntu 18.04.5 LTS with CUDA toolkit 11.3 and Python 3.8.10 with a VSCode as the IDE.

## Installation
We first need to install the required dependencies. We recommend using a virtual environment to avoid conflicts with other packages.
``` bash
conda create -n watermark_aging python=3.8 -y && conda activate watermark_aging

conda install -c pytorch torchvision pytorch==1.12.0 cudatoolkit==11.3 -y

pip install -r requirements.txt
```

## How to use this repository


## Citation
Most of the code in this repository is based on the following paper and their code repositories:

``` bibtex
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={ICCV},
  year={2023}
}
```
