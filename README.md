# README

This repository contains all the code and data required to reproduce the experiments in the paper "Watermark Aging of Images in Latent Diffusion Models".

Authors:

- Shreyas Prasad (<prasad.shre@northeastern.edu>)

## Links

<h3 align="center">
  <a href="https://youtu.be/emZZCZVmAjM">Video</a>
  <a href="Watermark-Aging-Presentation.pdf">Slides</a>
   <a href="report.pdf">Report</a>
</h3>

---


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

#### Data

Exactly like the original Stable Signature pipeline, we need to have the following data: The COCO dataset for fine-tuning the models. The dataset can be downloaded from the [official website](https://cocodataset.org/#download).

Before we run the experiments for aging, we first need to set-up the Stable Signature pipeline.

**Stable Signature**

It has two steps:

1. Pre-training the Watermark Extractor
   1. Uses the HiDDeN method to train a watermark extractor that embeds messages into images, resistant to common transformations.
   2. Applies PCA whitening to enhance message recovery and reduce output biases.

2. Fine-tuning the Generative Model
   1. Fine-tunes the decoder of a Latent Diffusion Model (LDM) to embed a predefined message into images without altering the diffusion process.
   2. Binary Cross Entropy for message loss and Watson-VGG perceptual loss for minimal perceptual distortion, optimized using AdamW.


To perform step 1 and download the required data, run the following command:

``` bash
./src/setup_script.sh
```

This takes around 20 minutes to run on a DGX2 machine.

What this does is;

- Installs the required dependencies for the code to run.
- Downloads the finetuning code for any LDM model.
- Downloads the watermark extractor model with PCA whitening.
- Downloads the COCO dataset for fine-tuning the LDM model.


Fortunately, we do not need to run the pre-training step as the pre-trained model is already available. We only need to fine-tune the LDM model on the COCO dataset.

  

#### Pre-training the Watermark Extractor

To pre-train the watermark extractor, run the following command:

``` bash
python src/train_watermark_extractor.py --train_dir data/train --val_dir data/test
```

To perform step 2, we will fine tune an LDM model on the COCO dataset according to the paper.

The Original paper uses a 2.2B parameter LDM model that is internally trained on a dataset of 330M licensed image-text pairs. Since, we do not have access to this model, we will use the COCO dataset to fine-tune a pre-trained LDM model on the COCO dataset.

we will use Stability AI's newest Stable Diffusion Turbo (SD-Turbo) model which is a 3.1B parameter model. We will adapt the code from the HuggingFace library.

This will download the entire model with the weights and the tokenizer.

First clone the repository:

``` bash
git lfs install
git clone https://huggingface.co/stabilityai/sd-turbo
```

Specifically, sd-turbo only comes with safetensors, so we will need to use a custom model to adapt the code to our needs as the stable-signature code expects model checkpoints to be `*.ckpt` files.

We will now use the `./src/safetensor_to_ckpt.py` script to convert the safetensor checkpoint to a ckpt checkpoint.


To fine-tune the model, run the following command:

``` bash
python stable_signature/finetune_ldm_decoder.py    --num_keys 1 \
   --ldm_config stable_signature/sd-turbo/vae/config.json \
   --ldm_ckpt stable_signature/sd-turbo/v1.ckpt \
   --msg_decoder_path stable_signature/models/dec_48b_whit.torchscript.pt \
   --train_dir data/train \
   --val_dir data/test
```

This will save the fine-tuned model in the `ckpts/` folder.


### Report and Results

For evaluating the results, run the following command:
Check the `evals.py` file for more details.

``` bash
python src/evals.py
```
Note: For Watermark Aging, Use `params.attack_mode == 'aging'` in the `evals.py` file.

## Acknowledgements

[Stable-Signature]: https://github.com/facebookresearch/stable_signature

[Perceptual-Similarity]: https://github.com/SteffenCzolbe/PerceptualSimilarity

[HiDDeN]: https://github.com/ando-khachatryan/HiDDeN


Most of the code in this repository is based on the following papers and their code repositories:

- Stable Signature: github.com/facebookresearch/stable_signature
- Perceptual Similarity: github.com/SteffenCzolbe/PerceptualSimilarity
- HiDDeN: github.com/ando-khachatryan/HiDDeN


