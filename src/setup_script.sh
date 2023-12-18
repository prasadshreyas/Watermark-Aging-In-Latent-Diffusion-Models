
#!/bin/bash

# Clone the Git repository
git clone https://github.com/facebookresearch/stable_signature.git

# Change directory
cd stable_signature

# Install requirements
pip install -r requirements.txt

# Create a models directory
mkdir -p models

# Download model files
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt -P models/
wget https://dl.fbaipublicfiles.com/ssl_watermarking/other_dec_48b_whit.torchscript.pt -P models/


# Install git lfs
git lfs install

# Clone the Git repository
git clone https://huggingface.co/stabilityai/sd-turbo

# Change directory
mkdir -p stable_signature/data
cd stable_signature/data


# Download COCO dataset
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip the files
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Remove the zip files
rm train2017.zip
rm val2017.zip
rm annotations_trainval2017.zip

# Move the files to the correct directory
mv train2017 stable_signature/data/
mv val2017 stable_signature/data/
mv annotations stable_signature/data/

