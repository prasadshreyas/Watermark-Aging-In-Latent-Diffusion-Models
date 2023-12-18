"""
Shreyas Prasad
CS 7180 - Advanced Perception

This code is a utility for loading watermark models from configuration and checkpoints.
"""


import torch
import torch.nn as nn
from watermark_models import WatermarkExtractor, WatermarkEmbedder

def get_watermark_extractor(bit_count, redundancy_factor=1, block_count=7, channel_count=64):
    """
    Returns a watermark extractor object.

    Parameters:
    - bit_count (int): The number of bits in the watermark.
    - redundancy_factor (int): The redundancy factor for error correction (default: 1).
    - block_count (int): The number of blocks in the watermark (default: 7).
    - channel_count (int): The number of channels in the watermark (default: 64).

    Returns:
    - extractor (WatermarkExtractor): The watermark extractor object.
    """
    extractor = WatermarkExtractor(block_count=block_count, bit_count=bit_count, channel_count=channel_count, redundancy_factor=redundancy_factor)
    return extractor

def get_watermark_extractor_ckpt(ckpt_path):
    """
    Retrieves the checkpoint of the watermark extractor from the given checkpoint path.

    Parameters:
        ckpt_path (str): The path to the checkpoint file.

    Returns:
        dict: The checkpoint of the watermark extractor.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    extractor_ckpt = {k.replace('module.', '').replace('extractor.', ''): v for k, v in ckpt['encoder_decoder'].items() if 'extractor' in k}
    return extractor_ckpt

def get_watermark_embedder(bit_count, block_count=4, channel_count=64):
    """
    Returns a watermark embedder object.

    Parameters:
    - bit_count (int): The number of bits to embed in the watermark.
    - block_count (int, optional): The number of blocks in the embedder. Default is 4.
    - channel_count (int, optional): The number of channels in each block. Default is 64.

    Returns:
    - embedder (WatermarkEmbedder): The watermark embedder object.
    """
    embedder = WatermarkEmbedder(block_count=block_count, bit_count=bit_count, channel_count=channel_count)
    return embedder

def get_watermark_embedder_ckpt(ckpt_path):
    """
    Retrieves the checkpoint of the watermark embedder from the given checkpoint path.

    Args:
        ckpt_path (str): The path to the checkpoint file.

    Returns:
        dict: The checkpoint of the watermark embedder.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    embedder_ckpt = {k.replace('module.', '').replace('embedder.', ''): v for k, v in ckpt['encoder_decoder'].items() if 'embedder' in k}
    return embedder_ckpt