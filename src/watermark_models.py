# Shreyas Prasad
# CS 7180 - Advanced Perception

"""
With the HiDDeN watermarking model, this code also provides all the utilities
to load and initialize the encoder and decoder models, and handle checkpoints.
This code is adapted from the PyTorch HiDDeN implementation by Ando Khachatryan
https://github.com/ando-khachatryan/HiDDeN/tree/master
"""


import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    A component in the Watermark network. Comprises a Convolution, Batch Normalization, and GELU activation
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Attributes:
        conv_sequence (nn.Sequential): Sequential module containing Convolution, Batch Normalization, and GELU activation layers
    
    Methods:
        forward(input_tensor): Performs forward pass through the ConvLayer
    
    """
    def __init__(self, in_channels, out_channels):

        super(ConvLayer, self).__init__()
        
        self.conv_sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.GELU()
        )

    def forward(self, input_tensor):
        """
        Performs forward pass through the ConvLayer
        
        Args:
            input_tensor (torch.Tensor): Input tensor to the ConvLayer
        
        Returns:
            torch.Tensor: Output tensor from the ConvLayer
        """
        return self.conv_sequence(input_tensor)

class WatermarkExtractor(nn.Module):
    """
    Extractor module. Takes an image with an embedded watermark and retrieves the watermark.

    Args:
        block_count (int): Number of convolutional blocks in the extractor.
        bit_count (int): Number of bits in the watermark.
        channel_count (int): Number of channels in the convolutional layers.
        redundancy_factor (int, optional): Redundancy factor for the watermark extraction. Defaults to 1.

    Attributes:
        conv_blocks (nn.Sequential): Sequential container for the convolutional layers.
        flatten (nn.Linear): Linear layer for flattening the extracted watermark.
        bit_count (int): Number of bits in the watermark.
        redundancy_factor (int): Redundancy factor for the watermark extraction.
    """

    def __init__(self, block_count, bit_count, channel_count, redundancy_factor=1):
        super(WatermarkExtractor, self).__init__()

        conv_blocks = [ConvLayer(3, channel_count)]
        for _ in range(block_count - 1):
            conv_blocks.append(ConvLayer(channel_count, channel_count))

        conv_blocks.append(ConvLayer(channel_count, bit_count*redundancy_factor))
        conv_blocks.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.flatten = nn.Linear(bit_count*redundancy_factor, bit_count*redundancy_factor)

        self.bit_count = bit_count
        self.redundancy_factor = redundancy_factor

    def forward(self, watermarked_img):
        """
        Forward pass of the WatermarkExtractor.

        Args:
            watermarked_img (torch.Tensor): Input watermarked image.

        Returns:
            torch.Tensor: Extracted watermark.
        """
        x = self.conv_blocks(watermarked_img) 
        x = x.squeeze(-1).squeeze(-1)
        x = self.flatten(x)

        x = x.view(-1, self.bit_count, self.redundancy_factor) 
        x = torch.sum(x, dim=-1) 

        return x

class WatermarkEmbedder(nn.Module):
    """
    Embeds a watermark into an image.

    Args:
        block_count (int): Number of convolutional blocks.
        bit_count (int): Number of bits in the watermark message.
        channel_count (int): Number of channels in the convolutional layers.
        apply_tanh (bool, optional): Whether to apply tanh activation to the output. Defaults to True.

    Attributes:
        conv_blocks (nn.Sequential): Sequential container for convolutional blocks.
        intermediate_layer (ConvLayer): Convolutional layer for intermediate processing.
        output_layer (nn.Conv2d): Convolutional layer for final output.
        apply_tanh (bool): Whether to apply tanh activation to the output.
        tanh (nn.Tanh): Tanh activation function.

    Methods:
        forward(images, messages): Forward pass of the watermark embedding process.

    Returns:
        watermarked_img (torch.Tensor): Watermarked image.
    """
    def __init__(self, block_count, bit_count, channel_count, apply_tanh=True):
        super(WatermarkEmbedder, self).__init__()
        conv_blocks = [ConvLayer(3, channel_count)]

        for _ in range(block_count-1):
            block = ConvLayer(channel_count, channel_count)
            conv_blocks.append(block)

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.intermediate_layer = ConvLayer(channel_count + 3 + bit_count, channel_count)

        self.output_layer = nn.Conv2d(channel_count, 3, kernel_size=1)

        self.apply_tanh = apply_tanh
        self.tanh = nn.Tanh()

    def forward(self, images, messages):
        """
        Forward pass of the watermark model.

        Args:
            images (torch.Tensor): Input images.
            messages (torch.Tensor): Watermark messages.

        Returns:
            torch.Tensor: Watermarked images.
        """
        messages = messages.unsqueeze(-1).unsqueeze(-1)
        messages = messages.expand(-1, -1, images.size(-2), images.size(-1))

        encoded_images = self.conv_blocks(images)

        concatenated = torch.cat([messages, encoded_images, images], dim=1)
        watermarked_img = self.intermediate_layer(concatenated)
        watermarked_img = self.output_layer(watermarked_img)

        if self.apply_tanh:
            watermarked_img = self.tanh(watermarked_img)

        return watermarked_img
    