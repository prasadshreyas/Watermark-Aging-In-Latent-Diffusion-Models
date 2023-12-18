"""
Shreyas Prasad
CS 7180 - Advanced Perception
"""

import io
import torch
import torchvision.transforms as transforms

class ImageFormatConverter:
    """
    A class for converting image tensors to different image formats.
    Supported formats: PNG, BMP, JPEG.
    """

    def __init__(self):
        """
        Initializes the ImageFormatConverter class.
        """
        self.supported_formats = ['PNG', 'BMP', 'JPEG']

    def convert_to_format(self, image_tensor, format_type='PNG'):
        """
        Convert an image tensor to a specified format.

        Args:
            image_tensor (torch.Tensor): The image tensor to convert. Should be a tensor in the format (C, H, W).
            format_type (str): The format type to convert to. Supported formats are 'PNG', 'BMP', 'JPEG'.

        Returns:
            bytes: A bytes representation of the image in the specified format.

        Raises:
            ValueError: If the specified format is not supported.
            TypeError: If the input image is not a torch.Tensor.
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("The image must be a torch.Tensor")

        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported formats are: {self.supported_formats}")

        try:
            pil_image = transforms.ToPILImage()(image_tensor).convert("RGB")
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format=format_type)
            img_byte_arr = img_byte_arr.getvalue()
            return img_byte_arr
        except Exception as e:
            raise RuntimeError(f"An error occurred during the format conversion: {e}")