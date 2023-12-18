"""
Shreyas Prasad
CS 7180 - Advanced Perception
"""

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as img_transforms

class ImageProcessor:
    """
    A class for image processing tasks including transformations and PSNR calculation.

    Args:
        color_space (str): Color space for PSNR calculation ('standard', 'image', or 'custom').

    Attributes:
        standard_transform (transforms.Compose): Standard image transformations.
        normalize_standard (transforms.Normalize): Normalization for standard color space.
        unnormalize_standard (transforms.Normalize): Unnormalization for standard color space.
        normalize_image (transforms.Normalize): Normalization for custom color space.
        unnormalize_image (transforms.Normalize): Unnormalization for custom color space.
    """

    def __init__(self, color_space='standard'):
        self.color_space = color_space

        # Standard image transformations
        self.standard_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Normalization and unnormalization transformations
        self.normalize_standard = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.unnormalize_standard = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        self.normalize_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.unnormalize_image = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])

    def calculate_psnr(self, source_img, target_img):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

        Args:
            source_img (torch.Tensor): The source image.
            target_img (torch.Tensor): The target image.

        Returns:
            torch.Tensor: The PSNR value.
        """
        if self.color_space == 'standard':
            difference = torch.clamp(self.unnormalize_standard(source_img), 0, 1) - torch.clamp(self.unnormalize_standard(target_img), 0, 1)
        elif self.color_space == 'image':
            difference = torch.clamp(self.unnormalize_image(source_img), 0, 1) - torch.clamp(self.unnormalize_image(target_img), 0, 1)
        else:
            difference = source_img - target_img
        difference = 255 * difference
        difference = difference.view(-1, source_img.size(-3), source_img.size(-2), source_img.size(-1))
        psnr_value = 20 * np.log10(255) - 10 * torch.log10(torch.mean(difference ** 2, dim=(1, 2, 3)))
        return psnr_value

    def crop_center(self, image, target_scale):
        """
        Crop the center of an image.

        Args:
            image (PIL.Image or torch.Tensor): The input image.
            target_scale (float): The scale factor for cropping.

        Returns:
            PIL.Image or torch.Tensor: The cropped image.
        """
        scale_factor = np.sqrt(target_scale)
        new_size = [int(dim * scale_factor) for dim in image.size[::-1]]
        return img_transforms.center_crop(image, new_size)

    def resize_image(self, image, scale):
        """
        Resize an image with a given scale.

        Args:
            image (PIL.Image or torch.Tensor): The input image.
            scale (float): The scale factor for resizing.

        Returns:
            PIL.Image or torch.Tensor: The resized image.
        """
        scale_factor = np.sqrt(scale)
        new_size = [int(dim * scale_factor) for dim in image.size[::-1]]
        return img_transforms.resize(image, new_size)

    def rotate_image(self, image, angle):
        """
        Rotate an image by a given angle.

        Args:
            image (PIL.Image or torch.Tensor): The input image.
            angle (float): The angle of rotation in degrees.

        Returns:
            PIL.Image or torch.Tensor: The rotated image.
        """
        return img_transforms.rotate(image, angle)

    def adjust_brightness(self, image, brightness_factor):
        """
        Adjust the brightness of an image.

        Args:
            image (torch.Tensor): The input image.
            brightness_factor (float): The brightness adjustment factor.

        Returns:
            torch.Tensor: The adjusted image.
        """
        return self.normalize_image(img_transforms.adjust_brightness(self.unnormalize_image(image), brightness_factor))

    def adjust_contrast(self, image, contrast_factor):
        """
        Adjust the contrast of an image.

        Args:
            image (torch.Tensor): The input image.
            contrast_factor (float): The contrast adjustment factor.

        Returns:
            torch.Tensor: The adjusted image.
        """
        return self.normalize_image(img_transforms.adjust_contrast(self.unnormalize_image(image), contrast_factor))

    def adjust_saturation(self, image, saturation_factor):
        """
        Adjust the saturation of an image.

        Args:
            image (torch.Tensor): The input image.
            saturation_factor (float): The saturation adjustment factor.

        Returns:
            torch.Tensor: The adjusted image.
        """
        return self.normalize_image(img_transforms.adjust_saturation(self.unnormalize_image(image), saturation_factor))

    def adjust_hue(self, image, hue_factor):
        """
        Adjust the hue of an image.

        Args:
            image (torch.Tensor): The input image.
            hue_factor (float): The hue adjustment factor.

        Returns:
            torch.Tensor: The adjusted image.
        """
        return self.normalize_image(img_transforms.adjust_hue(self.unnormalize_image(image), hue_factor))

    def adjust_gamma(self, image, gamma, gain=1):
        """
        Adjust the gamma of an image.

        Args:
            image (torch.Tensor): The input image.
            gamma (float): The gamma value for adjustment.
            gain (float): The gain factor for adjustment.

        Returns:
            torch.Tensor: The adjusted image.
        """
        return self.normalize_image(img_transforms.adjust_gamma(self.unnormalize_image(image), gamma, gain))

    def adjust_sharpness(self, image, sharpness_factor):
        """
        Adjust the sharpness of an image.

        Args:
            image (torch.Tensor): The input image.
            sharpness_factor (float): The sharpness adjustment factor.

        Returns:
            torch.Tensor: The adjusted image.
        """
        return self.normalize_image(img_transforms.adjust_sharpness(self.unnormalize_image(image), sharpness_factor))

    def overlay_text(self, image, text='Lorem Ipsum'):
        """
        Overlay text on an image.

        Args:
            image (torch.Tensor): The input image.
            text (str): The text to overlay.

        Returns:
            torch.Tensor: The image with text overlay.
        """
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
    
        img_aug = torch.zeros_like(image, device=image.device)
        for ii, img in enumerate(image):
            pil_img = to_pil(self.unnormalize_image(img))
            img_aug[ii] = to_tensor(img_augment.overlay_text(pil_img, text=text))
        return self.normalize_image(img_aug)



    def jpeg_compress(self, image, quality_factor):
        """
        Compress an image using JPEG compression.

        Args:
            image (torch.Tensor): The input image.
            quality_factor (int): The quality factor for compression.

        Returns:
            torch.Tensor: The compressed image.
        """
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        img_aug = torch.zeros_like(image, device=image.device)
        for ii, img in enumerate(image):
            pil_img = to_pil(self.unnormalize_image(img))
            img_aug[ii] = to_tensor(img_augment.encoding_quality(pil_img, quality=quality_factor))
        return self.normalize_image(img_aug)
    
    def apply_transformations(self):
        """
        Apply image transformations for preprocessing before model inference.

        Returns:
        torchvision.transforms.Compose: The composed transformations.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    


    
    