"""
Utility functions for loading and preprocessing images for inference in the gradient descent chapter.
The functions in this file are not based on the book but rather copied from the code used to train the model.
"""

import io
import torch
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_image(image_path: Path, checkpoint) -> torch.Tensor:
    """Load and preprocess an image for inference."""
    # Define the same transformations used during training for consistency
    transform = transforms.Compose([
        transforms.Lambda(flatten_alpha_to_white),
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"]), antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply(
            [transforms.Lambda(lambda img: ImageOps.autocontrast(img, cutoff=1))],
            p=0.15,
        ),
        transforms.RandomApply(
            [transforms.Lambda(lambda img: jpeg_compress(img, quality=85))],
            p=0.2,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.35))],
            p=0.1,
        ),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Lambda(lambda x: add_sensor_noise(x, std=0.002, p=0.15)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # Load and preprocess the image and add a batch dimension
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def flatten_alpha_to_white(img: Image.Image) -> Image.Image:
    """Composite RGBA / LA images onto a white background."""
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")

def jpeg_compress(img: Image.Image, quality: int = 50) -> Image.Image:
    """Simulate JPEG compression."""
    with io.BytesIO() as buffer:
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()

def add_sensor_noise(tensor: torch.Tensor, std: float = 0.003, p: float = 0.25) -> torch.Tensor:
    """Add low-amplitude Gaussian sensor noise with probability p."""
    if std <= 0.0 or p <= 0.0:
        return tensor
    if torch.rand(1).item() >= p:
        return tensor
    return (tensor + std * torch.randn_like(tensor)).clamp(0.0, 1.0)