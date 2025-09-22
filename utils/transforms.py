import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import io

class RandomJPEGCompression:
    def __init__(self, quality_min, quality_max):
        self.quality_min, self.quality_max = quality_min, quality_max
    def __call__(self, img):
        quality = np.random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        return Image.open(buffer)

class FFTTransform:
    def __call__(self, image):
        img_gray_tensor = transforms.functional.to_tensor(transforms.functional.to_grayscale(image))
        fft = torch.fft.fftshift(torch.fft.fft2(img_gray_tensor))
        magnitude, phase = torch.log1p(torch.abs(fft)), torch.angle(fft)
        fft_tensor = torch.stack([magnitude.squeeze(0), phase.squeeze(0)], dim=0)
        for i in range(fft_tensor.size(0)):
            min_val, max_val = fft_tensor[i].min(), fft_tensor[i].max()
            if max_val > min_val: fft_tensor[i] = (fft_tensor[i] - min_val) / (max_val - min_val)
        return fft_tensor

def get_transforms():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    spatial_transform = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
        transforms.RandomApply([RandomJPEGCompression(quality_min=50, quality_max=95)], p=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    return spatial_transform, FFTTransform()