import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict

class ArcsinhTransform(torch.nn.Module):
    def __init__(self, factor=1.0):
        super(ArcsinhTransform, self).__init__()
        self.factor = factor

    def forward(self, input: torch.Tensor):
        return torch.arcsinh(input * self.factor)
    
class NumpyToTensor:
    def __init__(self, device):
        self.device = device

    def __call__(self, pic):
        if not(isinstance(pic, np.ndarray)):
            raise TypeError('pic should be ndarray. Got {}.'.format(type(pic)))

        # handle numpy array
        img = torch.from_numpy(pic)
        return img

# class MinMaxScaling:
#     def __call__(self, pic):
#         if not(isinstance(pic, torch.Tensor)):
#             raise TypeError('pic should be Tensor. Got {}.'.format(type(pic)))

#         min_val = torch.min(pic)
#         max_val = torch.max(pic)
#         pic = (pic - min_val) / (max_val - min_val)
#         return pic

class MinMaxNorm(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps  # To avoid division by zero

    def forward(self, tensor):
        min_val = torch.amin(tensor, dim=(1,2,3), keepdim=True)
        max_val = torch.amax(tensor, dim=(1,2,3), keepdim=True)
        norm_tensor = (tensor - min_val) / (max_val - min_val + self.eps)
        return norm_tensor

class Preprocess(nn.Module):
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, **kwargs) -> torch.Tensor:
        image = torch.from_numpy(kwargs['image'].astype('float32'))
        if len(kwargs) == 1:
            return image
        if len(kwargs) == 2:
            return image, torch.from_numpy(kwargs['label'].astype('int'))
    
class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, transform) -> None:
        super().__init__()
        self.transforms = transform

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> torch.Tensor:
        x_out = self.transforms(x)
        return x_out
    
class ContrastiveDataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, transform) -> None:
        super().__init__()
        self.transforms = transform

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x['background'] = self.transforms(x['background'])
        x['target'] = self.transforms(x['target'])
        return x
    
