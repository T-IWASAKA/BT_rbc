# -*- coding: utf-8 -*-
"""
Created 2025

image_aug

@author: iwasaka14
"""
## import ##
import random
import numpy as np
from typing import Tuple

import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter

from typing import Union


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1 #　初期値random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class RandomRotate(object):
    """Implementation of random rotation.
    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.
    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.
    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing 
            any artifacts.
    
    """

    def __init__(self, prob: float = 0.5, angle: Union[int, list, tuple] = None):
        self.prob = prob
        self.angle = [90] if angle is None else list(angle) 

    def __call__(self, sample):
        """Rotates the images with a given probability.
        Args:
            sample:
                PIL image which will be rotated.
        
        Returns:
            Rotated image or original image.
        """
        prob = np.random.random_sample()
        selected_angle = int(np.random.choice(self.angle))
        if prob < self.prob:
            sample =  transforms.functional.rotate(sample, selected_angle)
        return sample


def random_rotation_transform(
    rr_prob: float = 0.5,
    rr_degrees: Union[None, float, Tuple[float, float]] = 90,
    ) -> Union[RandomRotate, transforms.RandomApply]:
    if rr_degrees == 90:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=[90, 180, 270])
    else:
        # Random rotation with random angle defined by rr_degrees.
        return transforms.RandomApply([transforms.RandomRotation(degrees=rr_degrees)], p=rr_prob)


class SSLTransform:
    def __init__(self, transform=None, transform_prime=None, crop_size=None) -> None:
        """
        transform for self-supervised learning

        Parameters
        ----------
        transform: torchvision.transforms
            transform for the original image

        transform_prime: torchvision.transforms
            transform to be applied to the second
        
        """
        if crop_size is None:
            crop_size = 32
        else:
            pass
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.95, 1.0), interpolation=Image.BICUBIC),#
                transforms.RandomHorizontalFlip(p=0.5),#
                random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),## 初期値rr_degrees=[0,180], 90の倍数で回したい場合はrr_degrees=90
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], # default brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
                    p=0.8
                    ),#
                transforms.RandomGrayscale(p=0.2),# default 0.2
                GaussianBlur(p=0.2),#
                #Solarization(p=0),#
                transforms.ToTensor(),#
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))#
            ])
        else:
            self.transform = transform
        if transform_prime is None:
            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.95, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], # default brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
                    p=0.9
                    ),
                transforms.RandomGrayscale(p=0.2),# default 0.2
                GaussianBlur(p=0.2),# default 0.5
                #Solarization(p=0),# default 0.2
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))# default (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ])
        else:
            self.transform_prime = transform_prime
        

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2