# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter

#from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import deque
import itertools

from .RBC_loader import SmearDataset_RBC, Smear_tiff

from typing import Union
import sys

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


class Dataset_SSL(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, mydataset, transform):
        if transform is None:
            raise ValueError('!! Give transform !!')
        self.transform = [transform]
        self.input = [mydataset[0][i][0] for i in range(len(mydataset[0]))]
        self.datanum = len(self.input)
        self.transform_totensor = transforms.Compose([
            transforms.ToTensor(),  # [H, W, C] を [C, H, W] に変換
        ])

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        input = Image.fromarray(self.input[idx]) # 上記判定が不要のため
        t = self.transform
        y1, y2 = t[0](input)
        return y1, y2
    

def prep_dataset(tif_file, patch_size=1024, goal=10000, splitn=1, check_ditect=True, split_sample=False):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()
    buffer = patch_size/2

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()

    # ペアを順番に処理
    n = 0 # errorの判定に使用
    with tqdm(total=goal, desc="total rbc") as pbar:
        for xy in loc_pairs:
            x = int(xy[0]*patch_size + buffer)
            y = int(xy[1]*patch_size + buffer)
            isolated_centroids, isolated_areasize = dat_smear.ditect_rbc(patch_size=patch_size, loc=(x, y), rbc_radius=60)
            if isolated_centroids is not None: # Noneを返したときはエラーなので避ける
                rbc_lst = dat_smear.get_rbcimage(isolated_centroids, isolated_areasize, loc=(x, y))
                total_image.extend(rbc_lst)
                pbar.update(len(rbc_lst))
                if len(total_image) > goal:
                    print("The goal has been reached.")
                    if check_ditect:
                        image_array = np.array(dat_smear.get_area(patch_size=patch_size, loc=(x, y)), dtype=np.uint8)
                        plt.scatter(isolated_centroids[:,0],isolated_centroids[:,1],s=50, marker='h',c='orangered')
                        plt.imshow(image_array)#これは縦横 (y, x)
                        plt.show()
                    break
            else:
                dat_smear = Smear_tiff(tif_file)
                pbar.update(0)
                n = 1
                continue

    total_image = list(total_image)
    total_image = total_image[0:goal]

    if check_ditect:
        show_get_img(total_image)

    if split_sample:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC(total_image)]

    if n == 1:
        print("This file contains corrupted regions.")

    #print("Number of  datasets :", len(my_datasets))
    #print("Number of images per dataset :", len(my_datasets[0]))

    return my_datasets


def show_get_img(total_image):
    num = len(total_image)
    fig = plt.figure(figsize=(8, 2*(((num-1)//4)+1)))
    n = 1
    for img in random.sample(total_image, len(total_image)): # ランダムに取り出したい
        ax = fig.add_subplot((((num-1)//4)+1), 4, n)
        ax.imshow(img)
        if n == 16:
            plt.tight_layout()
            plt.show()
            break
        n = n + 1


def prep_smeardataset(image_path, ssl_transform=None) -> torch.utils.data.Dataset:
    """
    prepare dataset using ImageFolder
    
    Parameters
    ----------
    image_path: list
        the path to the image folder
    
    transform: a list of transform functions
        each function should return torch.tensor by __call__ method
    
    ssl_transform=None: a list of ssl transform functions

    img_type: Type of image to be detected
        wbc: region containing only white blood cells
        patches: region that cleared qc of background percentage
    
    """
    if type(image_path) == str:
        image_paths = [image_path]
    elif type(image_path) == list:
        image_paths = image_path
    
    # ここでtrainとtestをスライドから分けるようにする
    N = len(image_paths)
    thresh = int(0.8*N) - 1 # 何となくこの数, ここは後から変える！！
    for n, path in enumerate(image_paths):
        smeardataset = prep_dataset(path, patch_size=1024, goal=2000, splitn=1, check_ditect=False, split_sample=False) #数を外からいじれるようにしたい
        mydataset = Dataset_SSL(smeardataset, ssl_transform)

        if n == 0:
            train_dataset = mydataset
        elif n <= thresh:
            train_dataset = data.ConcatDataset([train_dataset, mydataset])
        elif n == thresh+1:
            test_dataset = mydataset
        elif n > thresh+1:
            test_dataset = data.ConcatDataset([test_dataset, mydataset])

    print("===============================================================================")
    print("train:test =", str(len(train_dataset)),":", str(len(test_dataset)))

    return train_dataset, test_dataset


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )    
    return loader


def prep_smeardata(
    image_path=(None, None), batch_size:int=0,
    transform=(None, None), ssl_transform=None, 
    shuffle=(True, False),# デフォルトshuffle=(True, False)
    num_workers:int=2, pin_memory:bool=True, 
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    prepare train and test loader from data
    
    Parameters
    ----------
    image_path: (str, str)
        the path to the training and test image folders, respectively
            
    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list

    ssl_transform: a list of ssl transform functions
    
    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively
    
    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing

    """
    # check transform
    if transform[0] is None:
        transform = _default_transform()
    # dataset and dataloader preparation

    if ssl_transform is not None:
        train_dataset, test_dataset = prep_smeardataset(image_path[0], ssl_transform=ssl_transform)
        classes = [train_dataset[i][1] for i in range(len(train_dataset))] + [test_dataset[i][1] for i in range(len(test_dataset))]
        train_loader = prep_dataloader(
            train_dataset, batch_size, shuffle[0], num_workers, pin_memory
            )
        test_loader = prep_dataloader(
            test_dataset, batch_size, shuffle[1], num_workers, pin_memory
            )    

    else:
        raise ValueError("!! Give ssl_transform !!")
        
    return train_loader, test_loader, classes

def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _default_transform():
    """ return default transforms """
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (32, 32), scale=(0.8, 1.0),
                ratio=(0.75, 1.3333), interpolation=2
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return train_transform, test_transform






def random_dataset(tif_file, patch_size=80, goal=10000, splitn=1, check_ditect=True, split_sample=False):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()

    # ペアを順番に処理
    with tqdm(total=goal, desc="total img") as pbar:
        for xy in loc_pairs:
            x = int(xy[0]*patch_size)
            y = int(xy[1]*patch_size)
            image_array = np.array(dat_smear.get_area(patch_size=patch_size, loc=(x, y)))[:,:,:3]
            total_image.extend([image_array])
            pbar.update(1)
            if len(total_image) > goal:
                break
    
    total_image = list(total_image)
    total_image = total_image[0:goal]
    if check_ditect:
        show_get_img(total_image)

    if split_sample:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC(total_image)]

    print("Number of  datasets :", len(my_datasets))
    print("Number of images per dataset :", len(my_datasets[0]))

    return my_datasets


def background_dataset(tif_file, patch_size=1600, goal=10000, splitn=1, bg_p=0.01, check_ditect=True, split_sample=False):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()

    # ペアを順番に処理
    with tqdm(total=goal, desc="total bg") as pbar:
        for xy in loc_pairs:
            x = int(xy[0]*patch_size)
            y = int(xy[1]*patch_size)
            background_lst = dat_smear.get_background(patch_size=patch_size, loc=(x, y), rbc_size=80, bg_p=bg_p)
            total_image.extend(background_lst)
            pbar.update(len(background_lst))
            if len(total_image) > goal:
                break

    total_image = list(total_image)
    total_image = total_image[0:goal]
    if check_ditect:
        show_get_img(total_image)

    if split_sample:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC(total_image)]

    #print("Number of  datasets :", len(my_datasets))
    #print("Number of images per dataset :", len(my_datasets[0]))

    return my_datasets

def rbc_dataset(tif_file, patch_size=1024, goal=10000, splitn=1, check_ditect=True, split_sample=False):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()
    buffer = patch_size/2

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()

    # ペアを順番に処理
    n = 0 # errorの判定に使用
    with tqdm(total=goal, desc="total rbc", file=sys.stderr) as pbar:
        for loc_n, xy in enumerate(loc_pairs):
            if loc_n == len(loc_pairs) - 1:
                print("All regions have been searched.")
            x = int(xy[0]*patch_size + buffer)
            y = int(xy[1]*patch_size + buffer)

            isolated_centroids, isolated_areasize = dat_smear.ditect_rbc(patch_size=patch_size, loc=(x, y), rbc_radius=60)
            if isolated_centroids is not None: # Noneを返したときはエラーなので避ける
                rbc_lst = dat_smear.get_rbcimage(isolated_centroids, isolated_areasize, loc=(x, y))
                total_image.extend(rbc_lst)
                pbar.update(len(rbc_lst))
                if len(total_image) > goal:
                    print("The goal has been reached.")
                    if check_ditect:
                        image_array = np.array(dat_smear.get_area(patch_size=patch_size, loc=(x, y)), dtype=np.uint8)
                        plt.scatter(isolated_centroids[:,0],isolated_centroids[:,1],s=50, marker='h',c='orangered')
                        plt.imshow(image_array)#これは縦横 (y, x)
                        plt.show()
                    break
            else:
                dat_smear = Smear_tiff(tif_file)
                pbar.update(0)
                n = 1
                continue


    total_image = list(total_image)
    total_image = total_image[0:goal]

    if check_ditect:
        show_get_img(total_image)

    if split_sample:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC(total_image)]

    if n == 1:
        print("This file contains corrupted regions.")

    #print("Number of  datasets :", len(my_datasets))
    #print("Number of images per dataset :", len(my_datasets[0]))
    print("\n")

    return my_datasets