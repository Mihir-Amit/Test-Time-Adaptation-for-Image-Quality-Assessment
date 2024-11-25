from __future__ import print_function, division
from scipy import stats
import torchvision.transforms as T
import copy
from torchvision import models
import os
import sys
from torch.utils.data.dataloader import default_collate
from torch.nn.functional import cosine_similarity
import math

from torch.autograd import Variable

import argparse
import data_loader
from SSHead import *
from rotation import *
from util import *
from tqdm import tqdm
from scipy.stats import spearmanr


import pandas as pd
import torch.utils.data as data
import os.path
import scipy.io
import csv
import cv2
from pandas_ods_reader import read_ods
from skimage.util import random_noise
import scipy.ndimage

parent_dir = os.path.abspath(os.path.join(__file__, "../../"))  # Go up two levels from 'MetaIQA/denoise.py'
scunet_dir = os.path.join(parent_dir, "Image_Denoising", "SCUNet") 
sys.path.append(scunet_dir)   

from scunet_denoising import *

from skimage.util import random_noise
from PIL import Image
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional as F


transform_LIVE = torchvision.transforms.Compose([
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.CenterCrop(size=224),
    # torchvision.transforms.RandomCrop(size=patch_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
])

transform_Koniq = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
                                    ])


def noisy(image, transform):
    """
    Adds random Gaussian noise to the input image and applies transformations.

    Args:
        image (PIL Image or np.ndarray): The input image (in PIL or NumPy format).
        transform (callable): The transformation function to apply to the image.

    Returns:
        tuple: A tuple of two noisy images (image2, image1), each transformed.
    """

    # Define two random noise levels
    sigma1 = 0.00005 + np.random.random() * 0.000001
    sigma2 = 0.00001 + np.random.random() * 0.000001   # low noise

    # Ensure image is in NumPy array format
    if isinstance(image, Image.Image):
        ab = np.array(image)
    else:
        ab = image

    # Convert to RGB if the image is not already in 3 channels
    if len(ab.shape) == 2:  # If grayscale, convert to RGB
        ab = np.stack([ab] * 3, axis=-1)

    # Add noise to the image
    noise = random_noise(ab, mode='gaussian', var=sigma1)
    image1 = Image.fromarray((noise * 255).astype('uint8'))
    image1 = transform(image1)

    noise = random_noise(ab, mode='gaussian', var=sigma2)
    image2 = Image.fromarray((noise * 255).astype('uint8'))
    image2 = transform(image2)

    return image2, image1



def compress(image,transform,root):

    sigma1 = 40 + np.random.random() * 20  # 40-60
    sigma2 = 80 + np.random.random() * 10  # 80-90

    try:
        image.save(root+"/Compressed_" + '1.jpg', optimize=True, quality=int(sigma1))
        image1 = Image.open(root+'/Compressed_1.jpg')
    except:
        image.save(root + "/Compressed_" + '1.bmp', optimize=True, quality=int(sigma1))
        image1 = Image.open(root + '/Compressed_1.bmp')
    image1 = transform(image1)

    try:
        image.save(root + "/Compressed_" + '2.jpg', optimize=True, quality=int(sigma2))
        image2 = Image.open(root+'/Compressed_2.jpg')
    except:
        image.save(root + "/Compressed_" + '2.bmp', optimize=True, quality=int(sigma2))
        image2 = Image.open(root + '/Compressed_2.bmp')
    image2 = transform(image2)

    return image1,image2



def denoise(data_dict, labels, root):
    img_list = data_dict['image']
    for i in range(img_list.shape[0]):
        if(labels[i] >= 35):
            image = denoise_image_direct(img_list[i])
            image = Image.fromarray(image)
            data_dict['image'][i] = transform_LIVE(image)
            data_dict['comp_high'][i], data_dict['comp_low'][i] = compress(image,transform_LIVE, root)
            data_dict['nos_low'][i], data_dict['nos_high'][i] = noisy(image,transform_LIVE)
    return data_dict


