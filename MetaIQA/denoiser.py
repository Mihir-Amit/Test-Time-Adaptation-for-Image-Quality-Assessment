from __future__ import print_function, division
from scipy import stats
import torchvision.transforms as T
import copy
from torchvision import models
import os
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

def denoise(img_list):
    