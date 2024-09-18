import os
import torch
import numpy as np
import cv2
from skimage.util import random_noise
from PIL import Image
import csv
import pandas as pd
from pandas_ods_reader import read_ods

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class IQADataset(torch.utils.data.Dataset):

    def __init__(self, config, db_path, txt_file_name, scale_1, scale_2, transform, train_mode, scene_list,
                 train_size=0.8):

        super(IQADataset, self).__init__()

        self.db_path = db_path

        self.txt_file_name = txt_file_name

        self.scale_1 = scale_1

        self.scale_2 = scale_2

        self.transform = transform

        self.train_mode = train_mode

        self.scene_list = scene_list

        self.train_size = train_size

        self.config = config

        self.data_dict = IQADatalist(

            db_path=self.db_path,

            txt_file_name=self.txt_file_name,

            train_mode=self.train_mode,

            scene_list=self.scene_list,

            train_size=self.train_size

        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])

    def __len__(self):

        return self.n_images

    def __getitem__(self, idx):

        # d_img_org: H x W x C

        d_img_name = self.data_dict['d_img_list'][idx]

        d_img_org = cv2.imread(os.path.join((self.db_path), d_img_name), cv2.IMREAD_COLOR)

        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)

        d_img_org = cv2.resize(d_img_org, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        d_img_org = np.array(d_img_org).astype('float32') / 255

        h, w, c = d_img_org.shape

        d_img_scale_1 = cv2.resize(d_img_org, dsize=(self.scale_1, int(h * (self.scale_1 / w))),
                                   interpolation=cv2.INTER_CUBIC)

        d_img_scale_2 = cv2.resize(d_img_org, dsize=(self.scale_2, int(h * (self.scale_2 / w))),
                                   interpolation=cv2.INTER_CUBIC)

        d_img_scale_2 = d_img_scale_2[:160, :, :]

        d_img_org_copy = np.copy(d_img_org)

        d_img_scale_1_copy = np.copy(d_img_scale_1)

        d_img_scale_2_copy = np.copy(d_img_scale_2)

        score = self.data_dict['score_list'][idx]

        sample = {'d_img_org': d_img_org, 'd_img_scale_1': d_img_scale_1, 'd_img_scale_2': d_img_scale_2,

                  'score': score}

        data_dict = {}

        if self.transform:
            data_dict['image'] = self.transform(sample)

        if self.config.rank:
            d_img_org_copy1 = np.copy(d_img_org_copy)

            d_img_scale_1_copy1 = np.copy(d_img_scale_1_copy)

            d_img_scale_2_copy1 = np.copy(d_img_scale_2_copy)

            dish = {'d_img_org': d_img_org_copy1, 'd_img_scale_1': d_img_scale_1_copy1,

                    'd_img_scale_2': d_img_scale_2_copy1,

                    'score': score}

            data_dict['comp_high'], data_dict['comp_low'] = self.compress(dish, self.transform, self.db_path,

                                                                          d_img_name)

            d_img_org_copy1 = np.copy(d_img_org_copy)

            d_img_scale_1_copy1 = np.copy(d_img_scale_1_copy)

            d_img_scale_2_copy1 = np.copy(d_img_scale_2_copy)

            dish = {'d_img_org': d_img_org_copy1, 'd_img_scale_1': d_img_scale_1_copy1,

                    'd_img_scale_2': d_img_scale_2_copy1,

                    'score': score}

            data_dict['nos_low'], data_dict['nos_high'] = self.noisy(dish, self.transform)

        if self.config.comp:
            d_img_org_copy1 = np.copy(d_img_org_copy)

            d_img_scale_1_copy1 = np.copy(d_img_scale_1_copy)

            d_img_scale_2_copy1 = np.copy(d_img_scale_2_copy)

            dish = {'d_img_org': d_img_org_copy1, 'd_img_scale_1': d_img_scale_1_copy1,

                    'd_img_scale_2': d_img_scale_2_copy1,

                    'score': score}

            data_dict['comp_high'], data_dict['comp_low'] = self.compress(dish, self.transform, self.db_path,

                                                                          d_img_name)

        if self.config.nos:
            d_img_org_copy1 = np.copy(d_img_org_copy)

            d_img_scale_1_copy1 = np.copy(d_img_scale_1_copy)

            d_img_scale_2_copy1 = np.copy(d_img_scale_2_copy)

            dish = {'d_img_org': d_img_org_copy1, 'd_img_scale_1': d_img_scale_1_copy1,

                    'd_img_scale_2': d_img_scale_2_copy1,

                    'score': score}

            data_dict['nos_high'], data_dict['nos_low'] = self.noisy(dish, self.transform)  # correct

        if self.config.contrastive:
            d_img_org_copy1 = np.copy(d_img_org_copy)

            d_img_scale_1_copy1 = np.copy(d_img_scale_1_copy)

            d_img_scale_2_copy1 = np.copy(d_img_scale_2_copy)

            dish = {'d_img_org': d_img_org_copy1, 'd_img_scale_1': d_img_scale_1_copy1,

                    'd_img_scale_2': d_img_scale_2_copy1,

                    'score': score}

            data_dict['image1'] = self.transform(dish)

            d_img_org_copy1 = np.copy(d_img_org_copy)

            d_img_scale_1_copy1 = np.copy(d_img_scale_1_copy)

            d_img_scale_2_copy1 = np.copy(d_img_scale_2_copy)

            dish = {'d_img_org': d_img_org_copy1, 'd_img_scale_1': d_img_scale_1_copy1,

                    'd_img_scale_2': d_img_scale_2_copy1,

                    'score': score}

            data_dict['image2'] = self.transform(dish)

        if self.config.online:

            data_dict['online'] = {}

            data_dict['online']['d_img_org'] = []

            data_dict['online']['d_img_scale_1'] = []

            data_dict['online']['d_img_scale_2'] = []

            for i in range(32):
                d_img_org_copy1 = np.copy(d_img_org_copy)

                d_img_scale_1_copy1 = np.copy(d_img_scale_1_copy)

                d_img_scale_2_copy1 = np.copy(d_img_scale_2_copy)

                dish = {'d_img_org': d_img_org_copy1, 'd_img_scale_1': d_img_scale_1_copy1,

                        'd_img_scale_2': d_img_scale_2_copy1,

                        'score': score}

                new = self.transform(dish)

                data_dict['online']['d_img_org'].append(new['d_img_org'])

                data_dict['online']['d_img_scale_1'].append(new['d_img_scale_1'])

                data_dict['online']['d_img_scale_2'].append(new['d_img_scale_2'])

        return data_dict

    def noisy(self, sample, transform):

        sigma1 = 0.05 + np.random.random() * 0.05  # 0.1+ np.random.random() * 0.0001

        sigma2 = 0.005 + np.random.random() * 0.005  # 0.0005+ np.random.random() * 0.0001

        image1 = sample.copy()

        image2 = sample.copy()

        for key in sample.keys():
            image1[key] = random_noise(sample[key], mode='gaussian', var=sigma1)

            image1[key] = np.array(image1[key]).astype('float32')

            image2[key] = random_noise(sample[key], mode='gaussian', var=sigma2)

            image2[key] = np.array(image2[key]).astype('float32')

        image1 = transform(image1)

        image2 = transform(image2)

        return image1, image2

    def compress(self, sample, transform, root, path):

        image1 = {}

        image2 = {}

        image = pil_loader(path)

        for key, img in sample.items():

            if key == 'd_img_org':
                sigma1 = 40 + np.random.random() * 20  # 40-60

                sigma2 = 80 + np.random.random() * 10  # 80-90

                image.save(root + "/Compressed_" + '1.bmp', quality=int(sigma1))

                image1[key] = cv2.imread(root + '/Compressed_1.jpg', cv2.IMREAD_COLOR)

                image1[key] = cv2.cvtColor(image1[key], cv2.COLOR_BGR2RGB)

                image1[key] = cv2.resize(image1[key], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

                image1[key] = np.array(image1[key]).astype('float32') / 255

                image.save(root + "/Compressed_" + '2.bmp', quality=int(sigma2))

                image2[key] = cv2.imread(root + '/Compressed_1.jpg', cv2.IMREAD_COLOR)

                image2[key] = cv2.cvtColor(image2[key], cv2.COLOR_BGR2RGB)

                image2[key] = cv2.resize(image2[key], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

                image2[key] = np.array(image2[key]).astype('float32') / 255

        h, w, c = image1['d_img_org'].shape

        image1['d_img_scale_1'] = cv2.resize(image1['d_img_org'], dsize=(self.scale_1, int(h * (self.scale_1 / w))),

                                             interpolation=cv2.INTER_CUBIC)

        image1['d_img_scale_2'] = cv2.resize(image1['d_img_org'], dsize=(self.scale_2, int(h * (self.scale_2 / w))),

                                             interpolation=cv2.INTER_CUBIC)

        image1['d_img_scale_2'] = image1['d_img_scale_2'][:160, :, :]

        image2['d_img_scale_1'] = cv2.resize(image2['d_img_org'], dsize=(self.scale_1, int(h * (self.scale_1 / w))),

                                             interpolation=cv2.INTER_CUBIC)

        image2['d_img_scale_2'] = cv2.resize(image2['d_img_org'], dsize=(self.scale_2, int(h * (self.scale_2 / w))),

                                             interpolation=cv2.INTER_CUBIC)

        image2['d_img_scale_2'] = image2['d_img_scale_2'][:160, :, :]

        image1['score'] = sample['score']

        image2['score'] = sample['score']

        image1 = transform(image1)

        image2 = transform(image2)

        return image1, image2


class IQADatalist():

    def __init__(self, db_path, txt_file_name, train_mode, scene_list, train_size=0.8):
        self.txt_file_name = txt_file_name

        self.train_mode = train_mode

        self.train_size = train_size

        self.scene_list = scene_list

        self.db_path = db_path

    def load_data_dict(self):
        csv_path = os.path.join(self.db_path, 'MOS.csv')
        df = pd.read_csv(csv_path)
        df['0'] = df['0'].apply(lambda x: self.db_path + '/' + x)
        dataset = df['0'].tolist()
        labels = df['1'].tolist()

        scn_idx_list, d_img_list, score_list = [], [], []

        for i in range(len(df)):
                d_img_list.append(dataset[i])
                score_list.append(labels[i])

        # reshape score_list (1xn -> nx1)
        score_list = np.array(score_list)
        score_list = score_list.astype('float').reshape(-1, 1)

        data_dict = {'d_img_list': d_img_list, 'score_list': score_list}

        return data_dict
