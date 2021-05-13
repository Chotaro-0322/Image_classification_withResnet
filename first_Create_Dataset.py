import glob
from pathlib import Path
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

__all__ = ['ImageTransform', 'MWDataset', 'make_datapath_list']

# class ImageTransformは画像の前処理を行うClassである.
class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([ # trainでは様々な前処理を行うことで, 精度を上げていく
                transforms.RandomResizedCrop( # 指定されたPIL画像をランダムなサイズとアスペクト比にトリミングします。
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(), # 指定されたPIL画像を、指定された確率でランダムに水平方向に反転します。
                transforms.ToTensor(), # tensorに変換
                transforms.Normalize(mean, std) # mean: 平均, std: 標準偏差で正規化する.
            ]),
            'val': transforms.Compose([ # valでは, ネットワークに入れるために必要な大きさにする
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase='train'):
        '''
        model = ImageTransform
        model(img, phase='train')
        とすることで実行される処理
        '''
        return self.data_transform[phase](img) # これにより, imgを読み込み, data_transformにかけられる.


# make_datapath_list画像データへのファイルパスを取得する.
def make_datapath_list(phase='train'):
    path = list(Path('./image/'+phase).glob('**/*.jpg'))

    path_list = []
    label = []

    for i in path:
        label.append(i.parent.name)
        path_list.append(str(i))

    return path_list, label

class MWDataset(data.Dataset): # Datasetは、入力データとそれに対応するラベルを1組返すモジュール。
    '''
    オリジナルDatasetを実装するときに守る必要がある要件は以下３つ。
    1. torch.utils.data.Datasetを継承する。
    2. __len__を実装する。
    3. __getitem__を実装する。

    __len__は、len(obj)で実行されたときにコールされる関数。
    __getitem__は、obj[i]のようにインデックスで指定されたときにコールされる関数。
    '''
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list, self.label = file_list # ファイルパスのリスト, make_datapath_listのやつ
        self.transform = transform # 前処理クラスのインスタンス, class ImageTransformのやつ
        self.phase = phase # train or valの指定

    def __len__(self):
        # 画像の枚数を変えす
        return len(self.file_list)

    def __getitem__(self, index):
        # 前処理をした画像のtensor形式のデータとラベルを取得部分(本文と考えてもよい)

        img_path, label = self.file_list[index], self.label[index]
        #print(img_path)
        img = Image.open(img_path)
        if self.phase == "val":
            plt.imshow(img)
            plt.show()

        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)

        # 画像のラベルを数値に変更する
        if label == "man":
            label = 0
        elif label == "woman":
            label = 1
        return img_transformed, label