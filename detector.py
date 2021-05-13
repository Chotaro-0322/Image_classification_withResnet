'''
画像分類の手順
1. DataSetの作成(first_Create_Dataset.py)
2. DataLoaderの作成(このファイル内)
3. NetWorkの作成(このファイル内)
4. 最適化手法の設定(このファイル内)
5. train_model(fourth_eval_train.py)
6. TrainEngin(このファイル内)
といった手順になっている.
'''

from first_Create_Dataset import *
from fourth_eval_train import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
# ===========scond_Create_Dataset.pyと同じ===================================

# 画像データパスのリスト作成
train_list = make_datapath_list(phase = 'train')
val_list = make_datapath_list(phase = 'val')

# Datasetに必要な要素を定義(今回はResnetなので入力は256)
size = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Datasetの作成
'''
※もう一度詳しく説明
file_list の出力はただの画像のパスのリスト

ImageTransform は画像の前処理を行う ! ちなみにMWDatasetの中で, self.transformとして保存される
    self.transformはモジュールという事に注意.
    そして, self.transform(img, phase)とすることによって, 画像前処理が実行される.

MWDatasetは, train_dataset[0]は画像のテンソルデータ : torch.Size([3, 256, 256]).
                        train_dataset[1]は画像のラベルデータ(0や 1や 2などになっている).
                        となる.
'''
train_dataset = MWDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
#for  idx in train_dataset:
    #print(idx[1])
val_dataset = MWDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

# ミニバッチサイズを指定
batch_size = 32

# DataLoaderを作成
'''
torch.util.data.DataLoaderを用いる.
torch.utils.data.DataLoaderを理解することが必須.
下記サイトを参考
https://schemer1341.hatenablog.com/entry/2019/01/06/024605

前提として, first_Create_Dataset.pyにあるように, train_datasetの中身は
    (画像のテンソル, その画像のラベル)となっている.
    ↑どこに行っても, この出力の形になっている.
torch.utils.data.DataLoaderは上記のDatasetの出力をbatchごとに自動でまとめてくれる.
今回, batchサイズは32なので, 
これにより, ([32個分の画像テンソル, その32個分のクラスラベル])に. ランダム(shuffle=Trueの場合)に取り出してまとめてくれる.
おそらく, 
train_dataloader[0].shapeはtensor.Size([32, 3, 256, 256])
train_dataloader[1].shapeはtensor.Size([32, 1])
になっているはず. 
'''
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size = batch_size, shuffle=True) # 出力は2つ

#for idx in train_dataloader:
    #print(idx)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size = batch_size, shuffle=True)

# 辞書型変数にまとめる
dataloaders_dict = {"train":train_dataloader, "val":val_dataloader} # これらを辞書にまとめる.


# ====================================================================

# ========================ここからNetworkの作成================================

# 学習済みResnetモデルのロード

use_pretrained = True
net = models.resnext50_32x4d(pretrained=use_pretrained)

# Resnetの最後の出力層を2個に付け替える. (元はnet.fc = nn.Linear(in_features=2048, out_features=1000, bias = True))
net.fc = nn.Linear(in_features=2048, out_features=2, bias = True)

# 訓練モードに変更
net.train()

print('ネットワーク設定完了: 学習済みの重みをロードし, 訓練モードに設定しました.')

# ======================================================================

#===============損失関数を定義==============================================
# 損失関数の設定
criterion = nn.CrossEntropyLoss()
#=======================================================================

#===============最適化手法を設定=============================================
# ファインチューニング: 後ろの層に行けばいくほど学習率が大きくなっていくように設計するもの
# 転移学習: 最後の全結合層(全畳み込み層)のみを学習によって変化するように設計するもの
params_to_updata_1 = []
params_to_updata_2 = []
params_to_updata_3 = []

# 学習させるパラメータ名を指定
update_params_name_1 = ["layer1", "layer2", "layer3", "layer4.0","layer4.1" ]
update_params_name_2 = ["layer4.2.conv1.weight", "layer4.2.conv1.bias"]
update_params_name_3 = ["fc.weight", "fc.bias"]


# パラメータごとに各リストに収納する(true_model.named_parameter()をすることによって, パラメータの名前を取得できる(ウェイト部分が分かる))
for name, param in net.named_parameters():
    if update_params_name_1[0] in name:
        param.requires_grad = True
        params_to_updata_1.append(param)
        print("params_to_update_1に格納: ", name)
    elif name in update_params_name_2:
        param.requires_grad = True
        params_to_updata_2.append(param)
        print("params_to_update_2に格納: ", name)
    elif name in update_params_name_3:
        param.requires_grad = True
        params_to_updata_3.append(param)
        print("params_to_update_3に格納: ", name)
    else:
        param.requires_grad = False
        print("勾配計算なし。学習しない:",name)
# 最適化手法の設定
optimizer = optim.SGD([
    {'params': params_to_updata_1, 'lr': 1e-4},
    {'params': params_to_updata_2, 'lr': 5e-4},
    {'params': params_to_updata_3, 'lr': 1e-3}
], momentum=0.9)
    
# 学習を・検証を実行する
num_epochs = 10
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)