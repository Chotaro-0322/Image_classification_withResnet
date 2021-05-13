from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス: ", device)

    # ネットワークをGPUへ
    net.to(device)

    #ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-----------------')

        if epoch == num_epochs:
            phase = 'val'
            net.val() # valモードへ以降
        else:
            phase = 'train'
            net.train() # trainモードへ以降

        epoch_loss = 0.0 # epochごとの損失和
        epoch_corrects = 0 # epochごとの正解数

        # データローダーから道バッチを取り出すループ
        for inputs, labels in tqdm(dataloaders_dict[phase]): # 形は(tensor.Size[32, 3, 256, 256], tensor.Size[32, 1])
            
            # GPUにデータを送る
            inputs = inputs.to(device)
            labels = labels.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝播(forward計算)
            with torch.set_grad_enabled(phase=='train'):
                outputs = net(inputs)
                loss = criterion(outputs, labels) # 損失を計算
                _, preds = torch.max(outputs, 1) # ラベルを予測

                # 訓練時はバックプロバゲーション
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 結果の計算
                epoch_loss += loss.item() * inputs.size(0) # lossの合計を更新
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

                # epoch ごとのlossと正解率を表示
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'. format(phase, epoch_loss, epoch_acc))
                

                