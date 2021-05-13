from first_Create_Dataset import *

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

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size = batch_size, shuffle=True)

# 辞書型変数にまとめる
dataloaders_dict = {"train":train_dataloader, "val":val_dataloader} # これらを辞書にまとめる.

