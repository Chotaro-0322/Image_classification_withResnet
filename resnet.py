# coding: utf-8
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
#from util import *

# -------------画像の前処理--------------------------------------
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (256,256))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
# -------------------------------------------------------------------

def parse_cfg(cfgfile):
    
    #コンフィグファイルを取得します

    #ブロックのリストを返します. 各ブロックは構築するニューラルネットワークのブロックを記述します.
    #ブロックはリストの辞書として表されます.

    #ここではcfgを解析し, すべてのブロックを辞書として保存することを目的としている.
    
    # まず、cfgファイルの内容を文字列のリストに格納する
    file = open(cfgfile, 'r')
    lines = file.read().split('\n') # split(sep)  sepで文章を分割してリストとして読み込む
    lines = [x for x in lines if len(x) > 0] # ｘの長さが0以上だった場合, linesに格納
    lines = [x for x in lines if x[0] != '#'] # コメント出なかった場合に, linesに格納
    lines = [x.rstrip().lstrip() for x in lines] # resrip : 後ろから文字を削除. lstrip : 前から文字を削除. 改行コードをすべて取り除いている


    # ループ分を用いてリストをブロックに格納する
    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[": # [net]などのkeyであったら...
            if len(block) != 0: # blockにすでに値が入っているとき(blockが空でなかったら), ※もしもblock["type":net, 'batch':1....]となっているようなら, その結果をblocksリストに格納
                blocks.append(block) # ブロックリストにblockを格納
                block = {} # blockを再び初期化
            block["type"] = line[1:-1].rstrip() # [net]だった場合, 'type':'net'となるようにする
        else:
            key, value = line.split("=") # 
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

# 上記のparse_cfgによって返されたリストを使用して、構成ファイルに存在するブロックのpytorchモジュールを作成します.
# リストには[Yolo][convolutional][shortcut][upsample][Route]の５種類のレイヤーが存在する. [convolutional][upsample]はnn.Moduleに存在するが, 
# 残りのレイヤーは拡張によって追加しなければならない.
# 以下のcreate_modulesは上記のparse_cfgによって返されるリストブロックを受け取ります.
def create_modules(blocks):
    net_info = blocks[0] # ブロックのリストを反復処理する前に、[net]に関する情報を保存する変数net_infoを定義します。
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    # この関数はnn.ModuleListを返します. 

    # 新しく畳み込み層にはいるとき, そのカーネルの次元を定義する必要がある. カーネルの高さと幅はcfgガイルによって提供されるが, 
    # カーネルの深さは前のレイヤーに存在するフィルターの数と全く同じである.
    # そのため, 畳み込み層のフィルターの数を追跡する必要がある.

    # ここでの考え方は, ブロックのリストを反復処理し, 各ブロックに対してPytorchモジュールを作成することです. 
    # 以下は反復処理によって
    # レイヤーの作成
    for index, x in enumerate(blocks[1:]): # [net]はれいやーに含まないのでblocks[1:]となる.
        module = nn.Sequential() # ここでモジュールを宣言することによって, ここに新しいレイヤーの計算方法を追加できる

        # blockのtypeを確認する
        # blockのために新しくモジュールを作る
        # module_listに加える

        # nn.Sequentialクラスはいくつかのnn.Moduleオブジェクトを順番に実行するために使用します. cfgを見ると, ブロックに複数のレイヤーが含まれている.
        # たとえば, バッチ層やReLU関数層など, nn.Sequentialをもちいればつなぎ合わせられる.
        # 以下は畳み込み及びアップサンプリングレイヤーの作成方法です.

        if (x["type"] == "convolutional"):
            # レイヤーに関する情報を手にい入れる
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2 # パディングサイズからプーリングサイズを決定
            else:
                pad = 0

            # 上記で作成した畳み込みレイヤーをmodulesに追加する
            conv = nn.Conv2d(prev_filters, filters,kernel_size, stride, pad, bias = bias) # nn.Conv2dで計算
            module.add_module("conv_{0}".format(index), conv)

            # バッチ正規化レイヤーをmodulesに追加する
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # activationの確認.
            # YOLOレイヤーの場合はLinearまたはLeaky ReLUのどちらかです.
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        # アップサンプリングレイヤー
        # Bilinear2dUpsamplingを用います
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x["type"] == "avgpool"):
            avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            module.add_module("avgpool_{}".format(index), avgpool)

        # maxpoolレイヤーの表示
        elif x["type"] == "maxpool":
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            maxpool = nn.MaxPool2d(kernel_size, stride)
            module.add_module("maxpool_{}".format(index), maxpool)      

        # ショートカットはスキップ接続に対応
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)



        

        # Loopの最後にはこれらの設定を保存するコードを書く
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    # 最後に[net]の情報とmodel_listを返します.
    return (net_info, module_list)

#blocks = parse_cfg("resnet.cfg")
#print(create_modules(blocks))


class Resnet(nn.Module):
    def __init__(self, cfgfile):
        super(Resnet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks) # [net]の情報, [その他のレイヤー]の情報

    def forward(self, x, CUDA):
        modules = self.blocks[1:] # [net]を除くレイヤーを呼び出す
        outputs = {} # routeレイヤーの出力を格納します.

        write = 0 # 
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type =="convolutional" or module_type == "upsample" or module_type == 'avgpool' or module_type == 'maxpool':
                x = self.module_list[i](x) # [その他のレイヤーの情報]に入力を入れる

            elif module_type == "shortcut": # ショートカット層で層をスキップ設定をする
                from_ = int(module["from"]) # "from"の値を読み込む(例: -3)
                x = outputs[i-1] + outputs[i+from_] # その層と,3 つ前の層を足し合わせる.(結合ではない)

            output[i] = x
        return output[i]

        #model = Resnet("resnet.cfg") # cfgファイルを呼び出す
        #pred = model.forward(inp, torch.cuda.is_available()) # model内のforwardを実行
        #print(pred)

    def load_weights(self, weightfile):
        # ウェイトファイルを開く
        fp = open(weightfile, "rb")

        # 最初の5つの値はヘッダーファイルです.
        # 1.  メジャーバージョン番号?
        # 2.  マイナーバージョン番号?
        # 3.  サブヴィジョン番号? 
        # 4, 5.  ネットワークから見た画像(ネットワークが訓練した画像数)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header) # numpy形式のヘッダーをtensorにする
        self.seen = self.header[3] # これまで何枚画像を見てきたかをself.seenに格納

        # 残りのウェイトを格納してみる(ウェイトはfloat32の形式で格納されている)
        weights = np.fromfile(fp, dtype = np.float32)
        #print(weights) # 中身は1次元のただの数字の列になっている
        
        # 次に重みファイルを反復処理し, 重みをネットワークのモジュールにロードします.
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"] # module_typeにレイヤーの名前("type")をロードする.
            
            # もしもmodule_typeが"convlution"だった場合, ウェイトをロードする.
            # それ以外は無視する
            if module_type == "convolutional":
                model = self.module_list[i] # model[0]はConv2d, model[1]はBatchConv2d, model[2]はLeakyReLUというように情報が並んでいる.
                # --------------------試しにmodelの中身を覗いてみる---------------------------
                '''
                if i==0:
                    print('\nmodel[2]は\n', model[2])
                '''
                # -------------------------------------------------------------------------------

                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"]) # "batch_normalize"が存在するとき, batch_normalizeに格納
                except:
                    batch_normalize = 0

                # まず, convを取得
                conv = model[0] # convに"covolutional"を格納
                # ----------------ウェイトの様子を見てみる-----------------------------------------------------------------
                '''
                if i == 0: # 最初のひとつだけ表示
                    """conv(module_listには各層のウェイトなどの情報を格納している.weightが入っており, .weight.dataにはそのウェイトの値のみが格納されている"""
                    """おそらく最初の状態ではランダムにウェイトが格納されている状態であると思われる?"""
                    print("\n最初の畳み込み層の情報\n", conv)
                    print("\n最初の畳み込み層のウェイト\n",conv.weight[1, :, :, :]) # 1フィルタだけ出力 ([出力フィルタサイズ, 入力フィルタサイズ, フィルタの横の長さ, フィルタの縦の長さ])だと思われる
                    print("\n最初の畳み込み層のウェイトの形は\n", conv.weight.shape) # [32, 3, 3, 3]になっている, 
                '''
                # ---------------------------------------------------------------------------------------------------------

                if batch_normalize: # batch_normalizeが存在したら
                    bn = model[1] # "batch_noemalize"をbnに格納
                    
                    # ---------------------------batch_normの様子を見てみる--------------------------------------------
                    '''
                    if i == 0:
                        print("\nbatch_normalizeの情報\n", bn)
                        print("\nbatch_normalizeのバイアス\n", bn.bias)
                        print("\nbatch_normalizeのウェイト\n", bn.weight)
                        print("\nbatch_normalizeのrunning_mean\n", bn.running_mean)
                        print("\nbatch_normalizeのbn.running_var\n", bn.running_var)
                    '''
                    # -----------------------------------------------------------------------------------------------------

                    # Batch_norm_layerのウェイトの数字を格納する.
                    num_bn_biases = bn.bias.numel() #bn.bias.の要素数をすべて計算. (4, 4)の場合numel()=16

                    # ウェイトのロード
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]) # weightsの[ptr～num_bn_biases]分だけbn_biasesに入れる
                    #print("\nbatch_normalizeの情報\n", bn)
                    #print("\nウェイトチェック\n", weights[ptr: ptr + num_bn_biases], i)
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]) # 次のweightsの[ptr～num_bn_biases]分だけbn_weightsに入れる
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])  # 次の分はbn_running_meanに入れる
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]) # 次の分はbn_running_varに入れる
                    ptr += num_bn_biases

                    # ロードされたウェイトをモデルウェイトのdimsにキャストする.　ここがよくわからない.
                    bn_biases = bn_biases.view_as(bn.bias.data) # bn_biasesの形をbn.bias.dataに変形させる.
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # データをmodelへコピーする, 上書きして, weightsに入っていた値に変更している.
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                    # -------------------batch_normのバイアスの変化を確認-------------------------------------------------
                    '''
                    if i==0:
                        print("\nWeightsファイルの変形後は, batch_normalizeのバイアス\n", bn.bias) # しっかり代入されていることが確認させる
                    '''
                    # ------------------------------------------------------------------------------------------

                else: # batch_normalizeがなければ
                    # biasesの値
                    num_biases = conv.bias.numel()

                    # ウェイトのロード
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # ロードされたウェイトをモデルウェイトの寸法にしたがって再形成します.
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # データのコピー
                    conv.bias.data.copy_(conv_biases)

                # 畳み込み層の重みをロードしてみましょう
                num_weights = conv.weight.numel()

                # 重みについても上記と同じ
                conv_weights = torch.from_numpy(weights[ptr: ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                # -----------------畳み込み層のウェイトが変化しているかを確認----------------------------------
                '''
                if i == 0:
                    print('\n変換後のweightsのデータを表示する\n', conv.weight[1, :, :, :]) # しっかりとweightsのウェイトに変換されていることが確認できる.
                '''
                # ------------------------------------------------------------------------------------------------

