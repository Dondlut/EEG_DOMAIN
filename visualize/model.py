import torch
import torch.nn as nn
from einops import rearrange
import time
from torchsummary import summary
from thop import profile
from lossFun import ReverseLayerF


class First_Conv(nn.Module):
    # attention之前的卷积层
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 3, (1, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Channel_wise_mean(nn.Module):
    # 对卷积后的kernel个数做平均
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean([1])


class ECA_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        # Two different branches of ECA module
        y_avg = self.conv1(y_avg.squeeze(-1).transpose(-1, -2))  # b,1,c
        y_max = self.conv1(y_max.squeeze(-1).transpose(-1, -2))
        # Multi-scale information fusion
        y = torch.concat((y_max, y_avg), dim=1)
        y = self.conv3(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # b,c,1,1
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SSTAttentionBlock(nn.Module):
    def __init__(self, spaSize, speSize, temSize, dropOut):
        '''

        :param spaSize: channel_size
        :param speSize: frequency_size
        :param temSize: temporal_size
        :param spaSpe: 是否做空频注意力
        :param spaTem: 是否做空时注意力
        '''
        super().__init__()
        self.spaSize = spaSize
        self.speSize = speSize
        self.temSize = temSize

        self.eca_layer = ECA_layer()
        self.flatten = nn.Flatten()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding='same')
        self.sfMaxPool = nn.MaxPool3d((self.temSize, 1, 1))
        self.sfAvgPool = nn.AvgPool3d((self.temSize, 1, 1))
        self.dropOut = dropOut
        if self.dropOut is not None:
            self.drop = nn.Dropout(dropOut)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        batch, channel, temporal, spatial, frequency = x.shape
        x = rearrange(x, 'b c t s f -> (b c) t s f')  # 方便做池化，否则5维的数据做不了池化

        temAtt = self.eca_layer(x)

        AvgAtt = self.sfAvgPool(x)
        MaxAtt = self.sfMaxPool(x)
        spaFreAtt = torch.concat((AvgAtt, MaxAtt), dim=1)
        spaFreAtt = self.conv(spaFreAtt)
        spaFreAtt = self.activation(spaFreAtt)
        spaFreAtt = torch.mul(x, spaFreAtt)

        res = torch.add(0.5 * temAtt, 0.5 * spaFreAtt)

        res = rearrange(res, '(b c) t s f -> b c t s f', b=batch, c=channel)
        return res


class ConvBlock(nn.Module):
    def __init__(self, in_channels, bottleneck, dropOut, growth_rate, bn_size):
        """
        :param bottleneck: 是否使用瓶颈模块
        :param dropOut
        """
        super().__init__()
        self.dropOut = dropOut
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.bottleneck = bottleneck
        self.batchNorm = nn.BatchNorm3d(self.in_channels)
        self.activation = nn.ReLU()
        self.convSS = nn.Conv3d(self.in_channels, self.bn_size * self.growth_rate, (3, 3, 1),
                                padding='same')  # SpatialSpec
        self.convST = nn.Conv3d(self.bn_size * self.growth_rate, self.growth_rate, (1, 1, 3),
                                padding='same')  # SpatialTem

        self.convSST = nn.Conv3d(self.in_channels, self.growth_rate, (3, 3, 3),
                                 padding='same', bias=False)  # 三维卷积

        self.bottleConv = nn.Conv3d(self.in_channels, self.bn_size * self.growth_rate, (1, 1, 1))
        self.batchNormBottle = nn.BatchNorm3d(self.bn_size * self.growth_rate)
        self.convSSBottle = nn.Conv3d(self.bn_size * self.growth_rate, self.growth_rate, (3, 3, 1),
                                      padding='same')  # SpatialSpec
        self.convSTBottle = nn.Conv3d(self.growth_rate, self.growth_rate, (1, 1, 3), padding='same')  # SpatialTem

    def forward(self, x):

        x = self.batchNorm(x)
        x = self.activation(x)

        if self.bottleneck:
            x = self.bottleConv(x)
            x = self.batchNormBottle(x)
            x = self.activation(x)
            x = self.convSSBottle(x)
            x = self.convSTBottle(x)
        else:
            # x = self.convSS(x)
            # x = self.convST(x)
            x = self.convSST(x)

        return x


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_input_features, nb_layers, growth_rate, bn_size, bottleNeck=False, dropOut=None):
        super().__init__()
        self.num_input_features = num_input_features
        self.nb_layers = nb_layers
        self.growth_rate = growth_rate
        self.bn_size = bn_size  # 输出通道数 = bn_size * growth_rate
        self.bottleNeck = bottleNeck
        self.dropOut = dropOut
        for i in range(self.nb_layers):
            layer = ConvBlock(self.num_input_features, growth_rate=self.growth_rate,
                              bn_size=self.bn_size, bottleneck=self.bottleNeck, dropOut=self.dropOut)
            self.num_input_features = self.growth_rate
            self.add_module("number {} ConvBlock".format(i), layer)

    def forward(self, x):
        output = [x]
        for name, layer in self.items():
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)

        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, compression=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.compression = compression
        self.batchNorm = nn.BatchNorm3d(in_channels)
        self.activation = nn.ReLU()
        self.conv = nn.Conv3d(in_channels, int(self.in_channels * self.compression), (1, 1, 1), padding='same')
        self.pool = nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.batchNorm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class SST4Seizure(nn.Sequential):
    def __init__(self, nb_classes=4, nb_layers_per_block=4, nb_dense_block=2, growth_rate=16, compression=0.5,
                 chaSize=1, bottleNeck=False, dropOut=0.5, spaSpe=True, spaTem=True, in_channels=1,
                 spaSize=22, speSize=129, temSize=59, bn_size=4):
        super().__init__()

        self.bn_size = bn_size
        self.spaSize = spaSize  # 22通道
        self.temSize = temSize
        self.speSize = speSize
        self.chaSize = chaSize  # 卷积出来的通道数
        self.compression = compression
        self.in_channels = in_channels
        self.spaTem = spaTem
        self.spaSpe = spaSpe
        self.dropOut = dropOut
        self.bottleNeck = bottleNeck
        self.growth_rate = growth_rate
        self.nb_dense_block = nb_dense_block
        self.nb_layers_per_block = nb_layers_per_block
        self.nb_classes = nb_classes
        # self.initConv1 = nn.Conv3d(self.in_channels, self.chaSize, (3, 3, 1), padding='same')

        for block_idx in range(self.nb_dense_block - 1):
            # print("index {} size before attention ".format(block_idx), self.chaSize, self.spaSize, self.speSize,
            #       self.temSize)
            attention_block = SSTAttentionBlock(spaSize=self.spaSize, speSize=self.speSize,
                                                temSize=self.temSize, dropOut=self.dropOut)
            # print("index {} size after attention before dense ".format(block_idx), self.chaSize, self.spaSize,
            #       self.speSize, self.temSize)
            self.add_module("number {} attention_block".format(block_idx), attention_block)

            dense_block = DenseBlock(num_input_features=self.chaSize, nb_layers=self.nb_layers_per_block,
                                     growth_rate=self.growth_rate, bn_size=self.bn_size,
                                     bottleNeck=self.bottleNeck, dropOut=self.dropOut)

            self.add_module("number {} dense_block".format(block_idx), dense_block)
            # dense_block改变卷积核个数对应的通道数
            self.chaSize = self.chaSize + self.nb_layers_per_block * self.growth_rate

            # print("index {} size after dense before tran ".format(block_idx), self.chaSize, self.spaSize, self.speSize,
            #       self.temSize)

            transition_block = TransitionBlock(in_channels=self.chaSize, compression=self.compression)
            self.add_module("number {} transition_block".format(block_idx), transition_block)
            # transition_block 会改变时空频三个维度的shape
            self.spaSize = self.spaSize // 2
            self.speSize = self.speSize // 2
            self.temSize = self.temSize // 2
            # 通道数也变了
            self.chaSize = self.chaSize // 2
            # print("index {} size after tran ".format(block_idx), self.chaSize, self.spaSize, self.speSize, self.temSize)

        self.dense_block_final = DenseBlock(num_input_features=self.chaSize, nb_layers=self.nb_layers_per_block,
                                            growth_rate=self.growth_rate, bn_size=self.bn_size,
                                            bottleNeck=self.bottleNeck, dropOut=self.dropOut)
        self.chaSize = self.chaSize + self.nb_layers_per_block * self.growth_rate
        # print("chaSize", chaSize)

        self.activation = nn.ReLU()
        # self.avgPool3d = nn.AvgPool3d((self.temSize,self.spaSize,self.speSize))
        # self.avgPool3d = nn.AvgPool3d((self.temSize // 2, 1, self.speSize // 2))
        self.avgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.final_conv = nn.Conv3d(in_channels=self.chaSize, out_channels=4, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        # self.flatten = nn.Flatten(start_dim=1)
        # self.logSoftmax = nn.LogSoftmax(dim=1)


class DANNModel(nn.Module):
    def __init__(self, nb_classes, dropOut=None):
        super(DANNModel, self).__init__()
        self.feature = SST4Seizure(dropOut=dropOut)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('finalConvCls',nn.Conv3d(in_channels=96, out_channels=nb_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)))
        self.class_classifier.add_module('flatten', nn.Flatten(start_dim=1))
        self.class_classifier.add_module('logSoftmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('finalConvDomain',nn.Conv3d(in_channels=96, out_channels=2, kernel_size=(1, 1, 1), stride=(1, 1, 1)))
        self.domain_classifier.add_module('flatten', nn.Flatten(start_dim=1))
        self.domain_classifier.add_module('logSoftmax', nn.LogSoftmax(dim=1))

    def forward(self, x, alpha):
        feature = self.feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)  # 所有数据都要经过domain判别器的
        class_output = self.class_classifier(feature)
        sub_output = self.domain_classifier(reverse_feature)
        return class_output, sub_output

    def forward(self, x):
        feature = self.feature(x)
        reverse_feature = ReverseLayerF.apply(feature, 1)  # 所有数据都要经过domain判别器的
        class_output = self.class_classifier(feature)
        sub_output = self.domain_classifier(reverse_feature)
        return class_output

# model = DANNModel(nb_classes=4,dropOut=None)
# summary(model, input_size=(1, 59, 22, 129))
# input_data = torch.randn([2, 1, 59, 22, 129])
# flops, params = profile(model, (input_data,))
# print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

# model = DANNModel(nb_classes=4)
# input_data = torch.randn([2, 1, 59, 22, 129])
# output = model(input_data,1)
# print(output[0].shape)