import torch
from torch import nn

def multi_perceptrons(output=10,input = 64):
    '''多层感知机'''
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input,64),nn.ReLU(),
        nn.Linear(64,output)
    )


def conv_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1)
    )

def transition_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )

class DenseBlock(nn.Module):
    def __init__(self,num_convs,input_channels,num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
        
    def forward(self,X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X,Y), dim = 1)
        return X
    

def dense_net(num_channels=64,growth_rate=32,num_output=10,num_convs_in_dense_block=[4,4,4,4]):
    '''最终用于训练的模型，架构是dense net，前面的都是轮子'''
    '''使用方法为： net = classifier(...)'''
    blks = []
    input_layer = nn.Sequential(
        nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    )
    for i,num_convs in enumerate(num_convs_in_dense_block):
        blks.append(DenseBlock(num_convs,num_channels,growth_rate))
        num_channels += num_convs*growth_rate
        if i != len(num_convs_in_dense_block) - 1:
            blks.append(transition_block(num_channels,num_channels//2))
            num_channels = num_channels // 2
    return nn.Sequential(
        input_layer,
        *blks,
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(num_channels,num_output)
    )


def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    return nn.Sequential(*layers)

def vgg(conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512)),ratio = 1,output=10,size_x=224,size_y=224,in_channels=1,layers=5,size_linear = 2048):
    '''vgg架构，最终用于直接调用的分类器，size_x与size_y为输入数据的尺寸，应当为32的整数倍，ratio用于调整架构宽度，ratio越大宽度越窄，仅能输入2的n次幂(ratio <= 32)'''
    '''使用方法为: net = classifier(...) net.apply(weight_init)'''
    ratio1 = 2 ** layers
    size_linear = (size_x // ratio1) * (size_y // ratio1)
    conv_blks = []
    arch = [(pair[0],pair[1] // ratio) for pair in conv_arch]
    for(num_convs,out_channels) in arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * size_linear,size_linear),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(size_linear,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.AdaptiveAvgPool1d(output_size=4096),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,output)
    )
