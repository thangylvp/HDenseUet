import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchsummary import summary
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from medpy.io import load
from skimage.transform import resize
from multiprocessing.dummy import Pool as ThreadPool

device = 'cuda'

class Scale(nn.Module): 
    def __init__(self, num_feature): 
        super(Scale, self).__init__() 
        self.num_feature = num_feature 
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True) 
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True) 
    def forward(self, x): 
        y = torch.zeros(x.shape, dtype= x.dtype, device= x.device)
        for i in range(self.num_feature): 
            y[:, i, :, :] = x[:, i, :, :].clone() * self.gamma[i] + self.beta[i] 
        return y 

class Scale3d(nn.Module): 
    def __init__(self, num_feature): 
        super(Scale3d, self).__init__() 
        self.num_feature = num_feature 
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True) 
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True) 
    def forward(self, x): 
        y = torch.zeros(x.shape, dtype= x.dtype, device= x.device)
        for i in range(self.num_feature): 
            y[:, i, :, :, :] = x[:, i, :, :, :].clone() * self.gamma[i] + self.beta[i] 
        return y 


class conv_block(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0, weight_decay=1e-4):
        super(conv_block, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm2d(nb_inp_fea, eps= eps, momentum= 1))
        self.add_module('scale1', Scale(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2d1', nn.Conv2d(nb_inp_fea, 4 * growth_rate, (1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm2d(4 * growth_rate, eps= eps, momentum= 1))
        self.add_module('scale2', Scale(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace= True))
        self.add_module('conv2d2', nn.Conv2d(4 * growth_rate, growth_rate, (3, 3), padding = (1,1), bias= False))
        
    
    def forward(self, x):
        out = self.norm1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv2d1(out)

        if (self.drop > 0):
            out = F.dropout(out, p= self.drop)
        
        out = self.norm2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv2d2(out)

        if (self.drop > 0):
            out = F.dropout(out, p= self.drop)

        return out

class _Transition(nn.Sequential):
    def __init__(self, num_input, num_output, drop=0):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input))
        self.add_module('scale', Scale(num_input))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv2d', nn.Conv2d(num_input, num_output, (1, 1), bias=False))
        if (drop > 0):
            self.add_module('drop', nn.Dropout(drop, inplace=True))
        self.add_module('pool', nn.AvgPool2d(kernel_size= 2, stride= 2))




class dense_block(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0, weight_decay=1e-4, grow_nb_filters=True):
        super(dense_block, self).__init__()
        for i in range(nb_layers):
            layer = conv_block(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer%d' % (i + 1), layer)
        
    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)
            
        
    
class denseUnet(nn.Module):
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, drop_rate=0, weight_decay=1e-4, num_classes=1000):
        super(denseUnet, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, nb_filter, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(nb_filter, eps= eps)),
            ('scale0', Scale(nb_filter)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        for i, num_layer in enumerate(block_config):
            block = dense_block(num_layer, nb_filter, growth_rate, drop_rate)
            nb_filter += num_layer * growth_rate
            self.features.add_module('denseblock%d' % (i + 1), block)
            if i != len(block_config) - 1:
                trans = _Transition(nb_filter, nb_filter // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                nb_filter = nb_filter // 2
                
        self.features.add_module('norm5', nn.BatchNorm2d(nb_filter, eps= eps, momentum= 1))
        self.features.add_module('scale5', Scale(nb_filter))
        self.features.add_module('relu5', nn.ReLU(inplace= True))

        self.decode = nn.Sequential(OrderedDict([
            ('up0', nn.Upsample(scale_factor=2)),
            ('conv2d0', nn.Conv2d(nb_filter, 768, (3, 3), padding= 1)),
            ('bn0', nn.BatchNorm2d(768, momentum= 1)), 
            ('ac0', nn.ReLU(inplace=True)),
            
            ('up1', nn.Upsample(scale_factor=2)),
            ('conv2d1', nn.Conv2d(768, 384, (3, 3), padding= 1)),
            ('bn1', nn.BatchNorm2d(384, momentum= 1)), 
            ('ac1', nn.ReLU(inplace=True)),

            ('up2', nn.Upsample(scale_factor=2)),
            ('conv2d2', nn.Conv2d(384, 96, (3, 3), padding= 1)),
            ('bn2', nn.BatchNorm2d(96, momentum= 1)), 
            ('ac2', nn.ReLU(inplace=True)),

            ('up3', nn.Upsample(scale_factor=2)),
            ('conv2d3', nn.Conv2d(96, 96, (3, 3), padding= 1)),
            ('bn3', nn.BatchNorm2d(96, momentum= 1)), 
            ('ac3', nn.ReLU(inplace=True)),

            ('up4', nn.Upsample(scale_factor=2)),
            ('conv2d4', nn.Conv2d(96, 64, (3, 3), padding= 1)),
            ('bn4', nn.BatchNorm2d(64, momentum= 1)), 
            ('ac4', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        out = self.features(x)
        out = self.decode(out)
        return out
        
class conv_block3d(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0, weight_decay=1e-4):
        super(conv_block3d, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm3d(nb_inp_fea, eps= eps, momentum= 1))
        self.add_module('scale1', Scale3d(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv3d1', nn.Conv3d(nb_inp_fea, 4 * growth_rate, (1, 1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm3d(4 * growth_rate, eps= eps, momentum= 1))
        self.add_module('scale2', Scale3d(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace= True))
        self.add_module('conv3d2', nn.Conv3d(4 * growth_rate, growth_rate, (3, 3, 3), padding = (1,1,1), bias= False))
        
    
    def forward(self, x):
        out = self.norm1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv3d1(out)

        if (self.drop > 0):
            out = F.dropout(out, p= self.drop)
        
        out = self.norm2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv3d2(out)

        if (self.drop > 0):
            out = F.dropout(out, p= self.drop)

        return out

class dense_block3d(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0, weight_decay=1e-4, grow_nb_filters=True):
        super(dense_block3d, self).__init__()
        for i in range(nb_layers):
            layer = conv_block3d(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer3d%d' % (i + 1), layer)
        
    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class _Transition3d(nn.Sequential):
    def __init__(self, num_input, num_output, drop=0):
        super(_Transition3d, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input))
        self.add_module('scale', Scale3d(num_input))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv3d', nn.Conv3d(num_input, num_output, (1, 1, 1), bias=False))
        if (drop > 0):
            self.add_module('drop', nn.Dropout(drop, inplace=True))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2) , stride=(1, 2, 2) ))

class denseUnet3d(nn.Module):
    def __init__(self, num_input, growth_rate=32, block_config=(3, 4, 12, 8), num_init_features=96, drop_rate=0, weight_decay=1e-4, num_classes=1000):
        super(denseUnet3d, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(num_input, nb_filter, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(nb_filter, eps= eps)),
            ('scale0', Scale3d(nb_filter)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        for i, num_layer in enumerate(block_config):
            block = dense_block3d(num_layer, nb_filter, growth_rate, drop_rate)
            nb_filter += num_layer * growth_rate
            self.features.add_module('denseblock3d%d' % (i + 1), block)
            if i != len(block_config) - 1:
                trans = _Transition3d(nb_filter, nb_filter // 2)
                self.features.add_module('transition3d%d' % (i + 1), trans)
                nb_filter = nb_filter // 2
                
        self.features.add_module('norm5', nn.BatchNorm3d(nb_filter, eps= eps))
        self.features.add_module('scale5', Scale3d(nb_filter))
        self.features.add_module('relu5', nn.ReLU(inplace= True))

        self.decode = nn.Sequential(OrderedDict([
            ('up0', nn.Upsample(scale_factor=(1, 2, 2))),
            ('conv2d0', nn.Conv3d(nb_filter, 504, (3, 3, 3), padding= 1)),
            ('bn0', nn.BatchNorm3d(504, momentum= 1)), 
            ('ac0', nn.ReLU(inplace=True)),
            
            ('up1', nn.Upsample(scale_factor=(1, 2, 2))),
            ('conv2d1', nn.Conv3d(504, 224, (3, 3, 3), padding= 1)),
            ('bn1', nn.BatchNorm3d(224, momentum= 1)), 
            ('ac1', nn.ReLU(inplace=True)),

            ('up2', nn.Upsample(scale_factor=(1, 2, 2))),
            ('conv2d2', nn.Conv3d(224, 192, (3, 3, 3), padding= 1)),
            ('bn2', nn.BatchNorm3d(192, momentum= 1)), 
            ('ac2', nn.ReLU(inplace=True)),

            ('up3', nn.Upsample(scale_factor=(2, 2, 2))),
            ('conv2d3', nn.Conv3d(192, 96, (3, 3, 3), padding= 1)),
            ('bn3', nn.BatchNorm3d(96, momentum= 1)), 
            ('ac3', nn.ReLU(inplace=True)),

            ('up4', nn.Upsample(scale_factor=(2, 2, 2))),
            ('conv2d4', nn.Conv3d(96, 64, (3, 3, 3), padding= 1)),
            ('bn4', nn.BatchNorm3d(64, momentum= 1)), 
            ('ac4', nn.ReLU(inplace=True))

#            ('conv2d5', nn.Conv3d(64, 3, (1, 1, 1), padding= 0))
        ]))

    def forward(self, x):
        out = self.features(x)
        out = self.decode(out)
        return out


class dense_rnn_net(nn.Module):
    def __init__(self, num_slide, drop_rate = 0.3):
        super(dense_rnn_net, self).__init__()
        self.num_slide = num_slide
        self.drop = drop_rate
        self.dense2d = denseUnet()
        self.dense3d = denseUnet3d(4)
        self.conv2d5 = nn.Conv2d(64, 3, (1, 1), padding= 0)
        self.conv3d5 = nn.Conv3d(64, 3, (1, 1, 1), padding= 0)
        self.finalConv3d1 = nn.Conv3d(64, 64, (3, 3, 3), padding= (1, 1, 1))
        self.finalBn = nn.BatchNorm3d(64)
        self.finalAc = nn.ReLU(inplace=True)
        self.finalConv3d2 = nn.Conv3d(64, 3, (1, 1, 1))

    def forward(self, x):
        # x = x[0:1, :, :, :, :]
        print("x shape : ", x.shape)
        input2d = x[:, 0:2, :, :]
        single = x[:, 0:1, : , :]
        input2d = torch.cat((input2d, single), 1)
        for i in range(self.num_slide - 2):
            input2dtmp = x[:, i:i+3, :,  :]
            input2d = torch.cat((input2d, input2dtmp), 0)
            if i == self.num_slide - 3:
                f1 = x[:, self.num_slide - 2 : self.num_slide, :, :]
                f2 = x[:, self.num_slide - 1 : self.num_slide, :, :]
                ff = torch.cat((f1, f2), 1)
                input2d = torch.cat((input2d, ff), 0)

        # input2d = input2d[:, :, :, :, 0]
        # input2d = input2d.permute(0, 3, 1, 2)
        
        feature2d = self.dense2d(input2d)
        final2d = self.conv2d5(feature2d)

        
        input3d = final2d.permute(1, 0, 2, 3)
        feature2d = feature2d.permute(1, 0, 2, 3)
        input3d.unsqueeze_(0)
        feature2d.unsqueeze_(0)
        
        x_tmp = x.unsqueeze(0)
        x_tmp *= 250.0


        input3d = torch.cat((input3d, x_tmp), 1)

        feature3d = self.dense3d(input3d)
        output3d = self.conv3d5(feature3d)

        final = torch.add(feature2d, feature3d)

        final = self.finalConv3d1(final)
        if (self.drop > 0):
            final = F.dropout(final, p= self.drop)
        
        final = self.finalBn(final)
        final = self.finalAc(final)
        final = self.finalConv3d2(final)

        return final
