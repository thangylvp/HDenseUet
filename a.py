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
        input2d = x[:, :, :, 0:2, :]
        single = x[:, :, :, 0:1, :]
        input2d = torch.cat((input2d, single), 3)
        for i in range(self.num_slide - 2):
            input2dtmp = x[:, :, :, i:i+3, :]
            input2d = torch.cat((input2d, input2dtmp), 0)
            if i == self.num_slide - 3:
                f1 = x[:, :, :, self.num_slide - 2 : self.num_slide, :]
                f2 = x[:, :, :, self.num_slide - 1 : self.num_slide, :]
                ff = torch.cat((f1, f2), 3)
                input2d = torch.cat((input2d, ff), 0)

        input2d = input2d[:, :, :, :, 0]
        input2d = input2d.permute(0, 3, 1, 2)
        
        feature2d = self.dense2d(input2d)
        final2d = self.conv2d5(feature2d)

        
        input3d = final2d.permute(1, 0, 2, 3)
        feature2d = feature2d.permute(1, 0, 2, 3)
        input3d.unsqueeze_(0)
        feature2d.unsqueeze_(0)
        
        x_tmp = x.permute(0, 4, 3, 1, 2)
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
        














thread_num = 14
liverlist = [32,34,38,41,47,87,89,91,105,106,114,115,119]
parser = argparse.ArgumentParser(description='Keras DenseUnet Training')
#  data folder
parser.add_argument('-data', type=str, default='../H-DenseUNet/data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default='./model/model_best.hdf5')
parser.add_argument('-input_cols', type=int, default=8)
parser.add_argument('-arch', type=str, default='')

#  data augment
parser.add_argument('-mean', type=int, default=48)
args = parser.parse_args()


def load_seq_crop_data_masktumor_try(Parameter_List):
    img = Parameter_List[0]
    tumor = Parameter_List[1]
    lines = Parameter_List[2]
    numid = Parameter_List[3]
    minindex = Parameter_List[4]
    maxindex = Parameter_List[5]
    #  randomly scale
    scale = np.random.uniform(0.8,1.2)
    deps = int(args.input_size * scale)
    rows = int(args.input_size * scale)
    cols = args.input_cols

    sed = np.random.randint(1,numid)
    cen = lines[sed-1]
    cen = np.fromstring(cen, dtype=int, sep=' ')
    # print (cen)
    a = min(max(minindex[0] + deps/2, cen[0]), maxindex[0]- deps/2-1)
    b = min(max(minindex[1] + rows/2, cen[1]), maxindex[1]- rows/2-1)
    c = min(max(minindex[2] + cols/2, cen[2]), maxindex[2]- cols/2-1)
    # print(a - deps // 2, ' ', c - args.input_cols // 2, ' ', c + args.input_cols // 2)
    cropp_img = img[int(a - deps / 2):int(a + deps / 2), int(b - rows / 2):int(b + rows / 2),
                int(c - args.input_cols / 2): int(c + args.input_cols / 2)].copy()
    
    cropp_tumor = tumor[int(a - deps / 2):int(a + deps / 2), int(b - rows / 2):int(b + rows / 2),
                  int(c - args.input_cols / 2):int(c + args.input_cols / 2)].copy()

    cropp_img -= args.mean
     # randomly flipping
    flip_num = np.random.randint(0,8)
    if flip_num == 1:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
    elif flip_num == 2:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    elif flip_num == 3:
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    #
    cropp_tumor = resize(cropp_tumor, (args.input_size,args.input_size, args.input_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    cropp_img   = resize(cropp_img, (args.input_size,args.input_size, args.input_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropp_img, cropp_tumor


def generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorline_list, liverline_list, liverbox_list):
    while 1:
        X = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols,1), dtype='float32')
        Y = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols,1), dtype='int16')
        Parameter_List = []
        for idx in range(batch_size):
            count = np.random.choice(trainidx)
            # img = img_list[count]
            img, img_header = load(img_list[count] )
            print('xxxxxxxxxxxxxxxxxxxxxx', img.shape)
            # tumor = tumor_list[count]
            tumor, tumor_header = load(tumor_list[count])
            maxmin = np.loadtxt(liverbox_list[count], delimiter=' ')
            minindex = maxmin[0:3]
            maxindex = maxmin[3:6]
            minindex = np.array(minindex, dtype='int')
            maxindex = np.array(maxindex, dtype='int')
            minindex[0] = max(minindex[0]-3, 0)
            minindex[1] = max(minindex[1]-3, 0)
            minindex[2] = max(minindex[2]-3, 0)
            maxindex[0] = min(img.shape[0], maxindex[0]+3)
            maxindex[1] = min(img.shape[1], maxindex[1]+3)
            maxindex[2] = min(img.shape[2], maxindex[2]+3)
            
            # minindex = minindex_list[count]
            # maxindex = maxindex_list[count]
            f1 = open(tumorline_list[count],'r')
            tumorline = f1.readlines()
            # tumorlines.append(tumorline)
            # tumoridx.append(len(tumorline))
            f1.close()

            f2 = open(liverline_list[count],'r')
            liverline = f2.readlines()
            # liverlines.append(liverline)
            # liveridx.append(len(liverline))
            f2.close()

            num = np.random.randint(0,6)
            if num < 3 or (count in liverlist):
                lines = liverline
                numid = len(liverline)
            else:
                lines = tumorline
                numid = len(tumorline)
            Parameter_List.append([img, tumor, lines, numid, minindex, maxindex])
        pool = ThreadPool(thread_num)
        result_list = pool.map(load_seq_crop_data_masktumor_try, Parameter_List)
        pool.close()
        pool.join()
        # result_list = load_seq_crop_data_masktumor_try(Parameter_List[0])
        # result_list = list(result_list)
        # print(type(result_list))
        # print(type(result_list[0]), ' ', result_list[0].shape)
        for idx in range(len(result_list)):
            X[idx, :, :, :, 0] = result_list[idx][0]
            Y[idx, :, :, :, 0] = result_list[idx][1]
        print("----- 0 : ", np.sum(Y==0))
        print("----- 1 : ", np.sum(Y==1)) 
        print("----- 2 : ", np.sum(Y==2))
        if np.sum(Y==0)==0:
            continue
        if np.sum(Y==1)==0:
            continue
        if np.sum(Y==2)==0:
            continue
        return (X,Y)

if __name__ == '__main__':

        #  liver tumor LITS
    trainidx = list(range(131))
    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    tumorline_list = []
    liverline_list = []
    liveridx = []
    liverBox_list = []
    liverlines = []
    for idx in range(131):
        print('preload ', idx)
        # img, img_header = load(args.data + 'myTrainingData/volume-' + str(idx) + '.nii' )
        # tumor, tumor_header = load(args.data + 'myTrainingData/segmentation-' + str(idx) + '.nii')
        img_list.append(args.data + 'myTrainingData/volume-' + str(idx) + '.nii')
        tumor_list.append(args.data + 'myTrainingData/segmentation-' + str(idx) + '.nii')
        tumorline_list.append(args.data+ 'myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt')
        liverline_list.append(args.data+ 'myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt')
        liverBox_list.append(args.data+'myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt')
        '''
        maxmin = np.loadtxt(args.data+'myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0]-3, 0)
        minindex[1] = max(minindex[1]-3, 0)
        minindex[2] = max(minindex[2]-3, 0)
        maxindex[0] = min(img.shape[0], maxindex[0]+3)
        maxindex[1] = min(img.shape[1], maxindex[1]+3)
        maxindex[2] = min(img.shape[2], maxindex[2]+3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)

        f1 = open(args.data+ 'myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt','r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()

        f2 = open(args.data+ 'myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt','r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()
        '''
    
    '''
    if not os.path.exists(args.save_path +model_path):
        os.mkdir(args.save_path + model_path)
    if not os.path.exists(args.save_path + "history"):
        os.mkdir(args.save_path + 'history')
    else:
        if os.path.exists(args.save_path + "history/lossbatch.txt"):
            os.remove(args.save_path + 'history/lossbatch.txt')
        if os.path.exists(args.save_path + "history/lossepoch.txt"):
            os.remove(args.save_path + 'history/lossepoch.txt')
    model_checkpoint = ModelCheckpoint(args.save_path + model_path+'weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose = 1,
                                       save_best_only=False,save_weights_only=False,mode = 'min', period = 1)
    '''
    x, y = generate_arrays_from_file(args.b, trainidx, img_list, tumor_list, tumorline_list, liverline_list, liverBox_list)
    
    tmpT = torch.from_numpy(y)
    tmp1 = tmpT.permute(0, 4, 3, 1, 2)
    tmp2 = tmpT.permute(3, 0, 4, 2, 1)

    # segData = y.transpose([0, 4, 3, 1, 2])
    # xxxData = y.transpose([3, 0, 4, 1, 2])
    
    fig = plt.figure()
    for i in range(8):
        a = fig.add_subplot(2, 8, i + 1)
        plt.imshow(tmp1[0][0][i])

    for i in range(8):
        a = fig.add_subplot(2, 8, 9 + i)
        plt.imshow(tmp2[i][0][0])
    plt.show()
    
    '''
    print(x.shape)
    print(y.shape)
    rawData = x.transpose([0, 4, 3, 2, 1])
    segData = y.transpose([0, 4, 3, 2, 1])

    tmp1 = rawData[0][0][0]
    tmp2 = segData[0][0][0]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (40, 20))
    ax1.imshow(tmp1, cmap = plt.cm.gray)
    ax1.set_title('Image')
    ax2.imshow(tmp2)
    ax2.set_title('Mask')

    plt.show()
    '''
    print(x.shape)
    device = 'cuda'
    model = dense_rnn_net(8).to(device)
    # inp = torch.randn(2, 224, 224, 8, 1).to(device)
    inp = torch.from_numpy(x).to(device)

    pred = model(inp).to('cpu')
    prednp = pred.detach().numpy()
    print('xxxxxxxx ', prednp.shape)
    
    '''
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (40, 20))
    ax1.imshow(prednp[0][0][0], cmap = plt.cm.gray)
    ax1.set_title('Image')
    
    plt.show()
    '''




