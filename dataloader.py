import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchsummary import summary
import argparse
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils import data
import matplotlib.pyplot as plt
from medpy.io import load
from skimage.transform import resize
from multiprocessing.dummy import Pool as ThreadPool

testPath = 'digit-recognizer/test.csv'

class TestMnist(data.Dataset):
    def __init__(self, pathFolder, transform=None):
        self.liverlist = [32,34,38,41,47,87,89,91,105,106,114,115,119]
        self.img_list = []
        self.tumor_list = []
        self.tumorline_list = []
        self.liverline_list = []
        self.liverBox_list = []
        for idx in range(131):
            print('preload ', idx)
            # img, img_header = load(args.data + 'myTrainingData/volume-' + str(idx) + '.nii' )
            # tumor, tumor_header = load(args.data + 'myTrainingData/segmentation-' + str(idx) + '.nii')
            self.img_list.append(pathFolder + 'myTrainingData/volume-' + str(idx) + '.nii')
            self.tumor_list.append(pathFolder + 'myTrainingData/segmentation-' + str(idx) + '.nii')
            self.tumorline_list.append(pathFolder+ 'myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt')
            self.liverline_list.append(pathFolder+ 'myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt')
            self.liverBox_list.append(pathFolder+'myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt')
        self.trans = transform
        print('init done')

    def load_seq_crop_data_masktumor_try(self, Parameter_List):
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

    def __len__(self):
        'Denotes the total number of samples'
        return 131

    def __getitem__(self, idx):
        'Generates one sample of data'
        if idx >= len(self):
            raise StopIteration
        print("gen : ", idx)
        img, img_header = load(self.img_list[idx] )
        tumor, tumor_header = load(self.tumor_list[idx])
        maxmin = np.loadtxt(self.liverBox_list[idx], delimiter=' ')
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
        

        f1 = open(self.tumorline_list[idx],'r')
        tumorline = f1.readlines()
        f1.close()

        f2 = open(self.liverline_list[idx],'r')
        liverline = f2.readlines()
        f2.close()

        num = np.random.randint(0,6)
        if num < 3 or (idx in self.liverlist):
            lines = liverline
            numid = len(liverline)
        else:
            lines = tumorline
            numid = len(tumorline)
        Parameter_List = [img, tumor, lines, numid, minindex, maxindex]

        
        X, Y = self.load_seq_crop_data_masktumor_try(Parameter_List)
        c0 = np.sum(Y==0) 
        c1 = np.sum(Y==1) 
        c2 = np.sum(Y==2)
        print(c0, ' ', c1, ' ' , c2)
        # if (c0 * c1 * c2 == 0):
        #   continue
        return X.transpose([2, 0, 1]), Y.transpose([2, 0, 1])
            


        