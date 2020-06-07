# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from torchvision import models
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.backbone import build_backbone

logger = logging.getLogger(__name__)

def BatchNorm2d(out_channels):
    return SynchronizedBatchNorm2d(out_channels)

class CBBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out



class NestedUnet(nn.Module): # only maxpooling
    def __init__(self, n_classes , input_channels = 3 ):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.multiplier = 2
        self.layers = 5
        self.module = nn.ModuleList()
        nb_filter = [64, 256, 512, 1024, 2048]
        self.resnet = build_backbone('resnet', BatchNorm = SynchronizedBatchNorm2d)
        self.inifilter = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(self.inifilter ,n_classes , kernel_size=1)
        
        self.conv0_1 = CBBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = CBBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = CBBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = CBBlock(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = CBBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = CBBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = CBBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = CBBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = CBBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = CBBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        
        self.final = nn.Conv2d(self.inifilter,n_classes , kernel_size=1)


    def forward(self, input):
        
        x0_0,x1_0,x2_0,x3_0,x4_0 = self.resnet(input)

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output    

                                                
    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                           if k[6:] in model_dict.keys()}
            '''
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            '''
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    
def get_seg_model(cfg, **kwargs):

    model = NestedUnet(n_classes = cfg.DATASET.NUM_CLASSES)
    #model.init_weights(cfg.MODEL.PRETRAINED)

    return model
    

        
       
        
        
    
    