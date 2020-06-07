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
    def __init__(self, in_channels, out_channels, dilation):
        super(CBBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation,dilation = dilation)
        self.bn2 = BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation,dilation = dilation)
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

    
class ALLSearch_SE(nn.Module):
    def __init__(self, n_classes , dilation , deep_supervision = False, input_channels = 3 ):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.multiplier = 2
        self.layers = 9
        self.stages = 5
        self.module = nn.ModuleList()
        nb_filter = [64, 256, 512, 1024, 2048]
        out_filter = [32, 64, 128, 256, 512]
        #dilation = [4,8,16,32]
        self.dilation =  list(dilation)
        self.resnet = build_backbone('dilated_resnet', output_stride=16, BatchNorm = SynchronizedBatchNorm2d)
        #self.resnet = models.resnet101(pretrained=True)
        self.inifilter = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
            self.final5 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
            self.final6 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
            self.final7 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
            self.final8 = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(out_filter[0], self.n_classes, kernel_size=1)
        
        for layer in range(self.layers):
            
            if layer == 0:
                self.layermodule = nn.ModuleList()
                self.module.append(self.layermodule)
            elif layer == 1:
                self.layermodule = nn.ModuleList()
                for stage in range(self.stages):
                    if stage == 0:
                        self.layermodule.append(CBBlock(nb_filter[stage] + nb_filter[stage+1], out_filter[stage],dilation = self.dilation[stage]))
                    elif 0 < stage < 4 :
                        self.layermodule.append(CBBlock(nb_filter[stage] + nb_filter[stage-1] +  nb_filter[stage+1] , out_filter[stage],dilation = self.dilation[stage]))
                    else :
                        self.layermodule.append(CBBlock(nb_filter[stage] + nb_filter[stage-1] , out_filter[stage],dilation = self.dilation[stage]))
                        
                self.module.append(self.layermodule)
            elif layer == 2 :
                self.layermodule = nn.ModuleList()
                for stage in range(self.stages):
                    
                    if stage == 0:
                        self.layermodule.append(CBBlock(nb_filter[stage] + out_filter[stage] +  out_filter[stage+1],  out_filter[stage],dilation = self.dilation[stage]))
                    elif 0 < stage < 4 :
                        self.layermodule.append(CBBlock(nb_filter[stage] + out_filter[stage] +  out_filter[stage-1] +   out_filter[stage+1] ,  out_filter[stage],dilation = self.dilation[stage]))
                    else :
                        self.layermodule.append(CBBlock(nb_filter[stage] + out_filter[stage] +  out_filter[stage-1] ,  out_filter[stage],dilation = self.dilation[stage]))
                self.module.append(self.layermodule)
            else :
                self.layermodule = nn.ModuleList()
                for stage in range(self.stages):
                    
                    if stage == 0:
                        self.layermodule.append(CBBlock( out_filter[stage] * 2 +  out_filter[stage+1],  out_filter[stage],dilation = self.dilation[stage]))
                    elif 0 < stage < 4 :
                        self.layermodule.append(CBBlock( out_filter[stage] * 2 +  out_filter[stage-1] +   out_filter[stage+1] , out_filter[stage],dilation = self.dilation[stage]))
                    else :
                        self.layermodule.append(CBBlock( out_filter[stage] * 2 +  out_filter[stage-1] ,  out_filter[stage],dilation = self.dilation[stage]))

                self.module.append(self.layermodule)
    
    def forward(self, input): 
        layeroutput = []
                
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                layerin = []
                stem,outlayer1,outlayer2,outlayer3,outlayer4 = self.resnet(input)
                layerin.append(stem)
                layerin.append(outlayer1)
                layerin.append(outlayer2)
                layerin.append(outlayer3)
                layerin.append(outlayer4)
                layeroutput.append(layerin)
            elif layer == 1:
                layerin = []
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],self.up(layeroutput[layer-1][stage+1])], 1)))
                    elif 0 < stage < 3:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],self.up(layeroutput[layer-1][stage+1]), self.pool(layeroutput[layer-1][stage-1])], 1)))
                    elif stage == 3:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],layeroutput[layer-1][stage+1], self.pool(layeroutput[layer-1][stage-1])], 1)))
                    else:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],layeroutput[layer-1][stage-1]], 1)))
                layeroutput.append(layerin)
            else :
                layerin = []
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],layeroutput[layer-2][stage],self.up(layeroutput[layer-1][stage+1])], 1)))
                    elif 0 <stage < 3:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],layeroutput[layer-2][stage],self.up(layeroutput[layer-1][stage+1]), self.pool(layeroutput[layer-1][stage-1])], 1)))
                    elif stage == 3:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],layeroutput[layer-2][stage],layeroutput[layer-1][stage+1], self.pool(layeroutput[layer-1][stage-1])], 1)))
                    else:
                        layerin.append(self.module[layer][stage](torch.cat([layeroutput[layer-1][stage],layeroutput[layer-2][stage],layeroutput[layer-1][stage-1]], 1)))
                layeroutput.append(layerin)
            
        if self.deep_supervision:
            output1 = self.final1(layeroutput[1][0])
            output2 = self.final2(layeroutput[2][0])
            output3 = self.final3(layeroutput[3][0])
            output4 = self.final4(layeroutput[4][0])
            output5 = self.final5(layeroutput[5][0])
            output6 = self.final6(layeroutput[6][0])
            output7 = self.final7(layeroutput[7][0])
            output8 = self.final8(layeroutput[8][0])
            return [output1, output2, output3, output4,output5,output6,output7,output8]
        else :
            output = self.final(layeroutput[8][0])
            return output   
                                                
    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained,map_location="cuda:0")
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

    model = ALLSearch_SE(n_classes = cfg.DATASET.NUM_CLASSES, dilation = cfg.MODEL.DILATION,deep_supervision = cfg.MODEL.DS)
    #model.init_weights(cfg.MODEL.PRETRAINED)

    return model
    

        
       
        
        
    
    