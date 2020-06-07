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
from .Search_resnet101_baseblock_softmax_sybn_edit import ALLSearch_SE
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.backbone import build_backbone

logger = logging.getLogger(__name__)

class CBBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
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

class SEModule(nn.Module): 
    def __init__(self, channels, reduction): 
        super(SEModule, self).__init__() 
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0) 
        self.relu = nn.ReLU(inplace=True) 
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid() 
    def forward(self, x): 
        module_input = x 
        x = self.avg_pool(x) 
        x = self.fc1(x) 
        x = self.relu(x) 
        x = self.fc2(x) 
        x = self.sigmoid(x) 
        
        return module_input * x

    
class ALLSearch_Decode_SE(nn.Module):
    def __init__(self, weight, n_classes , input_channels = 3 ,):
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
        
        self.weight = weight 
        self.weight_decode = self.roundsoftmax()
        self.branchx1_0 = self.weight_decode[0]
        self.branchx2_0 = self.weight_decode[1]
        self.branchx3_0 = self.weight_decode[2]
        self.branchx4_0 = self.weight_decode[3]
        self.branchx1_1 = self.weight_decode[4]
        self.branchx2_1 = self.weight_decode[5]
        self.branchx3_1 = self.weight_decode[6]
        self.branchx1_2 = self.weight_decode[7]
        self.branchx2_2 = self.weight_decode[8]
        self.branchx1_3 = self.weight_decode[9]
        
        for layer in range(self.layers):
            if layer == 0:
                self.layermodule = nn.ModuleList()
                self.module.append(self.layermodule)
                
            elif layer == 1:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0:
                        self.layermodule.append(CBBlock(self.branchx1_0[0] * nb_filter[stage] + self.branchx1_0[1] * nb_filter[stage+1], nb_filter[stage]))
                    
                    elif stage == 1:
                        self.layermodule.append(CBBlock(self.branchx1_1[0] * nb_filter[stage] + self.branchx1_1[1] * nb_filter[stage+1] + self.branchx1_1[2] * nb_filter[stage-1], nb_filter[stage]))
                    elif stage == 2:
                        self.layermodule.append(CBBlock(self.branchx1_2[0] * nb_filter[stage] + self.branchx1_2[1] * nb_filter[stage+1] + self.branchx1_2[2] * nb_filter[stage-1], nb_filter[stage]))
                        
                    else :
                        self.layermodule.append(CBBlock(self.branchx1_3[0] * nb_filter[stage] + nb_filter[stage+1] + self.branchx1_3[1] * nb_filter[stage-1] , nb_filter[stage]))
                
                self.module.append(self.layermodule)
                
            elif layer == 2:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0: 
                        self.layermodule.append(CBBlock((self.branchx2_0[0] + self.branchx2_0[1]) * nb_filter[stage] + self.branchx2_0[2] * nb_filter[stage+1], nb_filter[stage]))
                    
                    elif stage == 1:    
                        self.layermodule.append(CBBlock((self.branchx2_1[0] + self.branchx2_1[1] ) * nb_filter[stage] + self.branchx2_1[2] * nb_filter[stage+1] + self.branchx2_1[3]  * nb_filter[stage-1], nb_filter[stage]))
                            
                    else :
                        self.layermodule.append(CBBlock((self.branchx2_2[0] + self.branchx2_2[1]) * nb_filter[stage] + nb_filter[stage+1] + self.branchx2_2[2] * nb_filter[stage-1] , nb_filter[stage]))
                
                self.module.append(self.layermodule)
                
                
            elif layer == 3:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0:
                        self.layermodule.append(CBBlock((self.branchx3_0[0] + self.branchx3_0[1] + self.branchx3_0[2]) * nb_filter[stage] + self.branchx3_0[3] * nb_filter[stage+1], nb_filter[stage]))
                    
                    else:    
                        self.layermodule.append(CBBlock((self.branchx3_1[0] + self.branchx3_1[1] + self.branchx3_1[2] ) * nb_filter[stage] + nb_filter[stage+1] + self.branchx3_1[3] * nb_filter[stage-1]  , nb_filter[stage]))
                            
                self.module.append(self.layermodule)
                
            elif layer == 4:
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):  
                    self.layermodule.append(CBBlock((self.branchx4_0[0] + self.branchx4_0[1] + self.branchx4_0[2] + self.branchx4_0[3]) * nb_filter[stage] + nb_filter[stage+1], nb_filter[stage]))
                    
                self.module.append(self.layermodule)           
        
    def roundsoftmax(self):
        lst = []
        result = []
        for i in range(len(self.weight)):
            self.weight[i] = torch.softmax(self.weight[i],dim=-1)
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                if self.weight[i][j] >= 1/len(self.weight[i]):
                    lst.append(1)
                else:
                    lst.append(0)
            result.append(lst)
            lst = []
                
        print(self.weight)
        print(result)
        return result

    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                stem,outlayer1,outlayer2,outlayer3,outlayer4 = self.resnet(input)
                layer0.append(stem)
                layer0.append(outlayer1)
                layer0.append(outlayer2)
                layer0.append(outlayer3)
                layer0.append(outlayer4)
                        
            elif layer == 1:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        concatlst = []
                        if self.branchx1_0[0] == 1:
                            concatlst.append(layer0[stage])
                        if self.branchx1_0[1] == 1:
                            concatlst.append(self.up(layer0[stage+1]))
                        layer1.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                    elif stage == 1: 
                        concatlst = []
                        if self.branchx1_1[0] == 1:
                            concatlst.append(layer0[stage])
                        if self.branchx1_1[1] == 1:
                            concatlst.append(self.up(layer0[stage+1]))
                        if self.branchx1_1[2] == 1:
                            concatlst.append(self.pool(layer1[stage-1]))
                        layer1.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                    elif stage == 2: 
                        concatlst = []
                        if self.branchx1_2[0] == 1:
                            concatlst.append(layer0[stage])
                        if self.branchx1_2[1] == 1:
                            concatlst.append(self.up(layer0[stage+1]))
                        if self.branchx1_2[2] == 1:
                            concatlst.append(self.pool(layer1[stage-1]))
                        layer1.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                    else : #最後一層up不乘機率 
                        concatlst = []
                        if self.branchx1_3[0] == 1:
                            concatlst.append(layer0[stage])
                        if self.branchx1_3[1] == 1:
                            concatlst.append(self.pool(layer1[stage-1]))
                        concatlst.append(self.up(layer0[stage+1]))
                        layer1.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        concatlst = []
                        if self.branchx2_0[0] == 1:
                            concatlst.append(layer0[stage])
                        if self.branchx2_0[1] == 1:
                            concatlst.append(layer1[stage])
                        if self.branchx2_0[2] == 1:
                            concatlst.append(self.up(layer1[stage+1]))
                        layer2.append(self.module[layer][stage](torch.cat(concatlst, 1)))
                    elif stage == 1: 
                        concatlst = []
                        if self.branchx2_1[0] == 1:
                            concatlst.append(layer0[stage])
                        if self.branchx2_1[1] == 1:
                            concatlst.append(layer1[stage])
                        if self.branchx2_1[2] == 1:
                            concatlst.append(self.up(layer1[stage+1]))
                        if self.branchx2_1[3] == 1:
                            concatlst.append(self.pool(layer2[stage-1]))
                        layer2.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                    else : #最後一層不乘機率
                        concatlst = []
                        if self.branchx2_2[0] == 1:
                            concatlst.append(layer0[stage])
                        if self.branchx2_2[1] == 1:
                            concatlst.append(layer1[stage])
                        if self.branchx2_2[2] == 1:
                            concatlst.append(self.pool(layer2[stage-1]))
                        concatlst.append(self.up(layer1[stage+1]))
                        layer2.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                        
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        concatlst = []
                        if self.branchx3_0[0] == 1:
                            concatlst.append(layer0[stage]) 
                        if self.branchx3_0[1] == 1:
                            concatlst.append(layer1[stage]) 
                        if self.branchx3_0[2] == 1:
                            concatlst.append(layer2[stage]) 
                        if self.branchx3_0[3] == 1:
                            concatlst.append(self.up(layer2[stage+1])) 
                        layer3.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                    else : 
                        concatlst = []
                        if self.branchx3_1[0] == 1:
                            concatlst.append(layer0[stage]) 
                        if self.branchx3_1[1] == 1:
                            concatlst.append(layer1[stage]) 
                        if self.branchx3_1[2] == 1:
                            concatlst.append(layer2[stage]) 
                        if self.branchx3_1[3] == 1:
                            concatlst.append(self.pool(layer3[stage-1])) 
                        concatlst.append(self.up(layer2[stage+1])) 
                        layer3.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
                    
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    concatlst = []
                    if self.branchx4_0[0] == 1:
                        concatlst.append(layer0[stage]) 
                    if self.branchx4_0[1] == 1:
                        concatlst.append(layer1[stage]) 
                    if self.branchx4_0[2] == 1:
                        concatlst.append(layer2[stage]) 
                    if self.branchx4_0[3] == 1:
                        concatlst.append(layer3[stage]) 
                    concatlst.append(self.up(layer3[stage+1])) 
                    layer4.append(self.module[layer][stage](torch.cat(concatlst, 1))) 
            
        output = self.final(layer4[0])
              
        return output
    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                           if k[6:] in model_dict.keys() and model_dict[k[6:]].size() == pretrained_dict[k].size()}
            '''
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            '''
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    
def get_seg_model(cfg, **kwargs):
    #model = HighResolutionNet(cfg, **kwargs)
    model_pre = ALLSearch_SE(n_classes = cfg.DATASET.NUM_CLASSES)
    model_pre.init_weights(cfg.MODEL.PRETRAINED)
    para = model_pre.arch_parameters()
    model = ALLSearch_Decode_SE(para , n_classes = cfg.DATASET.NUM_CLASSES)
    #model.init_weights(cfg.MODEL.PRETRAINED)

    return model
    

        
       
        
        
    
    