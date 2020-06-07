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

    
class ALLSearch_SE(nn.Module):
    def __init__(self, n_classes , dilation , deep_supervision = False, input_channels = 3 ):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.multiplier = 2
        self.layers = 5
        self.module = nn.ModuleList()
        nb_filter = [64, 256, 512, 1024, 2048]
        #dilation = [4,8,16,32]
        self.dilation =  list(dilation)
        self.resnet = build_backbone('resnet', BatchNorm = SynchronizedBatchNorm2d)
        #self.resnet = models.resnet101(pretrained=True)
        self.inifilter = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self._arch_param_names = ["branchx1_0","branchx2_0","branchx3_0","branchx4_0",
                                  "branchx1_1","branchx2_1","branchx3_1",
                                  "branchx1_2","branchx2_2",
                                  "branchx1_3",]
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        self._initialize_alphas ()
        
        for layer in range(self.layers):
            
            if layer == 0:
                self.layermodule = nn.ModuleList()
                self.module.append(self.layermodule)
            else :
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0:
                        self.layermodule.append(CBBlock(nb_filter[stage] * layer + nb_filter[stage+1], nb_filter[stage],dilation = self.dilation[stage]))

                    else :
                        self.layermodule.append(CBBlock(nb_filter[stage] * layer + nb_filter[stage-1] +  nb_filter[stage+1] , nb_filter[stage],dilation = self.dilation[stage]))

                self.module.append(self.layermodule)

    def _initialize_alphas(self):
        branchx1_0 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[0], nn.Parameter(branchx1_0))
  
        branchx2_0 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[1], nn.Parameter(branchx2_0))
        
        branchx3_0 = nn.Parameter(1e-3*torch.ones(4).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[2], nn.Parameter(branchx3_0))

        branchx4_0 = nn.Parameter(1e-3*torch.ones(4).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[3], nn.Parameter(branchx4_0))
  
        branchx1_1 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[4], nn.Parameter(branchx1_1))
        
        branchx2_1 = nn.Parameter(1e-3*torch.ones(4).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[5], nn.Parameter(branchx2_1))
        
        branchx3_1 = nn.Parameter(1e-3*torch.ones(4).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[6], nn.Parameter(branchx3_1))
        
        branchx1_2 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[7], nn.Parameter(branchx1_2))
        
        branchx2_2 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[8], nn.Parameter(branchx2_2))
        
        branchx1_3 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[9], nn.Parameter(branchx1_3))
        
    
    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        softmax1_0 = torch.softmax(self.branchx1_0, dim=-1)
        softmax2_0 = torch.softmax(self.branchx2_0, dim=-1)
        softmax3_0 = torch.softmax(self.branchx3_0, dim=-1)
        softmax4_0 = torch.softmax(self.branchx4_0, dim=-1)
        softmax1_1 = torch.softmax(self.branchx1_1, dim=-1)
        softmax2_1 = torch.softmax(self.branchx2_1, dim=-1)
        softmax3_1 = torch.softmax(self.branchx3_1, dim=-1)
        softmax1_2 = torch.softmax(self.branchx1_2, dim=-1)
        softmax2_2 = torch.softmax(self.branchx2_2, dim=-1)
        softmax1_3 = torch.softmax(self.branchx1_3, dim=-1)
        
                
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
                        layer1.append(self.module[layer][stage](torch.cat([softmax1_0[0] * layer0[stage],softmax1_0[1] * self.up(layer0[stage+1])], 1)))
                    elif stage == 1:
                        layer1.append(self.module[layer][stage](torch.cat([softmax1_1[0] * layer0[stage],softmax1_1[1] * self.up(layer0[stage+1]),softmax1_1[2] * self.pool(layer1[stage-1])], 1)))
                    elif stage == 2:
                        layer1.append(self.module[layer][stage](torch.cat([softmax1_2[0] * layer0[stage],softmax1_2[1] * self.up(layer0[stage+1]), softmax1_2[2] * self.pool(layer1[stage-1])], 1)))
                    elif stage == 3: #最後一層不乘機率
                        layer1.append(self.module[layer][stage](torch.cat([softmax1_3[0] * layer0[stage],self.up(layer0[stage+1]), softmax1_3[1] * self.pool(layer1[stage-1])], 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer2.append(self.module[layer][stage](torch.cat([softmax2_0[0] * layer0[stage],softmax2_0[1] * layer1[stage], softmax2_0[2] * self.up(layer1[stage+1])], 1)))
                    elif stage == 1:
                        layer2.append(self.module[layer][stage](torch.cat([softmax2_1[0] * layer0[stage],softmax2_1[1] * layer1[stage],softmax2_1[2] * self.up(layer1[stage+1]),softmax2_1[3] * self.pool(layer2[stage-1])], 1)))
                    elif stage == 2: #最後一層不乘機率
                        layer2.append(self.module[layer][stage](torch.cat([softmax2_2[0] * layer0[stage],softmax2_2[1] * layer1[stage],self.up(layer1[stage+1]),softmax2_2[2] * self.pool(layer2[stage-1])], 1)))
                        
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer3.append(self.module[layer][stage](torch.cat([softmax3_0[0] * layer0[stage],softmax3_0[1] * layer1[stage],softmax3_0[2] * layer2[stage],softmax3_0[3] * self.up(layer2[stage+1])], 1)))
                    
                    elif stage == 1: #最後一層不乘機率
                        layer3.append(self.module[layer][stage](torch.cat([softmax3_1[0] * layer0[stage],softmax3_1[1] * layer1[stage],softmax3_1[2] * layer2[stage],self.up(layer2[stage+1]),softmax3_1[3] * self.pool(layer3[stage-1])], 1)))
                        
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    layer4.append(self.module[layer][stage](torch.cat([softmax4_0[0] * layer0[stage],softmax4_0[1] * layer1[stage],softmax4_0[2] * layer2[stage],softmax4_0[3] * layer3[stage],self.up(layer3[stage+1])], 1)))
            
        if self.deep_supervision:
            output1 = self.final1(layer1[0])
            output2 = self.final2(layer2[0])
            output3 = self.final3(layer3[0])
            output4 = self.final4(layer4[0])
            return [output1, output2, output3, output4]
        else :
            output = self.final(layer4[0])
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

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]
def get_seg_model(cfg, **kwargs):

    model = ALLSearch_SE(n_classes = cfg.DATASET.NUM_CLASSES, dilation = cfg.MODEL.DILATION,deep_supervision = cfg.MODEL.DS)
    #model.init_weights(cfg.MODEL.PRETRAINED)

    return model
    

        
       
        
        
    
    