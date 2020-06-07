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

class Stemresnet101(nn.Module):
    def __init__(self, resnet):
        super(Stemresnet101, self).__init__()
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
    def forward(self, x):
        return self.stem(x)

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
    def __init__(self, n_classes , input_channels = 3 ):
        super().__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.multiplier = 2
        self.layers = 5
        self.module = nn.ModuleList()
        nb_filter = [64, 256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.inifilter = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conc = nn.Conv2d(nb_filter[0] * 2, nb_filter[0] * 2, kernel_size=1)
        self.final = nn.Conv2d(self.inifilter * 2,n_classes , kernel_size=1)
        
        self._arch_param_names = ["routesdw0","routesdw1","routesdw2",
                                  "routesup0","routesup1","routesup2",
                                  "routessk01","routessk12","routessk23","routessk34",
                                  "routessk02","routessk13","routessk24",
                                  "routessk03","routessk14","routessk04"]
        self._initialize_alphas ()
        
        for layer in range(self.layers):
            if layer == 0:
                if layer == 0:
                    self.layermodule = nn.ModuleList()
                    self.layermodule.append(Stemresnet101(resnet))
                    self.layermodule.append(resnet.layer1)
                    self.layermodule.append(resnet.layer2)
                    self.layermodule.append(resnet.layer3)
                    self.layermodule.append(resnet.layer4)
                self.module.append(self.layermodule)
            else :
                self.layermodule = nn.ModuleList()
                for stage in range(self.layers-layer):
                    if stage == 0:
                        self.layermodule.append(CBBlock(nb_filter[stage] * layer + nb_filter[stage+1], nb_filter[stage]))
                            
                    else :
                        self.layermodule.append(CBBlock(nb_filter[stage] * layer + nb_filter[stage-1] +  nb_filter[stage+1] , nb_filter[stage]))
                
                self.module.append(self.layermodule)         
            
        self.conv3 = CBBlock(nb_filter[3] + nb_filter[4],  nb_filter[3])
        self.conv2 = CBBlock(nb_filter[2] + nb_filter[3],  nb_filter[2])
        self.conv1 = CBBlock(nb_filter[1] + nb_filter[2],  nb_filter[1])
        self.conv0 = CBBlock(nb_filter[0] + nb_filter[1],  nb_filter[0])
        
        self.se = SEModule(nb_filter[0] * 2 , 2)
        
        
    def _initialize_alphas(self):
        routesdw0 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[0], nn.Parameter(routesdw0))
  
        routesdw1 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[1], nn.Parameter(routesdw1))
        
        routesdw2 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[2], nn.Parameter(routesdw2))

        routesup0 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[3], nn.Parameter(routesup0))
  
        routesup1 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[4], nn.Parameter(routesup1))
        
        routesup2 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[5], nn.Parameter(routesup2))
        
        routessk01 = nn.Parameter(1e-3*torch.ones(4).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[6], nn.Parameter(routessk01))
        
        routessk12 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[7], nn.Parameter(routessk12))
        
        routessk23 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[8], nn.Parameter(routessk23))
        
        routessk34 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[9], nn.Parameter(routessk34))
        
        routessk02 = nn.Parameter(1e-3*torch.ones(3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[10], nn.Parameter(routessk02))
        
        routessk13 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[11], nn.Parameter(routessk13))
        
        routessk24 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[12], nn.Parameter(routessk24))
        
        routessk03 = nn.Parameter(1e-3*torch.ones(2).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[13], nn.Parameter(routessk03))
        
        routessk14 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[14], nn.Parameter(routessk14))
        
        routessk04 = nn.Parameter(1e-3*torch.ones(1).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[15], nn.Parameter(routessk04))
        

    def forward(self, input):  
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layerbase = []
        for layer , mlayer in enumerate(self.module): 
            
            if layer == 0 :
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer0.append(self.module[layer][stage](input))        
                    elif stage == 1:
                        layer0.append(self.module[layer][stage](self.pool(layer0[stage-1])))
                    else :
                        layer0.append(self.module[layer][stage](layer0[stage-1]))
                        
            elif layer == 1:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],torch.sigmoid(self.routesup0[stage]) * self.up(layer0[stage+1])], 1)))
                    if stage > 0 and stage < len(mlayer) - 1: 
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],torch.sigmoid(self.routesup0[stage]) * self.up(layer0[stage+1]),torch.sigmoid(self.routesdw0[stage-1]) * self.pool(layer1[stage-1])], 1)))
                    if  stage == len(mlayer) - 1: #最後一層不乘機率
                        layer1.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk01[stage]) * layer0[stage],self.up(layer0[stage+1]),torch.sigmoid(self.routesdw0[stage-1]) * self.pool(layer1[stage-1])], 1)))
                        
            elif layer == 2:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],torch.sigmoid(self.routesup1[stage]) * self.up(layer1[stage+1])], 1)))
                    if stage > 0  and stage < len(mlayer) - 1: 
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],torch.sigmoid(self.routesup1[stage]) * self.up(layer1[stage+1]),torch.sigmoid(self.routesdw1[stage-1]) * self.pool(layer2[stage-1])], 1)))
                    if  stage == len(mlayer) - 1: #最後一層不乘機率
                        layer2.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk02[stage]) * layer0[stage],torch.sigmoid(self.routessk12[stage]) * layer1[stage],self.up(layer1[stage+1]),torch.sigmoid(self.routesdw1[stage-1]) * self.pool(layer2[stage-1])], 1)))
                        
            elif layer == 3:
                for stage ,layers in enumerate(mlayer) :
                    if stage == 0:
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],torch.sigmoid(self.routesup2[stage]) * self.up(layer2[stage+1])], 1)))
                    if stage > 0  and stage < len(mlayer) - 1: 
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],torch.sigmoid(self.routesup2[stage]) * self.up(layer2[stage+1]),torch.sigmoid(self.routesdw2[stage-1]) * self.pool(layer3[stage-1])], 1)))
                    if  stage == len(mlayer) - 1: #最後一層不乘機率
                        layer3.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk03[stage]) * layer0[stage],torch.sigmoid(self.routessk13[stage]) * layer1[stage],torch.sigmoid(self.routessk23[stage]) * layer2[stage],self.up(layer2[stage+1]),torch.sigmoid(self.routesdw2[stage-1]) * self.pool(layer3[stage-1])], 1)))
                        
            elif layer == 4:
                for stage ,layers in enumerate(mlayer) :
                    layer4.append(self.module[layer][stage](torch.cat([torch.sigmoid(self.routessk04[stage]) * layer0[stage],torch.sigmoid(self.routessk14[stage]) * layer1[stage],torch.sigmoid(self.routessk24[stage]) * layer2[stage],torch.sigmoid(self.routessk34[stage]) * layer3[stage],self.up(layer3[stage+1])], 1)))
                               
            
        layerbase.append(self.conv3(torch.cat([self.up(layer0[4]),layer0[3]], 1)) )
        layerbase.append(self.conv2(torch.cat([self.up(layer0[3]),layer0[2]], 1)) )
        layerbase.append(self.conv1(torch.cat([self.up(layer0[2]),layer0[1]], 1)) )
        layerbase.append(self.conv0(torch.cat([self.up(layer0[1]),layer0[0]], 1)) )
            
        fusion = self.se(self.conc(torch.cat([layer4[0], layerbase[3]], 1)))
            
        output = self.final(fusion)
              
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

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]
def get_seg_model(cfg, **kwargs):

    model = ALLSearch_SE(n_classes = cfg.DATASET.NUM_CLASSES)
    #model.init_weights(cfg.MODEL.PRETRAINED)

    return model
    

        
       
        
        
    
    