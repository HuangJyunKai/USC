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
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from models.backbone import build_backbone

logger = logging.getLogger(__name__)

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, BatchNorm, output_stride)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
            '''
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            '''
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
'''
if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
'''
def get_seg_model(cfg, **kwargs):

    model = DeepLab(backbone='dilated_resnet',output_stride=16, num_classes = cfg.DATASET.NUM_CLASSES,sync_bn=True)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model


