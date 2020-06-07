from models.backbone import dilated_resnet, xception, drn, mobilenet,resnet

def build_backbone(backbone, BatchNorm,output_stride=None):
    if backbone == 'dilated_resnet':
        return dilated_resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet':
        return resnet.ResNet101( BatchNorm)
    elif backbone == 'resnet18':
        return resnet18.ResNet18( BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception( BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2( BatchNorm)
    else:
        raise NotImplementedError
