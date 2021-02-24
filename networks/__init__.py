# -*- coding: utf-8 -*-
# @Author: luoling
# @Date:   2019-11-26 12:13:34
# @Last Modified by:   luoling
# @Last Modified time: 2019-12-09 11:22:29

from .dfanet import DFANet
from .unet import UNet
from .danet import DANet
from .dabnet import DABNet
from .ce2p import CE2P
from .ehanet import EHANet18
from .parsenet18 import FaceParseNet18, FaceParseNet34
from .parsenet50 import FaceParseNet50, FaceParseNet101


def get_model(model, url=None, n_classes=19, pretrained=True):
    if model == 'DFANet':
        net = DFANet(num_classes=n_classes)
    elif model == 'UNet':
        net = UNet(n_classes=n_classes)
    elif model == 'DANet':
        net = DANet(num_classes=n_classes)
    elif model == 'DABNet':
        net = DABNet(classes=n_classes)
    elif model == 'CE2P':
        net = CE2P(num_classes=n_classes, url=url, pretrained=pretrained)
    elif model == 'FaceParseNet18':
        net = FaceParseNet18(num_classes=n_classes, pretrained=pretrained)
    elif model == 'FaceParseNet34':
        net = FaceParseNet34(num_classes=n_classes, pretrained=pretrained)
    elif model == 'FaceParseNet50':
        net = FaceParseNet50(num_classes=n_classes, pretrained=pretrained)
    elif model == 'FaceParseNet101':
        net = FaceParseNet101(num_classes=n_classes, pretrained=pretrained)
    elif model == 'EHANet18':
        net = EHANet18(num_classes=n_classes, pretrained=pretrained)
    else:
        raise ValueError("No corresponding model was found...")
    return net
