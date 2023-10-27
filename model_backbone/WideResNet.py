import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ResNet


class WideBottleneck(ResNet.Bottleneck):
    expansion = 2


def generate_model(model_depth, k=2, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    inplanes = [x * k for x in resnet.get_inplanes()]
    if model_depth == 50:
        model = ResNet.ResNet(WideBottleneck, [3, 4, 6, 3], inplanes, **kwargs)
    elif model_depth == 101:
        model = ResNet.ResNet(WideBottleneck, [3, 4, 23, 3], inplanes, **kwargs)
    elif model_depth == 152:
        model = ResNet.ResNet(WideBottleneck, [3, 8, 36, 3], inplanes, **kwargs)
    elif model_depth == 200:
        model = ResNet.ResNet(WideBottleneck, [3, 24, 36, 3], inplanes, **kwargs)

    return model
