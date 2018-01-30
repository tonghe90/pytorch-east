import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import math
from torch.nn.init import xavier_normal


class resnet_east(nn.Module):
    def __init__(self):
        super(resnet_east, self).__init__()
        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(torch.load('./model/resnet50.pth'))
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        ) ### c = 256   1/4
        self.layer2 = resnet.layer2 ### c = 512   1/8
        self.layer3 = resnet.layer3 ### c = 1024  1/16
        self.layer4 = resnet.layer4 ### c = 2048  1/32

        bottom_channels = {'res_32': 2048, 'res_16': 1024, 'res_8': 512, 'res_4': 256}
        up_channels = {'res_32': 512, 'res_16': 256, 'res_8': 128, 'res_4': 64}

        ### upsample
        self.upsample_layer4 = nn.Sequential(
            nn.Conv2d(bottom_channels['res_32'], up_channels['res_32'], kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.upsample_layer3 = nn.Sequential(
            nn.Conv2d(bottom_channels['res_16'], up_channels['res_16'], kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.upsample_layer2 = nn.Sequential(
            nn.Conv2d(bottom_channels['res_8'], up_channels['res_8'], kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.upsample_layer1 = nn.Sequential(
            nn.Conv2d(bottom_channels['res_4'], up_channels['res_4'], kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self._weight_init([self.upsample_layer1, self.upsample_layer2, self.upsample_layer3, self.upsample_layer4])


    def _weight_init(self, layers):
        """
        :param layers: a list with layers to init
        :return: None
        """
        for layer in layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    xavier_normal(module.weight.data)
                    if module.bias:
                        xavier_normal(module.bias.data)


    def forward(self, x):
        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)


        return layer_4



x = Variable(torch.from_numpy(np.zeros((1, 3, 224, 224)))).float()
net = resnet_east()
layer_1 = net.layer1(x)
layer_2 = net.layer2(layer_1)
layer_3 = net.layer3(layer_2)
layer_4 = net.layer4(layer_3)
print 'layer1: ', layer_1.shape
print 'layer2: ', layer_2.shape
print 'layer3: ', layer_3.shape
print 'layer4: ', layer_4.shape


#
# x = Variable(torch.from_numpy(np.zeros((1,3,224,224))))
#
# for m in modules():
#     if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
# x1 = layer1(x)
# x2 = layer2(x1)
# x3 = layer3(x2)
# x4 = layer3(x3)
#
# print 'x1:', x1.shape
# print 'x2:', x2.shape
# print 'x3:', x3.shape
# print 'x4:', x4.shape
