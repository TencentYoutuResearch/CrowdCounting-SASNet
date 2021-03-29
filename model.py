# Copyright 2021 Tencent

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# the main implementation of the SASNet
class SASNet(nn.Module):
    def __init__(self, pretrained=False, args=None):
        super(SASNet, self).__init__()
        # define the backbone network
        vgg = models.vgg16_bn(pretrained=pretrained)

        features = list(vgg.features.children())
        # get each stage of the backbone
        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])
        self.features5 = nn.Sequential(*features[33:43])
        # docoder definition
        self.de_pred5 = nn.Sequential(
            Conv2d(512, 1024, 3, same_padding=True, NL='relu'),
            Conv2d(1024, 512, 3, same_padding=True, NL='relu'),
        )

        self.de_pred4 = nn.Sequential(
            Conv2d(512 + 512, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 256, 3, same_padding=True, NL='relu'),
        )

        self.de_pred3 = nn.Sequential(
            Conv2d(256 + 256, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 128, 3, same_padding=True, NL='relu'),
        )

        self.de_pred2 = nn.Sequential(
            Conv2d(128 + 128, 128, 3, same_padding=True, NL='relu'),
            Conv2d(128, 64, 3, same_padding=True, NL='relu'),
        )

        self.de_pred1 = nn.Sequential(
            Conv2d(64 + 64, 64, 3, same_padding=True, NL='relu'),
            Conv2d(64, 64, 3, same_padding=True, NL='relu'),
        )
        # density head definition
        self.density_head5 = nn.Sequential(
            MultiBranchModule(512),
            Conv2d(2048, 1, 1, same_padding=True)
        )

        self.density_head4 = nn.Sequential(
            MultiBranchModule(256),
            Conv2d(1024, 1, 1, same_padding=True)
        )

        self.density_head3 = nn.Sequential(
            MultiBranchModule(128),
            Conv2d(512, 1, 1, same_padding=True)
        )

        self.density_head2 = nn.Sequential(
            MultiBranchModule(64),
            Conv2d(256, 1, 1, same_padding=True)
        )

        self.density_head1 = nn.Sequential(
            MultiBranchModule(64),
            Conv2d(256, 1, 1, same_padding=True)
        )
        # confidence head definition
        self.confidence_head5 = nn.Sequential(
            Conv2d(512, 256, 1, same_padding=True, NL='relu'),
            Conv2d(256, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head4 = nn.Sequential(
            Conv2d(256, 128, 1, same_padding=True, NL='relu'),
            Conv2d(128, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head3 = nn.Sequential(
            Conv2d(128, 64, 1, same_padding=True, NL='relu'),
            Conv2d(64, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head2 = nn.Sequential(
            Conv2d(64, 32, 1, same_padding=True, NL='relu'),
            Conv2d(32, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head1 = nn.Sequential(
            Conv2d(64, 32, 1, same_padding=True, NL='relu'),
            Conv2d(32, 1, 1, same_padding=True, NL=None)
        )

        self.block_size = args.block_size
    # the forward process
    def forward(self, x):
        size = x.size()
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        # begining of decoding
        x = self.de_pred5(x5)
        x5_out = x
        x = F.upsample_bilinear(x, size=x4.size()[2:])

        x = torch.cat([x4, x], 1)
        x = self.de_pred4(x)
        x4_out = x
        x = F.upsample_bilinear(x, size=x3.size()[2:])

        x = torch.cat([x3, x], 1)
        x = self.de_pred3(x)
        x3_out = x
        x = F.upsample_bilinear(x, size=x2.size()[2:])

        x = torch.cat([x2, x], 1)
        x = self.de_pred2(x)
        x2_out = x
        x = F.upsample_bilinear(x, size=x1.size()[2:])

        x = torch.cat([x1, x], 1)
        x = self.de_pred1(x)
        x1_out = x
        # density prediction
        x5_density = self.density_head5(x5_out)
        x4_density = self.density_head4(x4_out)
        x3_density = self.density_head3(x3_out)
        x2_density = self.density_head2(x2_out)
        x1_density = self.density_head1(x1_out)
        # get patch features for confidence prediction
        x5_confi = F.adaptive_avg_pool2d(x5_out, output_size=(size[-2] // self.block_size, size[-1] // self.block_size))
        x4_confi = F.adaptive_avg_pool2d(x4_out, output_size=(size[-2] // self.block_size, size[-1] // self.block_size))
        x3_confi = F.adaptive_avg_pool2d(x3_out, output_size=(size[-2] // self.block_size, size[-1] // self.block_size))
        x2_confi = F.adaptive_avg_pool2d(x2_out, output_size=(size[-2] // self.block_size, size[-1] // self.block_size))
        x1_confi = F.adaptive_avg_pool2d(x1_out, output_size=(size[-2] // self.block_size, size[-1] // self.block_size))
        # confidence prediction
        x5_confi = self.confidence_head5(x5_confi)
        x4_confi = self.confidence_head4(x4_confi)
        x3_confi = self.confidence_head3(x3_confi)
        x2_confi = self.confidence_head2(x2_confi)
        x1_confi = self.confidence_head1(x1_confi)
        # upsample the density prediction to be the same with the input size
        x5_density = F.upsample_nearest(x5_density, size=x1.size()[2:])
        x4_density = F.upsample_nearest(x4_density, size=x1.size()[2:])
        x3_density = F.upsample_nearest(x3_density, size=x1.size()[2:])
        x2_density = F.upsample_nearest(x2_density, size=x1.size()[2:])
        x1_density = F.upsample_nearest(x1_density, size=x1.size()[2:])
        # upsample the confidence prediction to be the same with the input size
        x5_confi_upsample = F.upsample_nearest(x5_confi, size=x1.size()[2:])
        x4_confi_upsample = F.upsample_nearest(x4_confi, size=x1.size()[2:])
        x3_confi_upsample = F.upsample_nearest(x3_confi, size=x1.size()[2:])
        x2_confi_upsample = F.upsample_nearest(x2_confi, size=x1.size()[2:])
        x1_confi_upsample = F.upsample_nearest(x1_confi, size=x1.size()[2:])

        # =============================================================================================================
        # soft √
        confidence_map = torch.cat([x5_confi_upsample, x4_confi_upsample,
                                    x3_confi_upsample, x2_confi_upsample, x1_confi_upsample], 1)
        confidence_map = torch.nn.functional.sigmoid(confidence_map)

        # use softmax to normalize
        confidence_map = torch.nn.functional.softmax(confidence_map, 1)

        density_map = torch.cat([x5_density, x4_density, x3_density, x2_density, x1_density], 1)
        # soft selection
        density_map *= confidence_map
        density = torch.sum(density_map, 1, keepdim=True)

        return density

# the module definition for the multi-branch in the density head
class MultiBranchModule(nn.Module):
    def __init__(self, in_channels, sync=False):
        super(MultiBranchModule, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch1x1_1 = BasicConv2d(in_channels//2, in_channels, kernel_size=1, sync=sync)

        self.branch3x3_1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch3x3_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=(3, 3), padding=(1, 1), sync=sync)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch3x3dbl_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=5, padding=2, sync=sync)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.branch1x1_1(branch1x1)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        outputs = [branch1x1, branch3x3, branch3x3dbl, x]
        return torch.cat(outputs, 1)

# the module definition for the basic conv module
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, sync=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        if sync:
            # for sync bn
            print('use sync inception')
            self.bn = nn.SyncBatchNorm(out_channels, eps=0.001)
        else:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)