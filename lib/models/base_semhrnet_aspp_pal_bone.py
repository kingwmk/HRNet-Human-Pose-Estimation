# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from models.layers import SemanticMultiGroupConv

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)




class ASPP(nn.Module):

    def __init__(self, in_ch, out_ch, groups = 16, dilation_series=[1,2,3,4], padding_series=[1,2,3,4]):
        super(ASPP, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.groups = groups
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, dilation=dilation, groups=groups, bias = True))
#            self.conv2d_list.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))
        
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class SemanticMultiGroupConv(nn.Module):
  
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1 ,groups=16):
        super(SemanticMultiGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride 
        self.padding = padding 

    #                右脚踝     右膝盖   右臀    左臀   左膝盖  左脚踝   骨盆    胸膛   上颈部   头顶    右手腕   右肘部  右肩膀  左肩膀   左肘部  左手腕 
#                r ankle  r knee  r hip  l hip   l knee  l ankle pelvis  thorax up neck  headtop r wrist r elbow rshlde lshlde  lelbow  lwrist
        bone = np.array([[ 1 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 1 ,     1 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     1 ,     1 ,     1 ,     0 ,     0 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     1 ,     1 ,     1 ,     0 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     1 ,     1 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     1 ,     1 ,     0 ,     0 ,     1 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     1 ,     0 ,     0 ,     0 ,     1 ,     1 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     0 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     1 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     0 ,     0 ,     0 ,     1 ,     1 ,     0 ,     0 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     0],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1 ,     1],
                 [ 0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     0 ,     1 ,     1]
                ])

        self.bone = torch.from_numpy(bone)

        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, \
                "head number can not be divided by input channels"
        assert self.out_channels % self.groups == 0, \
                "head number can not be divided by output channels"

        self.gconv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                padding, dilation, groups, bias=False)
        
        self.grid = 25
        aff_out_channels = self.grid * out_channels
        kernel_size = 1
        self.gconv2 = nn.Conv2d(out_channels, aff_out_channels, kernel_size, stride, 
                padding, dilation, groups, bias=False)
        self.norm2 = nn.BatchNorm2d(aff_out_channels)
        

    def forward(self, x):
        """
        The code here is just a coarse implementation.
        The forward process can be quite slow and memory consuming, need to be optimized.
        """
        x = self.gconv1(x)
        b, c, h, w = x.size() 
        x = self.norm(x)
        x = self.relu(x)
        
        aff_x = self.gconv2(x)
        aff_x = self.norm2(aff_x)
        aff_x = self.relu(aff_x)
        x_averaged = self.avg_pool(aff_x)
        
#        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
#        theta_x = theta_x.permute(0, 2, 1)
#        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
#        f = torch.matmul(theta_x, phi_x)
#        N = f.size(-1)
#        f_div_C = f / N
        
        x_vec = x_averaged.view(b, self.groups, -1)

        theta_x = x_vec       
        phi_x = x_vec.permute(0, 2, 1) 

        aff = torch.matmul(theta_x, phi_x)
        bone = self.bone.cuda().repeat(b,1,1)
        print(bone.shape)
        aff = aff + bone
        print(aff.shape)
        print(aff)
        
        N = aff.size(-1)
        aff_div_C = aff / N

        x = x.view(b, self.groups, -1)
        z = torch.matmul(aff_div_C, x)
        
        z = z.view(b, c, h, w)

        return z


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class SemanticBlock(nn.Module):
    def __init__(self, in_channels, num_joints):
        super(SemanticBlock, self).__init__()
            
        ### 3x3 group conv: in_channels --> in_channels
        self.conv_1 = nn.Conv2d(in_channels, in_channels,padding=1, bias=False,
                           kernel_size=3, groups=num_joints)
        
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        
        self.relu = nn.ReLU(inplace=True)
            
        ### 3x3 Semantic conv: in_channels --> in_channels
        self.conv_2 = SemanticMultiGroupConv(
                in_channels, in_channels,kernel_size=3, stride=1,
                 padding=1, groups= num_joints)
        
    def forward(self, x):
        residual = x
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn2(x)
        x = residual + x
        x = self.relu(x)
        return x

class SemconvNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(SemconvNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        extend_channels = cfg.MODEL.NUM_JOINTS*4
        self.extend_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=extend_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.bn4 = nn.BatchNorm2d(extend_channels, momentum=BN_MOMENTUM)
        self.stage4_semantic_block = SemanticBlock(extend_channels, cfg.MODEL.NUM_JOINTS)
#        self.stage4_semantic_block_2 = SemanticBlock(extend_channels, cfg.MODEL.NUM_JOINTS)
        self.hrnet_predict_layer = nn.Conv2d(
            in_channels=extend_channels,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            groups = cfg.MODEL.NUM_JOINTS,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
#        extend_channels = extend_channels*2
        self.stage4_predict_layer = nn.Conv2d(
            in_channels=extend_channels,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            groups = cfg.MODEL.NUM_JOINTS,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.stage4_aspp_predict_layer = ASPP(extend_channels, cfg.MODEL.NUM_JOINTS, groups = cfg.MODEL.NUM_JOINTS)

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if not isinstance(self.transition1[i], nn.Identity):
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if not isinstance(self.transition2[i], nn.Identity):
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if not isinstance(self.transition3[i], nn.Identity):
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.extend_layer(y_list[0]) 
        x = self.bn4(x)
        x = self.relu(x)
        hrnet_predict = self.hrnet_predict_layer(x)
        
        sem_x = self.stage4_semantic_block(x)
 #       sem_2 = self.stage4_semantic_block_2(x)

        # Permutation : channels come from each branches in turn
#        sem = torch.cat ( (sem_1, sem_2), dim=1)
#        b, c, h, w = sem.size()
#        branches = 2
#        sem_x = sem.view(b, branches, c // branches, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        aspp_predict = self.stage4_aspp_predict_layer(sem_x)
        sem_predict = self.stage4_predict_layer(sem_x)
        stage4_predict = aspp_predict + sem_predict
        return hrnet_predict, stage4_predict

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = SemconvNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
