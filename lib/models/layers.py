from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticMultiGroupConv(nn.Module):
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=4):
        super(SemanticMultiGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation


        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, \
                "head number can not be divided by input channels"
        assert self.out_channels % self.groups == 0, \
                "head number can not be divided by output channels"

        self.gconv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                padding, dilation, groups, bias=False)
        
        self.grid = 5*5
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
        
        N = aff.size(-1)
        aff_div_C = aff / N
        
        x = x.view(b, self.groups, -1)
        z = torch.matmul(aff_div_C, x)
        
        z = z.view(b, c, h, w)

        return z
