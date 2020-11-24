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
                 padding=1, dilation=0, groups=16):
        super(SemanticMultiGroupConv, self).__init__()

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
        self.grid = 25
        aff_out_channels = self.grid * out_channels

        
        self.gconv1 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(groups):
            self.gconv1.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                    padding, dilation, groups, bias=False)))
            self.norm.append(nn.Sequential(nn.BatchNorm2d(in_channels)))
            self.gconv2.append(nn.Sequential(nn.Conv2d(out_channels, aff_out_channels, 1, stride, 
                    padding, dilation, groups, bias=False)))
            self.norm2.append(nn.Sequential(nn.BatchNorm2d(aff_out_channels)))
        

    def forward(self, x):
        """
        The code here is just a coarse implementation.
        The forward process can be quite slow and memory consuming, need to be optimized.
        """

        result_x = None
        for i in range(self.groups): 
            each_x = self.gconv1[i](x)
            b, c, h, w = each_x.size() 

            each_x = self.norm[i](each_x)
            each_x = self.relu(each_x)
        
            aff_x = self.gconv2[i](each_x)
            aff_x = self.norm2[i](aff_x)
            aff_x = self.relu(aff_x)
            x_averaged = self.avg_pool(aff_x)
#            print(x_averaged.shape)
        
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
#            print(aff.shape)
            aff = aff[:,i]
            print(aff.shape)
            N = aff.size(-1)
            aff_div_C = aff / N
        
            each_x= each_x.view(b, self.groups, -1)
            print(aff_div_C)
            print(each_x.shape)
            z = torch.matmul(aff_div_C, each_x)
            print(z.shape)
            z = z.view(b, -1, h, w)
            print(z.shape)
            if result_x == None :
                result_x = z
            else:
                result_x = torch.cat ( (result_x, z), dim=1)
        print(result_x.shape)
        return result_x
