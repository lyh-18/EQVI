import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .LSE_acceleration import compute_acceleration
from .forward_warp_gaussian import ForwardWarp as ForwardWarp
from .UNet2 import UNet2 as UNet
from .Small_UNet import Small_UNet
from .scopeflow_models.IRR_PWC_V2 import PWCNet as ScopeFlow

import sys
from collections import OrderedDict
import torchvision
import matplotlib.pyplot as plt


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False, ).cuda()
    gridY = torch.tensor(gridY, requires_grad=False, ).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2 * (x / (W - 1) - 0.5)
    y = 2 * (y / (H - 1) - 0.5)
    # stacking X and Y
    grid = torch.stack((x, y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut


class SmallMaskNet(nn.Module):
    """docstring for SmallMaskNet"""

    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x



class AcSloMoS_scope_unet_residual_synthesis_edge_LSE(nn.Module):
    """docstring for AcSloMo_b"""

    def __init__(self, path='./network-default.pytorch'):
        super(AcSloMoS_scope_unet_residual_synthesis_edge_LSE, self).__init__()

        
        self.fwarp = ForwardWarp()
        self.refinenet = UNet(20, 8).cuda()

        # extract the contextual information
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.feat_ext = list(resnet18.children())[0].cuda()
        self.feat_ext.stride = (1, 1)
        self.feat_ext.requires_grad = False

        ### Scope Flow
        args = None
        self.flownet = ScopeFlow(args, div_flow=0.05).cuda()
        # load model
        checkpoint = 'checkpoints/scopeflow/Sintel_ft/checkpoint_best.ckpt'
        #if isinstance(self.flownet, nn.DataParallel) or isinstance(self.flownet, DistributedDataParallel):
        #    self.flownet = self.flownet.module
        load_net = torch.load(checkpoint)
        #print(self.flownet.state_dict().keys())
        #print('#########################')

        # remove unuseful keys
        load_net_clean = OrderedDict()
        for k, v in load_net['state_dict'].items():
            if k.startswith('_model.module'):
                load_net_clean[k[14:]] = v
            else:
                load_net_clean[k] = v       
        #print(load_net_clean.keys()) 
        
        self.flownet.load_state_dict(load_net_clean, strict=True)
        print('Load ScopeFlow successfully!')
        ### Scope Flow
        
        
        
        self.masknet = SmallMaskNet(38, 1).cuda()
        self.synthesisnet = Small_UNet(140, 3)
        
        
        self.acc = compute_acceleration(self.flownet)
        
        # grad
        self.get_grad = Get_gradient()

    def forward(self, I0, I1, I2, I3, t):

        # Input: I0-I3: (N, C, H, W)
        #          t: (N, 1, 1, 1) or constant*
        with torch.no_grad():
            feat1 = self.feat_ext(I1)
            feat2 = self.feat_ext(I2)
        
            F12 = self.flownet(I1, I2).float()
            F21 = self.flownet(I2, I1).float()

        
        
        
        if I0 is not None and I3 is not None:
            F1ta = self.acc(I0, I1, I2, I3, t)
            F2ta = self.acc(I3, I2, I1, I0, 1-t)
            
            F1t = F1ta
            F2t = F2ta

        else:
            with torch.no_grad():
                F12 = self.flownet(I1, I2).float()
                F21 = self.flownet(I2, I1).float()
        
            F1t = t * F12
            F2t = (1-t) * F21
        

        Ft1, norm1 = self.fwarp(F1t, F1t)
        Ft1 = -Ft1
        Ft2, norm2 = self.fwarp(F2t, F2t)
        Ft2 = -Ft2

        # Ft1 = -(1-t)*t*F12 + t*t*F21
        # Ft2 = (1-t)*(1-t)*F12 - t*(1-t)*F21

        Ft1[norm1 > 0] = Ft1[norm1 > 0] / norm1[norm1 > 0].clone()
        Ft2[norm2 > 0] = Ft2[norm2 > 0] / norm2[norm2 > 0].clone()

        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)

        output, feature = self.refinenet(torch.cat([I1, I2, I1t, I2t, F12, F21, Ft1, Ft2], dim=1))

        Ft1r = backwarp(Ft1, 10 * torch.tanh(output[:, 4:6])) + output[:, :2]
        Ft2r = backwarp(Ft2, 10 * torch.tanh(output[:, 6:8])) + output[:, 2:4]

        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)
        
        # grad
        G1 = self.get_grad(I1)
        G2 = self.get_grad(I2)
        G1tf = backwarp(G1, Ft1r)
        G2tf = backwarp(G2, Ft2r)

        feat_warp1 = backwarp(feat1, Ft1r)
        feat_warp2 = backwarp(feat2, Ft2r)

        M = torch.sigmoid(self.masknet(torch.cat([I1tf, I2tf, feature], dim=1))).repeat(1, 3, 1, 1)

        It_warp = ((1 - t) * M * I1tf + t * (1 - M) * I2tf) / ((1 - t) * M + t * (1 - M)).clone()

        residual = self.synthesisnet(torch.cat([feat_warp1, feat_warp2, G1tf, G2tf, I1tf, I2tf], dim=1))

        It_warp = It_warp + residual
        
        It_warp = torch.clamp(It_warp, 0, 1)

        return It_warp, I1t, I2t, It_warp, It_warp, F12, F21, I1tf, I2tf, M, output[:, :2], output[:, 2:4], Ft1, Ft2, Ft1r, Ft2r, F1t, F2t, norm1, norm2




