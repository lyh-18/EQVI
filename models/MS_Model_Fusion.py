import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


from .AcSloMoS_scope_unet_residual_synthesis_edge_LSE import AcSloMoS_scope_unet_residual_synthesis_edge_LSE


def interp_resize(input_tensor, scale_factor):
    # input tensor: [B, C, H, W]
    B, C, H, W = input_tensor.size()   
    out = F.interpolate(input_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=None)
    
    return out

        
class FuisonNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=1):
        super(FuisonNet, self).__init__()
        print('Using Fusion Map!')
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=out_ch, kernel_size=3, padding=1)
        self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x1, x2):
        x_in = torch.cat([x1, x2], dim=1)
        Mask = self.LeakyReLU(self.conv1(x_in))
        Mask = self.LeakyReLU(self.conv2(Mask))
        Mask = self.Sigmoid(self.conv3(Mask))
        
        out = Mask*x1 + (1-Mask)*x2
        return out, Mask


class MS_Model_Fusion(nn.Module):
    def __init__(self, path):
        super(MS_Model_Fusion, self).__init__()
        
        
        self.QVI_model = AcSloMoS_scope_unet_residual_synthesis_edge_LSE().cuda()
        '''
        load_net = torch.load('checkpoints/GG01_qvi_vtsr_scope_lap_ft_synthesis_edge_LSE_lap_L1/AcSloMo28.ckpt')
        # remove unuseful keys
        load_net_clean = OrderedDict()
        for k, v in load_net['model_state_dict'].items():
            if k.startswith('module'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v       
        #print(load_net_clean.keys()) 
        
        
        
        self.QVI_model.load_state_dict(load_net_clean, strict=True)
        '''
        
        self.FusionNet = FuisonNet(in_ch=6, out_ch=1)
        
        self.MS_scale = 0.5

    def forward(self, I0, I1, I2, I3, t):
        I0_down = interp_resize(I0, self.MS_scale) if I0 is not None else I0
        I1_down = interp_resize(I1, self.MS_scale)
        I2_down = interp_resize(I2, self.MS_scale)
        I3_down = interp_resize(I3, self.MS_scale) if I3 is not None else I3
        
        with torch.no_grad():
            It_warp, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = self.QVI_model(I0, I1, I2, I3, t)
            It_warp_down, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = self.QVI_model(I0_down, I1_down, I2_down, I3_down, t)
            
        It_warp_up = interp_resize(It_warp_down, 1/self.MS_scale)
        
        out, fuse_weight = self.FusionNet(It_warp, It_warp_up)

        return out, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, fuse_weight, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _