import warnings
warnings.filterwarnings("ignore")

import models
import datas
import configs

import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import datetime
from utils.config import Config
from tensorboardX import SummaryWriter
import sys

import time

import cv2

# loading configures
parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
# args = parser.parse_config()

config = Config.from_file(args.config)
MS_test = False
flip_test = False # False | True
rotation_test = False
reverse_test = False
reverse_flip = False
reverse_rotation = False


# preparing datasets
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in config.mean]
revstd = [1.0 / x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

testset = datas.AIMSequence(config.testset_root, trans, config.test_size, config.test_crop_size, config.inter_frames)
sampler = torch.utils.data.SequentialSampler(testset)
validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)


# model

model = getattr(models, config.model)(config.pwc_path).cuda()
model = nn.DataParallel(model)

tot_time = 0
tot_frames = 0

print('Everything prepared. Ready to test...')

to_img = TF.ToPILImage()



def interp_resize(input_tensor, scale_factor):
    # input tensor: [B, C, H, W]
    B, C, H, W = input_tensor.size()   
    out = F.interpolate(input_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=None)
    
    return out


def generate():
    global tot_time, tot_frames
    retImg = []
   
    
    store_path = config.store_path

    with torch.no_grad():
        for validationIndex, validationData in enumerate(validationloader, 0):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
            sample, folder, index, img_name = validationData

            # make sure store path exists
            if not os.path.exists(config.store_path + '/' + folder[1][0]):
                os.mkdir(config.store_path + '/' + folder[1][0])

            # if sample consists of four frames (ac-aware)
                       
            if len(sample) is 4:
                frame0 = sample[0]
                frame1 = sample[1]
                frame2 = sample[-2]
                frame3 = sample[-1]

                I0 = frame0.cuda()
                I3 = frame3.cuda()

                I1 = frame1.cuda()
                I2 = frame2.cuda()
                
                if config.preserve_input:
                    revtrans(I1.clone().cpu()[0]).save(store_path + '/' + folder[1][0] + '/'  + index[1][0] + '.png')
                    revtrans(I2.clone().cpu()[0]).save(store_path + '/' + folder[-2][0] + '/' +  index[-2][0] + '.png')
            # else two frames (linear)
            else:
                frame0 = None
                frame1 = sample[0]
                frame2 = sample[-1]
                frame3 = None

                I0 = None
                I3 = None
                I1 = frame1.cuda()
                I2 = frame2.cuda()
             
                if config.preserve_input:
                    revtrans(I1.clone().cpu()[0]).save(store_path + '/' + folder[0][0] + '/'  + index[0][0] + '.png')
                    revtrans(I2.clone().cpu()[0]).save(store_path + '/' + folder[1][0] + '/' +  index[1][0] + '.png')

            
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                print(t)


                # record duration time
                start_time = time.time()
                
                aug_count = 0
                # 1: normal
                It_warp, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0, I1, I2, I3, t)
                aug_count += 1
                
                if MS_test:                    
                    scale = 0.5
                    I0_down = interp_resize(I0, scale) if I0 is not None else I0
                    I1_down = interp_resize(I1, scale)
                    I2_down = interp_resize(I2, scale)
                    I3_down = interp_resize(I3, scale) if I3 is not None else I3
                    
                    It_warp_down, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0_down, I1_down, I2_down, I3_down, t)
                    It_warp_up = interp_resize(It_warp_down, 1/scale)
                    It_warp += It_warp_up
                    aug_count += 1
              
                if flip_test:
                    # 2: flip W
                    I0_fW = torch.flip(I0, (-1, )) if I0 is not None else I0
                    I1_fW = torch.flip(I1, (-1, ))
                    I2_fW = torch.flip(I2, (-1, ))
                    I3_fW = torch.flip(I3, (-1, )) if I3 is not None else I3
                    It_warp2, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0_fW, I1_fW, I2_fW, I3_fW, t)
                    It_warp += torch.flip(It_warp2, (-1, ))
                    aug_count += 1
                    
                    # 3: flip H
                    I0_fH = torch.flip(I0, (-2, )) if I0 is not None else I0
                    I1_fH = torch.flip(I1, (-2, ))
                    I2_fH = torch.flip(I2, (-2, ))
                    I3_fH = torch.flip(I3, (-2, )) if I3 is not None else I3
                    It_warp3, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0_fH, I1_fH, I2_fH, I3_fH, t)
                    It_warp += torch.flip(It_warp3, (-2, ))
                    aug_count += 1
                    
                    '''
                    # 4: flip W and H
                    I0_fWH = torch.flip(I0, (-2, -1)) if I0 is not None else I0
                    I1_fWH = torch.flip(I1, (-2, -1))
                    I2_fWH = torch.flip(I2, (-2, -1))
                    I3_fWH = torch.flip(I3, (-2, -1)) if I3 is not None else I3
                    It_warp4, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0_fWH, I1_fWH, I2_fWH, I3_fWH, t)
                    It_warp += torch.flip(It_warp4, (-2, -1))
                    aug_count += 1    
                    '''
                    
                # rotation
                if rotation_test:
                    # 5: rotate 90
                    I0_r90 = torch.rot90(I0, 1, (-1, -2)) if I0 is not None else I0
                    I1_r90 = torch.rot90(I1, 1, (-1, -2))
                    I2_r90 = torch.rot90(I2, 1, (-1, -2))
                    I3_r90 = torch.rot90(I3, 1, (-1, -2)) if I3 is not None else I3
                    It_warp5, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0_r90, I1_r90, I2_r90, I3_r90, t)
                    It_warp += torch.rot90(It_warp5, 3, (-1, -2))
                    aug_count += 1
                       
                    # 6: rotate 270
                    I0_r270 = torch.rot90(I0, 3, (-1, -2)) if I0 is not None else I0
                    I1_r270 = torch.rot90(I1, 3, (-1, -2))
                    I2_r270 = torch.rot90(I2, 3, (-1, -2))
                    I3_r270 = torch.rot90(I3, 3, (-1, -2)) if I3 is not None else I3
                    It_warp6, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0_r270, I1_r270, I2_r270, I3_r270, t)
                    It_warp += torch.rot90(It_warp6, 1, (-1, -2))
                    aug_count += 1
                    
                    
                    
                if reverse_test:
                    # 7: reverse normal
                    It_warp7, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I3, I2, I1, I0, 1-t)
                    It_warp += It_warp7
                    aug_count += 1
                        
                    if reverse_flip:
                        # 8: reverse flip W
                        It_warp8, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I3_fW, I2_fW, I1_fW, I0_fW, 1-t)
                        It_warp += torch.flip(It_warp8, (-1, ))
                        aug_count += 1                            
                        # 9: reverse flip H
                        It_warp9, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I3_fH, I2_fH, I1_fH, I0_fH, 1-t)
                        It_warp += torch.flip(It_warp9, (-2, ))
                        aug_count += 1
                        # 10: reverse flip W and H
                        It_warp10, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I3_fWH, I2_fWH, I1_fWH, I0_fWH, 1-t)
                        It_warp += torch.flip(It_warp10, (-2, -1))
                        aug_count += 1
                            
                    if reverse_rotation:
                        # 11: reverse rotate 90
                        It_warp11, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I3_r90, I2_r90, I1_r90, I0_r90, t)
                        It_warp += torch.rot90(It_warp11, 3, (-1, -2))
                        aug_count += 1
                            
                        # 12: reverse rotate 270
                        It_warp12, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I3_r270, I2_r270, I1_r270, I0_r270, t)
                        It_warp += torch.rot90(It_warp12, 1, (-1, -2))
                        aug_count += 1
                            
                            
                    
                # summary
                It_warp = It_warp/aug_count
                
                tot_time += (time.time() - start_time)
                tot_frames += 1
                

                if len(sample) is 4:
                    #print(img_name[1][0], img_name[2][0]) 
                                     
                    int_name = (int(img_name[1][0].split('/')[-1].split('.')[0]) + int(img_name[2][0].split('/')[-1].split('.')[0]))//2
                    if t==0.25:
                        int_name = int(int_name-2)
                    elif t==0.5:
                        int_name = int(int_name)
                    elif t == 0.75:
                        int_name = int(int_name+2)
                    
                    print('Input 4 frames: Quadratic; Interp name: {}'.format(int_name))
                    
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + '{:08d}'.format(int(int_name)) + '.png')
                else:
                    #print(img_name[0][0], img_name[1][0])  
                    
                    int_name = (int(img_name[0][0].split('/')[-1].split('.')[0]) + int(img_name[1][0].split('/')[-1].split('.')[0]))//2
                    if t==0.25:
                        int_name = int(int_name-2)
                    elif t==0.5:
                        int_name = int(int_name)
                    elif t == 0.75:
                        int_name = int(int_name+2)
                        
                    print('Input 2 frames: Linear; Interp name: {}'.format(int_name))
                    
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + '{:08d}'.format(int(int_name)) + '.png')
                    
def test():

    dict1 = torch.load(config.checkpoint)  
    print(dict1.keys())
    print(dict1['Detail'])
    print(dict1['epoch'])
    max_psnr = 0
    max_id = 0
    psnr_dict = {}
    
    print('MS test: ', MS_test)
    print('flip test: ', flip_test)
    print('rotation test: ', rotation_test)
    print('reverse test: ', reverse_test)
    print('reverse flip test: ', reverse_flip)
    print('reverse rotation test: ', reverse_rotation)
    #exit()
    
    
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    if not os.path.exists(config.store_path):
        os.mkdir(config.store_path)
    generate()

print(testset)
test()

print ('Avg time is {} second'.format(tot_time/tot_frames))
