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

def generate():
    global tot_time, tot_frames
    retImg = []
      
    store_path = config.store_path

    with torch.no_grad():
        for validationIndex, validationData in enumerate(validationloader):
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

            print(int(config.inter_frames))
            print(str(index * 8))
            for tt in range(int(config.inter_frames)):
                x = int(config.inter_frames)
                t = 1.0/(x+1) * (tt + 1)
                print(t)


                # record duration time
                start_time = time.time()

                It_warp, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = model(I0, I1, I2, I3, t)
                
                # It_warp = output
                
                tot_time += (time.time() - start_time)
                tot_frames += 1
                

                if len(sample) is 4:
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[1][0] + '_' + str(tt) + '.png')
                else:
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[0][0] + '_' + str(tt) + '.png')

            
                    
def test():

    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'])

    if not os.path.exists(config.store_path):
        os.mkdir(config.store_path)
    generate()

print(testset)
test()

print ('Avg time is {} second'.format(tot_time/tot_frames))
