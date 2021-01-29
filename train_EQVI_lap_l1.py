import warnings
warnings.filterwarnings("ignore")

import models
import datas
import configs

# import configs.c1
import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time
import os
from math import log10
import numpy as np
import datetime
from config import Config
from tensorboardX import SummaryWriter
import sys

import cv2
# import torchvision.utils.transforms as TF

# prepare perceptual loss
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22]).cuda()
vgg16_conv_4_3 = nn.DataParallel(vgg16_conv_4_3.cuda())

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

# loss function
def lossfn(outputs, I1, I2, IT):
    It_warp, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = outputs
    
    recnLoss = F.l1_loss(It_warp, IT)

    LapLoss_module = LapLoss()
    laplacian_loss = LapLoss_module(It_warp, IT)

    loss = 5 * laplacian_loss + 10 * recnLoss

    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



# loading configures
parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
# args = parser.parse_config()

config = Config.from_file(args.config)

# preparing datasets
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in config.mean]
revstd = [1.0/x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

trainset = getattr(datas, config.trainset)(config.trainset_root, trans, config.train_size, config.train_crop_size)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=32)

validationset = getattr(datas, config.validationset)(config.validationset_root, trans, config.validation_size, config.validation_crop_size)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=8)

print(validationset)


# model
model = getattr(models, config.model)(config.pwc_path).cuda()
model = nn.DataParallel(model)

# optimizer
optim_params = []
for k, v in model.module.named_parameters():
    if v.requires_grad:
        optim_params.append(v)
    else:
        print('Params [{:s}] will not optimize.'.format(k))

#params = list(model.module.refinenet.parameters()) + list(model.module.masknet.parameters())
optimizer = optim.Adam(optim_params, lr=config.init_learning_rate)

# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
recorder = SummaryWriter(config.record_dir)

print('Everything prepared. Ready to train...')
print('We print training logs after each epoch, so it dose take a while to show the logs.')
print('We use 4 GTX 2080Ti GPUs to train the model. About 3600s for one epoch. The training procedure lasts about 3-5 days.')

to_img = TF.ToPILImage()

def validate():
    retImg = []
    # For details see training.
    # slomo = slomo.eval()
    psnr = 0
    psnrs = [0 , 0, 0]
    tloss = 0
    tlosses = [0, 0, 0]
    flag = True
    retImg = []

    with torch.no_grad():

        for validationIndex, validationData in enumerate(validationloader, 0):
            # if validationIndex > 10:
            #     break

            # frame0, frame1, frameT, frame2, frame3 = validationData
            frame0, frame1, frameT1, frameT2, frameT3, frame2, frame3 = validationData

            ITs = [frameT1.cuda(), frameT2.cuda(), frameT3.cuda()]

            I0 = frame0.cuda()
            I1 = frame1.cuda()
            I2 = frame2.cuda()
            I3 = frame3.cuda()

            It_warps = []
            Ms = []

            for tt in range(3):
                IT = ITs[tt]

                outputs = model(I0, I1, I2, I3, tt/4.0 + 0.25)
                It_warp, I1t, I2t, I1_warp, I2_warp, F12, F21, I1tf, I2tf, M, dFt1, dFt2, Ft1, Ft2, Ft1r, Ft2r, _, _, _, _ = outputs


                It_warps.append(It_warp)
                Ms.append(M)


                loss = lossfn(outputs, I1, I2, IT)
                tlosses[tt] += loss.item()

            #psnr
                MSE_val = F.mse_loss(It_warp, IT)
                psnrs[tt] += (10 * log10(1 / MSE_val.item()))
               

            img_grid = []
            img_grid.append(revNormalize(frame1[0]))
            for tt in range(3):
                img_grid.append(Ms[tt].cpu()[0])
                img_grid.append(revNormalize(It_warps[tt].cpu()[0]))
            img_grid.append(revNormalize(frame2[0]))

            retImg.append(torchvision.utils.make_grid(img_grid, nrow=10, padding=10))

        for tt in range(3):
            psnrs[tt] /= len(validationloader)
            tlosses[tt] /= len(validationloader)


    # slomo = slomo.train()
    return psnrs, tlosses, retImg

def train():

    if config.train_continue:        
        dict1 = torch.load(config.checkpoint)
        model.load_state_dict(dict1['model_state_dict'])
        print('Continue Training:', config.checkpoint)
    else:
        dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    start = time.time()
    cLoss   = dict1['loss']
    valLoss = dict1['valLoss']
    valPSNR = dict1['valPSNR']
    checkpoint_counter = 0


    for epoch in range(dict1['epoch'] + 1, config.epochs):

        print("Epoch: ", epoch)

        # Append and reset
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iLoss = 0

        # Increment scheduler count
        scheduler.step()

        trainFrameIndex = 3
        for trainIndex, (trainData, t) in enumerate(trainloader, 0):
            # if trainIndex >= 200:
            #     break
            # print("Training iteration [{}/{}]".format(trainIndex, len(trainloader)))
            # sys.stdout.flush()
            ## Getting the input and the target from the training set
            frame0, frame1, frameT, frame2, frame3 = trainData


            I0 = frame0.cuda()
            I1 = frame1.cuda()
            I2 = frame2.cuda()
            I3 = frame3.cuda()
            IT = frameT.cuda()
            t = t.view(t.size(0,), 1, 1, 1).float().cuda()

            optimizer.zero_grad()
            outputs = model(I0, I1, I2, I3, t)
            loss = lossfn(outputs, I1, I2, IT)
            loss.backward()
            optimizer.step()

            iLoss += loss.item()

        if epoch % 2 == 0:
            end = time.time()

            psnrs, vLosses, valImgs = validate()

            psnr = np.mean(psnrs)
            vLoss = np.mean(vLosses)

            valPSNR[epoch].append(np.mean(psnrs))
            valLoss[epoch].append(np.mean(vLosses))

            # Tensorboard
            itr = trainIndex + epoch * (len(trainloader))

            recorder.add_scalars('Loss', {'trainLoss': iLoss / len(trainloader), 'validationLoss': vLoss}, itr)
            recorder.add_scalar('PSNR', psnr, itr)

            vtdict = {}
            psnrdict = {}
            for tt in range(3):
                vtdict['validationLoss' + str(tt + 1)] = vLosses[tt]
                psnrdict['PSNR' + str(tt + 1)] = psnrs[tt]

            recorder.add_scalars('Losst', vtdict, itr)
            recorder.add_scalars('PSNRt', psnrdict, itr)

            # for vi, valImg in enumerate(valImgs):
            #    recorder.add_image('Validation' + str(vi), valImg , itr)

            endVal = time.time()

            print(
                " Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (
                iLoss / config.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr, endVal - end,
                get_lr(optimizer)))
            sys.stdout.flush()

            cLoss[epoch].append(iLoss / len(trainloader))
            iLoss = 0
            start = time.time()

        # Create checkpoint after every `config.checkpoint_epoch` epochs
        if (epoch >config.min_save_epoch):
            dict1 = {
                    'Detail':"Acceleration Aware Frame Interpolation.",
                    'epoch':epoch,
                    'timestamp':datetime.datetime.now(),
                    'trainBatchSz':config.train_batch_size,
                    'validationBatchSz':1,
                    'learningRate':get_lr(optimizer),
                    'loss':cLoss,
                    'valLoss':valLoss,
                    'valPSNR':valPSNR,
                    'model_state_dict': model.state_dict(),
                    }
            torch.save(dict1, config.checkpoint_dir + "/AcSloMo" + str(epoch) + ".ckpt")
            checkpoint_counter += 1


train()
