# dataloader for multi frames (acceleration), modified from superslomo

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import sys

def _make_dataset_all(dir):
    framesPath = []
    framesIndex = []
    framesFolder = []
    # Find and loop over all the clips in root `dir`.

    totindex = 0

    for folder in sorted(os.listdir(dir)):

        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue


        # Find and loop over all the frames inside the clip.
        frames = sorted(os.listdir(clipsFolderPath))

        group_len = (len(frames) - 1)

        for index in range(group_len):
            framesPath.append([])
            framesIndex.append([])
            framesFolder.append([])
            # Add path to list.
            #print(totindex)
            # if the f0 is at the begining or f1 is at the end of the sequence,
            # just push two frames into the sequence
            if index == 0 or index == (group_len - 1):
                # f0
                framesFolder[totindex].append(folder)
                framesPath[totindex].append(os.path.join(clipsFolderPath, frames[index]))
                framesIndex[totindex].append(frames[index][:-4])

                # f1
                framesFolder[totindex].append(folder)
                framesPath[totindex].append(os.path.join(clipsFolderPath, frames[index + 1]))
                framesIndex[totindex].append(frames[index + 1][:-4])

            else:
                #print(index)
                # frame -1 .... frame 2
                for ii in range (-1, 3):
                    framesFolder[totindex].append(folder)
                    framesPath[totindex].append(os.path.join(clipsFolderPath, frames[index + ii]))
                    framesIndex[totindex].append(frames[index + ii][:-4])

            totindex += 1
    # print(framesPath)
    # print(folder)
    return framesPath, framesFolder, framesIndex


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=None):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # frameFlip = 1
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = resized_img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')


class AIMSequence(data.Dataset):
    def __init__(self, root, transform=None, resizeSize=(640, 360), randomCropSize=(352, 352), inter_frames=7):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, framesFolder, framesIndex = _make_dataset_all(root)
        #print(framesPath)
        

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.dim = resizeSize
        self.randomCropSize = randomCropSize
        self.cropX0         = self.dim[0] - randomCropSize[0]
        self.cropY0         = self.dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform

        self.inter_frames   = inter_frames
        self.framesPath     = framesPath
        self.framesFolder   = framesFolder
        self.framesIndex     = framesIndex

    def __getitem__(self, index):
        sample = []
        folders = []
        indeces = []

        cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
        IFrameIndex = ((index) % 7  + 1)
            # returnIndex = IFrameIndex - 1
        frameLen = len(self.framesPath[index])
        #print(frameLen)

        randomFrameFlip = 0
        # print(frameRange)
        # print(index)
        # Loop over for all frames corresponding to the `index`.

        # if frame is at the begining or at the end
        # if frameLen is 2:
        #     sample.append(None)
        #     indeces.append(None)
        #     folders.append(None)
        # print(frameLen)
        
        img_name = []
        for frameIndex in range(frameLen):
            # Open image using pil and augment the image.
            # print(frameIndex)
            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, resizeDim=self.dim, frameFlip=randomFrameFlip)
            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            

            # print(self.framesPath[index][frameIndex])
            # print(folder)
            # print()

            # image.save(str(frameIndex) + '.jpg')
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
            indeces.append(iindex)
            folders.append(folder)
            
            img_name.append(self.framesPath[index][frameIndex])
        print(img_name)
        # if frameLen is 2:
        #     sample.append(None)
        #     indeces.append(None)
        #     folders.append(None)

        # while True:
        #     pass

        return sample, folders, indeces, img_name

    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """


        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """


        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

