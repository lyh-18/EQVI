# dataloader for multi frames (acceleration), modified from superslomo

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import sys

def _make_dataset(dir, inter_frames=3):

    # inter_frames = 3
    framesPath = []
    framesIndex = []
    framesFolder = []
    # Find and loop over all the clips in root `dir`.

    totindex = 0

    for folder in os.listdir(dir):
        
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue


        # Find and loop over all the frames inside the clip.
        frames = sorted(os.listdir(clipsFolderPath))

        group_len = (len(frames) - 1) // (inter_frames + 1) - 2
        for index in range(group_len):
            framesPath.append([])
            framesIndex.append([])
            framesFolder.append([])
            # Add path to list.

            # frame 0
            framesFolder[totindex].append(folder)
            framesPath[totindex].append(os.path.join(clipsFolderPath, frames[index * (inter_frames + 1)]))
            framesIndex[totindex].append(frames[index * (inter_frames + 1)][:-4])

            # frame 1 .... frame 2
            for ii in range (0, inter_frames + 2):
                framesFolder[totindex].append(folder)
                framesPath[totindex].append(os.path.join(clipsFolderPath, frames[(index + 1) * (inter_frames + 1) + ii]))
                framesIndex[totindex].append(frames[(index + 1) * (inter_frames + 1) + ii][:-4])

            # frame 3
            framesFolder[totindex].append(folder)
            framesPath[totindex].append(os.path.join(clipsFolderPath, frames[(index + 3)* (inter_frames + 1)]))
            framesIndex[totindex].append(frames[(index + 3) * (inter_frames + 1)][:-4])

            totindex += 1
    # print(folder)
    return framesPath, framesFolder, framesIndex


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = resized_img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')

    
class VTSR(data.Dataset):
    def __init__(self, root, transform=None, resizeSize=(640, 360), randomCropSize=(352, 352), inter_frames=3):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, framesFolder, framesIndex = _make_dataset(root, inter_frames)

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
        
        self.dim = resizeSize
        self.randomCropSize = randomCropSize
        self.cropX0 = resizeSize[0] - randomCropSize[0]
        self.cropY0 = resizeSize[1] - randomCropSize[1]
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

        cropX0 = random.randint(0, self.cropX0)
        cropY0 = random.randint(0, self.cropY0)

        cropArea = (cropX0, cropY0, cropX0 + self.randomCropSize[0], cropY0 + self.randomCropSize[1])

            # returnIndex = IFrameIndex - 1
        frameRange = [0, 1, 3, 5, 6]
        t = 0.5

        frameRange[2] = (random.randint(2, 4))        
        t = (frameRange[2] - 1) * 1.0 / 4.0

        randomFrameFlip = random.randint(0, 1)
        if randomFrameFlip:
            frameRange.reverse()
            t = 1 - t

        # print(frameRange)
        # print(index)
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            # print(frameIndex)
            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, resizeDim=self.dim, frameFlip=randomFrameFlip)
            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            
            # image.save(str(frameIndex) + '.jpg')
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
            indeces.append(iindex)
            folders.append(folder)


        # while True:
        #     pass

        return sample, t

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
    
