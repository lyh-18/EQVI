import cv2
import numpy as np
import os

from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    
    # for sorting the file names properly
    # TODO: can be changed depending on your data
    files.sort()
    #print(files)
    #exit()
    print('#####################')
    print('Input path: {}'.format(pathIn))
    print('Total frames: {}'.format(len(files)))
    print('FPS: {}'.format(fps))
    print('Output path: {}'.format(pathOut))
    print('Processing......')

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    print('Finished!')
    print('#####################')

def main():
    # TODO: change your input & output path here
    pathIn= '/home/yhliu/EQVI_release/outputs/old_films_interp3/1/'
    pathOut = '/home/yhliu/EQVI_release/outputs/old_films_interp3/1_inter3.mp4'
    
    fps = 25
    convert_frames_to_video(pathIn, pathOut, fps)

if __name__=="__main__":
    main()