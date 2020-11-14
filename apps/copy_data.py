import argparse
import subprocess

import os
import shutil
import pathlib
import os
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default=r'C:\\pifu\\trainingData\\RENDER')
    parser.add_argument('-c', '--copy_dir', type=str, default=r'D:\Documents\Unity Projects\Playground\Dataset\Views')
    parser.add_argument('-m', '--ms_rate', type=int, default=1, help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('-e', '--egl',  action='store_true', help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('-s', '--size',  type=int, default=512, help='rendering image size')
    args = parser.parse_args()

    # NOTE: GL context has to be created before any other OpenGL function loads.
    dirs = []

    res = args.size   
    for (dirpath, dirnames, filenames) in os.walk(args.output):
        dirs.extend(dirnames)
        break

    for d in dirs[6:]:
        fromPath = os.path.join(args.copy_dir,d)
        outPath = os.path.join(args.output, d)
               
        p = os.path.join(fromPath, "top_Blueprint.png")
        t = cv2.imread(p)
        
        if t is not None:        
            imgTop = cv2.resize(t,(res, res))
            imgTop = cv2.flip(imgTop, 1)

            imgSide = cv2.resize(cv2.imread(os.path.join(fromPath, "side_Blueprint.png")),(res, res))
            imgSide = cv2.flip(imgSide, 1)
            
            imgFront = cv2.resize(cv2.imread(os.path.join(fromPath, "front_Blueprint.png")),(res, res))
            imgBack = cv2.resize(cv2.imread(os.path.join(fromPath, "back_Blueprint.png")),(res, res))

            cv2.imwrite(os.path.join(outPath, "90_90_00.jpg"), imgTop)
            cv2.imwrite(os.path.join(outPath, "90_0_00.jpg"), imgSide)
            cv2.imwrite(os.path.join(outPath, "180_0_00.jpg"), imgFront)
            cv2.imwrite(os.path.join(outPath, "0_0_00.jpg"), imgBack)
        else:
            print("Error reading {0}".format(d))

        
        
        
        
   



