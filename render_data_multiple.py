import argparse
import subprocess

import os
import shutil
import pathlib
import os
import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='C:\\pifu\\baseData')
    parser.add_argument('-o', '--out_dir', type=str, default='C:\\pifu\\trainingData')
    parser.add_argument('-m', '--ms_rate', type=int, default=1, help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('-e', '--egl',  action='store_true', help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('-s', '--size',  type=int, default=512, help='rendering image size')
    args = parser.parse_args()

    # NOTE: GL context has to be created before any other OpenGL function loads.
    dirs = []

    for (dirpath, dirnames, filenames) in os.walk(args.input):
        dirs.extend(dirnames)
        break

    for idx, d in enumerate(dirs):
        print("------------- {0} / {1} -------------".format(idx, len(dirs)))

        arg = os.path.join(args.input, d)

        print(arg)
        out_path = r'C:\pifu\trainingData'
        subject_name = d
        view_dir =os.path.join(r'C:\Blueprint2Car\data\training', subject_name)

        if not os.path.isfile(os.path.join(view_dir, "top_Blueprint.npy")):
            print("No views for {0}, Skipping!".format(subject_name))
            continue

        copy = True
        if copy:
            basePathRender = os.path.join(out_path, 'RENDER', subject_name)

            topFrom = os.path.join(os.path.join(view_dir, "top_Blueprint.npy"))
            topTo = os.path.join(os.path.join(basePathRender, "90_90_00.npy"))
            sideFrom = os.path.join(os.path.join(view_dir, "side_Blueprint.npy"))
            sideTo = os.path.join(os.path.join(basePathRender, "90_0_00.npy"))
            frontFrom = os.path.join(os.path.join(view_dir, "front_Blueprint.npy"))
            fromTo = os.path.join(os.path.join(basePathRender, "180_0_00.npy"))
            backFrom = os.path.join(os.path.join(view_dir, "back_Blueprint.npy"))
            backTo = os.path.join(os.path.join(basePathRender, "0_0_00.npy"))

            topF = cv2.flip(np.load(topFrom), 1)
            sideF = cv2.flip(np.load(sideFrom), 1)
            np.save(topTo, topF)
            np.save(sideTo, sideF)
            # shutil.copy(topFrom, topTo)
            # shutil.copy(sideFrom, sideTo)
            shutil.copy(frontFrom, fromTo)
            shutil.copy(backFrom, backTo)

            print("Copied to {0} ".format(basePathRender))

        #os.system(r'conda activate pifu & python -m apps.render_data -i {0} &'.format(arg))



