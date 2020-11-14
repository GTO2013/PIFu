import argparse
import subprocess

import os
import shutil
import pathlib
import os

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

    for d in dirs:
        print(d)
        arg = os.path.join(args.input, d)
        os.system(r'conda activate pifu & python -m apps.render_data -i {0} &'.format(arg))       



