import argparse
import subprocess

import os
import shutil
import pathlib
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=r'C:\pifu\trainingData\GEO\OBJ')
    parser.add_argument('-m', '--res', type=int, default=90000, help='Resolution')

    args = parser.parse_args()

    # NOTE: GL context has to be created before any other OpenGL function loads.
    dirs = []

    for (dirpath, dirnames, filenames) in os.walk(args.input):
        dirs.extend(dirnames)
        break

    for idx, d in enumerate(dirs):
        arg = os.path.join(args.input, d, d + "_100k.obj")
        os.system(r'conda activate pifu & apps\manifold {0} {1} {2} &'.format(arg, arg, args.res))

        print("{0} / {1}".format(idx+1, len(dirs)))


