import os
import trimesh
import numpy as np
import math
from scipy.special import sph_harm
import argparse
from tqdm import tqdm
import pathlib
import re

def computePRT(mesh_path, n, order):
    mesh = trimesh.load(mesh_path, process=False, skip_materials=True)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(True)
      
    origins = mesh.vertices
    
    return np.ones((origins.shape[0], 9)), mesh,  mesh.faces

def testPRT(dir_path, d, output_path):
    obj_path = os.path.join(dir_path, d, 'models', 'model_normalized.obj')
    
    PRT, mesh, F = computePRT(obj_path, 1, 2)

    combPath = os.path.join(output_path, d)
    
    os.makedirs(os.path.join(combPath, 'bounce'), exist_ok=True)   
    mesh.export(os.path.join(combPath, d + "_100k.obj"))
    np.savetxt(os.path.join(combPath, 'bounce', 'bounce0.txt'), PRT, fmt='%.8f')
    np.save(os.path.join(combPath, 'bounce', 'face.npy'), F)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='D:\\Documents\\Unity Projects\\Playground\\Dataset\\Models')
    parser.add_argument('-o', '--output', type=str, default='C:\\pifu\\baseData')
    parser.add_argument('-n', '--n_sample', type=int, default=40, help='squared root of number of sampling. the higher, the more accurate, but slower')
    args = parser.parse_args()

    dirs = []

    for (dirpath, dirnames, filenames) in os.walk(args.input):
        dirs.extend(dirnames)
        break

    for d in dirs:
        print(d)
        try:
            testPRT(os.path.join(args.input), d, args.output)
        except:
            pass
