import argparse

import shutil
import os
import cv2
import numpy as np
import psutil
from multiprocessing import Process, Manager
import trimesh
import mesh_to_sdf
import time
import tqdm

def load_chunk(args, foldersLocal, idx, progress):
    for i, model_id in enumerate(foldersLocal):
        progress[idx] = ((i+1) / len(foldersLocal)) * 100

        #createDirs(model_id, args)
        #copyRenders(model_id, args)

        job = Process(target=calculateSDF, args=(model_id, args))
        job.start()
        job.join()

        #calculateSDF(model_id, args)

def updateProgress(args, progress):
    bars = [tqdm.tqdm(total=100) for i in range(len(progress))]

    completion = 0
    while completion < 99:
        completion = sum(progress) // len(progress)

        for idx, b in enumerate(bars):
            b.update(progress[idx] - b.n)

        time.sleep(1)

    for b in bars:
        b.close()

def centerAndNormalizeMesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    extends = mesh.extents
    vmin = mesh.vertices.min(0)
    vmax = mesh.vertices.max(0)
    center = (vmin+vmax)*0.5

    m = trimesh.transformations.translation_matrix(-center)
    meshPre = mesh.apply_transform(m)
    m = trimesh.transformations.scale_matrix(1/np.max(extends))
    mesh = meshPre.apply_transform(m)

    return mesh

def calculateSDF(model_id, args):
    mesh_file = os.path.join(args.input, model_id, model_id + "_100k.obj")
    baseNewPath = os.path.join(args.out_dir, 'GEO', 'OBJ', model_id)
    newPath = os.path.join(baseNewPath, model_id + "_100k.obj")

    meshNew = trimesh.load_mesh(mesh_file)
    meshNew = centerAndNormalizeMesh(meshNew)
    meshNew.export(newPath)
    points, sdf = mesh_to_sdf.sample_sdf_near_surface(meshNew,  number_of_points=100000)
    outPathPoints = os.path.join(os.path.join(baseNewPath, model_id + "_points"))
    outPathSDF = os.path.join(os.path.join(baseNewPath, model_id + "_sdf"))

    np.save(outPathPoints, points)
    np.save(outPathSDF, sdf > 0)
    
    del meshNew
    #colors = np.zeros(points.shape)
    #colors[sdf > 0,2] = 1
    #colors[sdf < 0,0] = 1
    #cloud = pyrender.Mesh.from_points(points, colors=colors)
    #scene = pyrender.Scene()
    #scene.add(cloud)
    #viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size = 2)
    #exit(0)

def createDirs(model_id, args):
    os.makedirs(os.path.join(args.out_dir, 'GEO', 'OBJ', model_id), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'PARAM', model_id), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'RENDER', model_id), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'MASK', model_id), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'UV_RENDER', model_id), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'UV_MASK', model_id), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'UV_POS', model_id), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'UV_NORMAL', model_id), exist_ok=True)

def copyRenders(model_id, args):
    basePathRender = os.path.join(args.out_dir, 'RENDER', model_id)
    view_dir = os.path.join(args.render_dir, model_id)

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

def preprocess_parallel(args, folders, num_cpus=None):
    if num_cpus is None:
        num_cpus = psutil.cpu_count(logical=False)

    chunks = Manager().list([folders[i::num_cpus] for i in range(num_cpus)])
    progress = Manager().list([0]*num_cpus)

    job = [Process(target=load_chunk, args=(args, chunks[i], i, progress)) for i in range(num_cpus)]
    job.append(Process(target=updateProgress, args=(args, progress)))

    _ = [p.start() for p in job]
    _ = [p.join() for p in job]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=r'C:\pifu\baseData')
    parser.add_argument('-o', '--out_dir', type=str, default=r'C:\pifu\trainingData')
    parser.add_argument('-r', '--render_dir', type=str, default=r'C:\Blueprint2Car\data\training')
    parser.add_argument('-m', '--ms_rate', type=int, default=1, help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('-e', '--egl',  action='store_true', help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('-s', '--size',  type=int, default=512, help='rendering image size')
    args = parser.parse_args()

    dirsStart = []

    #Get all folders
    dirsStart = next(os.walk(args.input))[1]

    #Filter them further
    dirs = []
    for dir in dirsStart:
        view_dir = os.path.join(args.render_dir, dir)

        if os.path.isfile(os.path.join(view_dir, "top_Blueprint.npy")):
            dirs.append(dir)
        else:
            print("No views for {0}, Skipping!".format(dir))

    #Do the stuff we can do in parallel
    #MeshToSDF already seems to run parallel though
    preprocess_parallel(args, dirs, num_cpus=2)

    #To the stuff we shouldnt  do in parallel
    for idx, d in enumerate(dirs):
        print("------------- {0} / {1} -------------".format(idx, len(dirs)))

        arg = os.path.join(args.input, d)
        #os.system(r'conda activate pifu & python -m apps.render_data -i {0} &'.format(arg))



