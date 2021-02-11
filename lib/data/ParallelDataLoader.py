import os
from multiprocessing import Manager
from multiprocessing.context import Process
import src.datasetInterfaces.datasetInterfaceProcessed  as datasetInterfaceProcessed
import numpy as np
import psutil

##########################################################
#This loads sdf and normals from the preprocessing step in parallel
##########################################################

def loadData(root_dir, sub_name, loadNormals=True):
    combinedPath = os.path.join(root_dir, sub_name)
    pointsSDF = np.load(os.path.join(combinedPath, 'points.npy'))
    sdf = np.load(os.path.join(combinedPath, 'sdf.npy')) < 0

    pointsNormals = None
    normals = None

    if loadNormals:
        pointsNormals = np.load(os.path.join(combinedPath, 'points_Normals.npy'))
        normals = np.load(os.path.join(combinedPath, 'Normals.npy'))

    return pointsSDF, sdf, pointsNormals, normals


def load_chunk(root_dir, foldersLocal, pointsSDF, sdf, pointsNormals, normals, loadNormals = True):

    for i, sub_name in enumerate(foldersLocal):
        if psutil.virtual_memory().percent < 85:
            #print("Loading ... {0} / {1}".format(i, len(foldersLocal)))

            pointsSDF_l, sdf_l, pointsNormals_l, normals_l = loadData(root_dir, sub_name, loadNormals)

            pointsSDF[sub_name] = pointsSDF_l
            sdf[sub_name] = sdf_l

            if loadNormals:
                pointsNormals[sub_name] = pointsNormals_l
                normals[sub_name] = normals_l
        else:
            print("Stopping, running out of memory soon. Rest will be streamed from disc.")
            break


def loadDataParallel(root_dir, folders, loadNormals):

    num_cpus = psutil.cpu_count(logical=False)
    chunks = [folders[i::num_cpus] for i in range(num_cpus)]

    manager = Manager()
    points = manager.dict()
    sdf = manager.dict()
    pointsNormal = manager.dict()
    normals = manager.dict()

    job = [Process(target=load_chunk, args=(root_dir, chunks[i], points, sdf, pointsNormal, normals, loadNormals)) for i in range(num_cpus)]
    _ = [p.start() for p in job]
    _ = [p.join() for p in job]

    return points, sdf, pointsNormal, normals

##########################################################
#This loads depth and normal images for the rendering step
##########################################################

def load_depth_normal_views(root_dir, foldersLocal, normals, depth):
    for i, sub_name in enumerate(foldersLocal):
        if psutil.virtual_memory().percent < 85:
            n = datasetInterfaceProcessed.getViewsNormals(os.path.join(root_dir, sub_name))
            d = datasetInterfaceProcessed.getViewsDepth(os.path.join(root_dir, sub_name))

            normals[sub_name] = n
            depth[sub_name] = d
        else:
            print("Stopping, running out of memory soon. Rest will be streamed from disc.")
            break

def loadDepthNormalViewsParallel(root_dir, folders):

    num_cpus = psutil.cpu_count(logical=False)
    chunks = [folders[i::num_cpus] for i in range(num_cpus)]

    manager = Manager()
    depth = manager.dict()
    normals = manager.dict()

    job = [Process(target=load_depth_normal_views, args=(root_dir, chunks[i], normals, depth)) for i in range(num_cpus)]
    _ = [p.start() for p in job]
    _ = [p.join() for p in job]

    return depth, normals