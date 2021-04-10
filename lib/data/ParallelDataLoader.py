import os
from multiprocessing import Manager
from multiprocessing.context import Process
import src.datasetInterfaces.datasetInterfaceProcessed  as datasetInterfaceProcessed
import numpy as np
import psutil

##########################################################
#This loads sdf and normals from the preprocessing step in parallel
##########################################################

def loadData(root_dir, sub_name, opt):
    combinedPath = os.path.join(root_dir, sub_name)
    pointsSDF = np.load(os.path.join(combinedPath, 'points.npy'))
    sdf = np.load(os.path.join(combinedPath, 'sdf.npy'))

    if not opt.regression:
        sdf = sdf < 0

    pointsNormals = None
    normals = None
    edges = None

    if opt.use_normal_loss or opt.use_edge_loss:
        pointsNormals = np.load(os.path.join(combinedPath, 'points_Normals.npy'))

    if opt.use_normal_loss:
        normals = np.load(os.path.join(combinedPath, 'Normals.npy'))
    if opt.use_edge_loss:
        edges = np.load(os.path.join(combinedPath, 'Edges.npy'))
        #Calc length of each vector
        edges = np.linalg.norm(edges, axis=1, keepdims=True)
        #print("Loaded Edges Min: {0}, Max: {1}".format(np.min(edges), np.max(edges)))

    return pointsSDF, sdf, pointsNormals, normals, edges


def load_chunk(root_dir, foldersLocal, pointsSDF_dict, sdf_dict, pointsNormals_dict, normals_dict, edges_dict, opt):

    for i, sub_name in enumerate(foldersLocal):
        if psutil.virtual_memory().percent < 75:
            #print("Loading ... {0} / {1}".format(i, len(foldersLocal)))

            pointsSDF, sdf, pointsNormals, normals, edges = loadData(root_dir, sub_name, opt)

            pointsSDF_dict[sub_name] = pointsSDF
            sdf_dict[sub_name] = sdf

            if opt.use_normal_loss or opt.use_edge_loss:
                pointsNormals_dict[sub_name] = pointsNormals

            if opt.use_normal_loss:
                normals_dict[sub_name] = normals

            if opt.use_edge_loss:
                edges_dict[sub_name] = edges
        else:
            print("Stopping, running out of memory soon. Rest will be streamed from disc.")
            break


def loadDataParallel(root_dir, folders, opt):

    num_cpus = psutil.cpu_count(logical=False)
    chunks = [folders[i::num_cpus] for i in range(num_cpus)]

    manager = Manager()
    points = manager.dict()
    sdf = manager.dict()
    pointsNormal = manager.dict()
    normals = manager.dict()
    edges = manager.dict()

    job = [Process(target=load_chunk, args=(root_dir, chunks[i], points, sdf, pointsNormal, normals, edges, opt)) for i in range(num_cpus)]
    _ = [p.start() for p in job]
    _ = [p.join() for p in job]

    return points, sdf, pointsNormal, normals, edges

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