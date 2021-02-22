from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid, batch_eval_normal
from skimage import measure
from .dual_contouring.dual_contour_3d import dual_contour_3d
from .dual_contouring.utils_3d import V3

def reconstructionDualContouring(net, images, cuda, calib_tensor, img_sizes, resolution, b_min, b_max):
    print("Dual Contouring...")

    #Manually only move the image data to GPU, keep the rest CPU
    features = net.filter(images)

    def func(x,y,z):
        points = np.expand_dims(np.array([x/resolution,y/resolution,z/resolution], dtype=np.float32), axis=0)
        points = np.expand_dims(points, axis=2)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).float().to(device=cuda)
        pred = net.query(features, samples, calib_tensor, img_sizes)[0][0]
        return pred.detach().cpu().numpy()

    def normal_func(x,y,z):
        points = np.expand_dims(np.array([x/resolution,y/resolution,z/resolution], dtype=np.float32), axis=0)
        points = np.expand_dims(points, axis=2)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).float().to(device=cuda)
        normal = net.calc_normal(features, samples, calib_tensor, img_sizes)[0][0]
        normal = normal.detach().cpu().numpy()
        return V3(normal[0], normal[1], normal[2])

    return dual_contour_3d(func, normal_func,
                           xmin=int(b_min[0]*resolution), xmax=int(b_max[0]*resolution),
                           ymin=int(b_min[1]*resolution), ymax=int(b_max[1]*resolution),
                           zmin=int(b_min[2]*resolution), zmax=int(b_max[2]*resolution))

def reconstruction(net, images, cuda, calib_tensor, img_sizes,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, predict_vertex_normals = True, transform=None, dual_contouring=True):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''

    if dual_contouring:
        return NotImplementedError()
        #return reconstructionDualContouring(net, images, cuda, calib_tensor, img_sizes, resolution, b_min, b_max)
    else:
        features = net.filter(images)
        print("Marching cubes...")

        # First we create a grid by resolution
        # and transforming matrix for grid coordinates to real world xyz
        coords, mat = create_grid(resolution, resolution, resolution,
                                  b_min, b_max, transform=transform)

        # Then we define the lambda function for cell evaluation
        def eval_func(features, points):
            points = np.expand_dims(points, axis=0)
            points = np.repeat(points, net.num_views, axis=0)
            samples = torch.from_numpy(points).to(device=cuda).float()
            pred = net.query(features, samples, calib_tensor, img_sizes)[0][0]
            return pred.detach().cpu().numpy()

        # Then we define the lambda normal function for cell evaluation
        def eval_func_normals(features, points):
            points = np.expand_dims(points, axis=0)
            points = np.repeat(points, net.num_views, axis=0)
            samples = torch.from_numpy(points).to(device=cuda).float()
            pred = net.calc_normal(features, samples, calib_tensor, img_sizes)[0][0:3]
            return pred.detach().cpu().numpy()

        # Then we evaluate the grid
        if use_octree:
            sdf = eval_grid_octree(features, coords, eval_func, num_samples=num_samples)
        else:
            sdf = eval_grid(features, coords, eval_func, num_samples=num_samples)

        # Finally we do marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
            # transform verts into world coordinate system
            verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]

            if predict_vertex_normals:
                normals = batch_eval_normal(features, verts, eval_func_normals, num_samples=num_samples)
            else:
                normals = None

            verts = verts.T

            if normals is not None:
                normals = normals.T

            return verts, faces, normals, values
        except:
            import traceback
            traceback.print_exc()
            print('error cannot marching cubes')
            return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
