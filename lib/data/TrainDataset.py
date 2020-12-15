
from torch.utils.data import Dataset
import sys
sys.path.append(r'C:\Blueprint2Car')
from src.datasetInterfaces import datasetInterfaceUnity, datasetInterfaceProcessed
from src.utils import viewUtils
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import psutil
import math
from multiprocessing import Manager, Process

log = logging.getLogger('trimesh')
log.setLevel(40)

def load_chunk(root_dir, foldersLocal, meshs, sdf):

    for i, sub_name in enumerate(foldersLocal):
        if psutil.virtual_memory().percent < 85:
            print("Loading ... {0} / {1}".format(i, len(foldersLocal)))
            if sdf is None:
                meshs[sub_name] = trimesh.load(os.path.join(root_dir, sub_name, '%s_100k.obj' % sub_name))
            else:
                meshs[sub_name] = np.load(os.path.join(root_dir, sub_name, 'points.npy'))
                sdf[sub_name] = np.load(os.path.join(root_dir, sub_name, 'sdf.npy'))
        else:
            print("Stopping, running out of memory soon. Rest will be streamed from disc.")
            break

def load_trimesh(root_dir, folders, loadSdf = True):

    num_cpus = psutil.cpu_count(logical=False)
    chunks = [folders[i::num_cpus] for i in range(num_cpus)]

    manager = Manager()
    points = manager.dict()
    sdf = None

    if loadSdf:
        sdf = manager.dict()

    job = [Process(target=load_chunk, args=(root_dir, chunks[i], points, sdf)) for i in range(num_cpus)]
    _ = [p.start() for p in job]
    _ = [p.join() for p in job]

    return points, sdf

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'
        self.is_train = (phase == 'train')

        sub_dir = "training" if self.is_train else "test"

        self.use_normals = True

        # Path setup
        self.root = os.path.join(self.opt.dataroot, sub_dir)

        #self.RENDER = os.path.join(self.root, 'RENDER')
        #self.MASK = os.path.join(self.root, 'MASK')
        #self.PARAM = os.path.join(self.root, 'PARAM')
        #self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        #self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        #self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        #self.UV_POS = os.path.join(self.root, 'UV_POS')
        #self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.B_MIN = np.array([-0.5, -0.5, -0.55])
        self.B_MAX = np.array([0.5, 0.5, 0.55])

        self.load_size = self.opt.loadSize
        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,181,90))
        self.pitch_list = [0, 90]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            #transforms.Resize(self.load_size),
            transforms.ToTensor(),
            #transforms.Normalize(0.5, 0.5)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.loadSdf = True
        #self.mesh_dic = load_trimesh(self.OBJ)
        self.points_dic = None
        self.sdf_dic = None

        points, sdf = load_trimesh(self.root, self.subjects, loadSdf=self.loadSdf)
        self.points_dic = points
        self.sdf_dic = sdf

    def get_subjects(self):
        #all_subjects = os.listdir(self.OBJ)

        all_subjects = []
        listAll = os.listdir(self.root)

        if(self.opt.max_train_size > 0):
            listAll = listAll[:min(len(listAll), self.opt.max_train_size)]

        for subject in listAll:
            path = os.path.join(self.root, subject, 'sdf.npy')
            if os.path.exists(path):
                testDatatype = np.load(path)
                if testDatatype.dtype is np.dtype(np.bool):
                    if os.path.exists(os.path.join(self.root, subject, 'top_Normals.npy' if self.use_normals else 'top_Blueprint.npy')):
                        all_subjects.append(subject)
                    else:
                        print("%s has no .npy render!" % subject)
                else:
                    print("Type is not bool!")
            else:
                print("%s has no sdf file!" % subject)

        #var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)

        #if len(var_subjects) == 0:
            #return all_subjects

        #if self.is_train:
            #return sorted(list(set(all_subjects) - set(var_subjects)))
        #else:
            #return sorted(list(var_subjects))

        return all_subjects

    def __len__(self):
        return len(self.subjects) # * len(self.yaw_list) * len(self.pitch_list)

    def angles_to_name(self, yaw, pitch):
        if yaw == 0 and pitch == 0:
            return "back"
        elif yaw == 90 and pitch == 0:
            return "side"
        elif yaw == 180 and pitch == 0:
            return "front"
        elif yaw == 90 and pitch == 90:
            return "top"
        else:
            return '%d_%d_%02d' % (yaw, pitch, 0)


    def make_divisible(self, x, div):
        return int(((x // div) + 1) * div)

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        #pitch = self.pitch_list[pid]

        # The ids are an even distribution of num_views around view_id
        #view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    #for offset in range(num_views)]

        #if random_sample:
            #view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        if num_views == 4:
            yaw_list = [0, 90, 90, 180]
        elif num_views == 3:
            yaw_list = [0, 90, 180]
        else:
            yaw_list = [90]
        
        calib_list = []
        render_list = []
        mask_list = []
        size_list = []
        extrinsic_list = []

        sample_path = os.path.join(self.root, subject)
        if self.use_normals:
            views = datasetInterfaceProcessed.getViewsNormals(sample_path)
        else:
            views = datasetInterfaceProcessed.getViewsBlueprint(sample_path)

        views = viewUtils.resizeSquareViews(views, self.load_size)
        bb = datasetInterfaceUnity.getBoundingBox(sample_path)
        views = viewUtils.trimViewsByBB(views, bb)

        for key in views:
            div = 32
            width = views[key].shape[1]
            height = views[key].shape[0]
            newWidth = width
            newHeight = height

            diffWidth = 0
            diffHeight = 0
            if width % div != 0:
                newWidth = self.make_divisible(width, div)
                diffWidth = (newWidth - width) //2

            if height % div != 0:
                newHeight = self.make_divisible(height, div)
                diffHeight = (newHeight - height) //2

            newImg = np.zeros((newHeight, newWidth, 3 if self.use_normals else 1), np.uint8)
            newImg[diffHeight:height+diffHeight, diffWidth:width + diffWidth, :] = views[key]
            views[key] = newImg

        for idx, yaw in enumerate(yaw_list):
            pitch = 0
            
            if num_views == 4 and idx == 2:
                pitch = 90

            poseName = self.angles_to_name(yaw, pitch)
            #param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (yaw, pitch, 0))
            #render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
            #render_path = os.path.join(sample_path, '%s_%s.npy' % (self.angles_to_name(yaw, pitch),
                                                                          #"Normals" if self.use_normals else "Blueprint"))
            #mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # loading calibration data
            #param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            #ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            #scale = param.item().get('scale') * (self.opt.loadSize/512)
            # camera center world coordinate
            #center = param.item().get('center')
            # model rotation
            #R = param.item().get('R')

            ortho_ratio = 1.0
            scale = self.opt.loadSize
            center = np.array([0, 0, 0], np.float32)
            R = np.matmul(make_rotate(math.radians(pitch), 0, 0), make_rotate(0, math.radians(yaw), 0))

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            #mask = Image.open(mask_path).convert('L')
            #dataRen = np.load(render_path)
            #render = Image.fromarray(dataRen, mode='RGB' if self.use_normals else "L")

            render = Image.fromarray(views[poseName], mode='RGB' if self.use_normals else "L")
            #render = Image.open(render_path).convert('L')
            #render = Image.open(render_path).convert('RGB')

            #render = render.resize((self.load_size, self.load_size), Image.BILINEAR)
            #mask = mask.resize((self.load_size, self.load_size), Image.BILINEAR)

            if (yaw == 90 and pitch == 0) or (yaw == 90 and pitch == 90):
                render = ImageOps.mirror(render)

            if self.is_train:
                # Pad images
                #pad_size = int(0.1 * self.load_size)
                #render = ImageOps.expand(render, pad_size, fill=0)
                #mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    #mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.8, 1.2)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    #mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                #render = render.crop((x1, y1, x1 + tw, y1 + th))
                #mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            #mask = transforms.Resize(self.load_size)(mask)
            #mask = transforms.ToTensor()(mask).float()
            #mask_list.append(mask)
            size = torch.Tensor([self.opt.loadSize / render.size[0], self.opt.loadSize / render.size[1]])
            render = self.to_tensor(render)
            #render = mask.expand_as(render) * render

            #render_list.append(render)
            size_list.append(size)
            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            #'img': torch.stack(render_list, dim=0),
            'img': render_list,
            'size': torch.stack(size_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),

            #'mask': torch.stack(mask_list, dim=0)
        }

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)

        #bounds = {'b_min': self.B_MIN, 'b_max': self.B_MAX}
        bb = datasetInterfaceUnity.getBoundingBox(os.path.join(self.root, subject))
        minBB = np.array([bb['min'][0], bb['min'][1], bb['min'][2] - 0.05])
        maxBB = np.array([bb['max'][0], bb['max'][1], bb['max'][2] + 0.05])
        bounds = {'b_min': minBB - 0.5, 'b_max': maxBB - 0.5}

        if self.loadSdf:
            points = self.points_dic[subject]
            sdf = self.sdf_dic[subject]

            num = 4 * self.num_sample_inout + self.num_sample_inout // 4
            #num = self.num_sample_inout
            index = np.random.choice(sdf.shape[0], num, replace=False)
            sample_points = points[index]
            #sdf_values = sdf[index]
            inside = np.logical_not(sdf[index])

            #sdf = np.clip(sdf, -self.opt.sigma, self.opt.sigma)
            #sdf = sdf * (1/self.opt.sigma)
            #sdf = sdf * 0.5 + 0.5

            #samples = sample_points.T
            #labels = np.expand_dims(sdf, 0)
        else:
            if subject in self.mesh_dic:
                mesh = self.mesh_dic[subject]
            else:
                mesh = trimesh.load(os.path.join(self.root, subject, '%s_100k.obj' % subject))

            #bounds = {'b_min': -mesh.extents/2, 'b_max': mesh.extents/2}
            surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
            sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

            # add random points within image space
            length = bounds['b_max'] - bounds['b_min']
            random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + bounds['b_min']

            sample_points = np.concatenate([sample_points, random_points], 0)
            np.random.shuffle(sample_points)

            inside = mesh.contains(sample_points)
            del mesh

        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[:self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[:self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[:(self.num_sample_inout - nin)]

        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        #save_samples_truncted_prob('out_{0}_old.ply'.format(subject), samples.T, labels.T)
        #exit(0)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()

        return {
            'samples': samples,
            'labels': labels
        }, bounds

    def get_color_sampling(self, subject, yid, pid=0):
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

        # Segmentation mask for the uv render.
        # [H, W] bool
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # UV render. each pixel is the color of the point.
        # [H, W, 3] 0 ~ 1 float
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

        # Normal render. each pixel is the surface normal of the point.
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
            surface_points = surface_points[sample_list].T
            surface_colors = surface_colors[sample_list].T
            surface_normal = surface_normal[sample_list].T

        # Samples are around the true surface with an offset
        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal

        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)
        tmp = 0#index // len(self.subjects)
        yid = 0#tmp % len(self.yaw_list)
        pid = 0#tmp // len(self.yaw_list)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.root, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }

        if self.opt.num_sample_inout:
            sample_data, boundingBox = self.select_sampling_method(subject)
            res.update(sample_data)

        #print(boundingBox)

        res.update(boundingBox)

        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=False)
        res.update(render_data)

        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        # for p in pts:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
