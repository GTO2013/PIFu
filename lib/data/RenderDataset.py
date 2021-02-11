from torch.utils.data import Dataset
from .ParallelDataLoader import loadData, loadDepthNormalViewsParallel

import torch.nn.functional as F
from src.datasetInterfaces import datasetInterfaceUnity, datasetInterfaceProcessed
from src.utils import viewUtils
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch
import cv2
import matplotlib.pyplot as plt
import logging
import math
from ..train_util import make_rotate, save_samples_rgb, save_samples_truncted_prob

log = logging.getLogger('trimesh')
log.setLevel(40)

class RenderDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'
        self.is_train = (phase == 'train')
        self.is_eval = (phase == 'eval')

        sub_dir = "training" if self.is_train else "test"

        self.use_normals_input = opt.use_normal_input
        self.use_normals = self.opt.use_normal_loss

        # Path setup
        self.root = os.path.join(self.opt.dataroot, sub_dir)
        self.B_MIN = np.array([-0.55, -0.55, -0.55])
        self.B_MAX = np.array([0.55, 0.55, 0.55])

        self.load_size = self.opt.loadSize
        self.num_views = self.opt.num_views

        self.yaw_list = list(range(0,181,90))
        self.pitch_list = [0, 90]
        self.subjects = self.get_subjects() if not self.is_eval else None

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            #transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.depth_dic, self.normals_dic = loadDepthNormalViewsParallel(self.root, self.subjects)
        self.bounding_box = None
        self.views = None

    def get_subjects(self):
        all_subjects = []
        listAll = os.listdir(self.root)

        if(self.opt.max_train_size > 0):
            listAll = listAll[:min(len(listAll), self.opt.max_train_size)]

        for subject in listAll:
            try:
                path = os.path.join(self.root, subject, 'sdf.npy')
                if os.path.exists(path):
                    testDatatype = np.load(path,allow_pickle=True)
                    if testDatatype.dtype is np.dtype(np.float32):
                        if os.path.exists(os.path.join(self.root, subject, 'top_Normals.npy' if self.use_normals_input else 'top_Blueprint.npy')):
                            all_subjects.append(subject)
                        else:
                            print("%s has no .npy render!" % subject)
                    else:
                        print("Type is not float!")
                else:
                    print("%s has no sdf file!" % subject)
            except Exception as e:
                print(e)

        #Make sure subject count is divisible by batchsize for multi gpu
        if len(all_subjects) % self.opt.batch_size != 0:
            all_subjects = all_subjects[len(all_subjects) % self.opt.batch_size:]

        return all_subjects

    def __len__(self):
        return len(self.subjects)

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

    def set_views_from_path(self, subject):
        sample_path = os.path.join(self.root, subject)
        bb = datasetInterfaceUnity.getBoundingBox(sample_path)
        views = datasetInterfaceProcessed.getViewsBlueprint(sample_path)
        views = viewUtils.augmentViews(views, self.opt.random_scale, self.opt.loadSize, bb)
        #views = viewUtils.colorcodeViews(views)

        self.views = views
        self.bounding_box = bb

    def get_render(self, subject, num_views):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
        '''

        if num_views == 4:
            yaw_list = [0, 90, 90, 180]
        elif num_views == 3:
            yaw_list = [0, 90, 180]
        else:
            yaw_list = [90]

        calib_list = []
        normal_list = []
        depth_list = []
        render_list = []
        extrinsic_list = []

        #Loads
        self.set_views_from_path(subject)

        depth_views = self.depth_dic[subject]
        depth_views = viewUtils.resizeViews(depth_views, self.opt.loadSize)
        depth_views = viewUtils.trimViewsByBB(depth_views, self.bounding_box)

        normal_views = self.normals_dic[subject]
        normal_views = viewUtils.resizeViews(normal_views, self.opt.loadSize)
        normal_views = viewUtils.trimViewsByBB(normal_views, self.bounding_box)

        for idx, yaw in enumerate(yaw_list):
            pitch = 0
            if num_views == 4 and idx == 2:
                pitch = 90

            poseName = self.angles_to_name(yaw, pitch)

            scale = self.opt.loadSize
            center = np.array([0, 0, 0], np.float32)
            R = np.matmul(make_rotate(math.radians(pitch), 0, 0), make_rotate(0, math.radians(yaw), 0))

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale
            scale_intrinsic[1, 1] = -scale
            scale_intrinsic[2, 2] = scale
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            render = Image.fromarray(self.views[poseName], mode='L')
            normal = Image.fromarray(normal_views[poseName], mode='RGB')
            normal_img_tensor = transforms.ToTensor()(normal)
            normal_img_tensor = F.normalize(normal_img_tensor, dim=0, eps=1e-8)
            #normal.show()

            if (yaw == 90 and pitch == 0) or (yaw == 90 and pitch == 90):
                render = ImageOps.mirror(render)

            if self.is_train:
                dx = 0
                dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                render = self.aug_trans(render)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic))
            extrinsic = torch.Tensor(extrinsic)

            if not self.opt.use_gan_input:
                render = self.to_tensor(render)



            normal_list.append(normal_img_tensor)
            depth_list.append(np.expand_dims(depth_views[poseName],0))
            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            'img': render_list,
            'img_nml': normal_list,
            'img_depth': depth_list,
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0)
        }

    def select_sampling_method(self, subject):
        bb = datasetInterfaceUnity.getBoundingBox(os.path.join(self.root, subject))
        minBB = np.array([bb['min'][0] - 0.05, bb['min'][1] - 0.05, bb['min'][2] - 0.05]) - 0.5
        maxBB = np.array([bb['max'][0] + 0.05, bb['max'][1] + 0.05, bb['max'][2] + 0.05]) - 0.5

        #depth_views = self.depth_dic[subject]
        #depth_views = voxelUtils.undoScalingDepthVoxel(depth_views, bb)

        #pc_front = torch.Tensor(pointcloudUtils.generatePointcloud({'front': depth_views['front']}, secondSide=False, epsilonTopSide=1.2, epsilonFrontBack=1.2).vertices.T)
        #pc_back = torch.Tensor(pointcloudUtils.generatePointcloud({'back': depth_views['back']}, secondSide=False, epsilonTopSide=1.2, epsilonFrontBack=1.2).vertices.T)
        #pc_side = torch.Tensor(pointcloudUtils.generatePointcloud({'side': depth_views['side']}, secondSide=False, epsilonTopSide=1.2, epsilonFrontBack=1.2).vertices.T)
        #pc_top = torch.Tensor(pointcloudUtils.generatePointcloud({'top': depth_views['top']}, secondSide=False, epsilonTopSide=1.2, epsilonFrontBack=1.2).vertices.T)

        return {
            'samples': None,
            'samples_render': None,
            'labels': None,
            'samples_normals': None,
            'normals': None,
            'b_min': minBB,
            'b_max': maxBB
        }

    def get_item(self, index):
        sid = index % len(self.subjects)
        subject = self.subjects[sid]

        res = {
            'name': subject,
            'mesh_path': os.path.join(self.root, subject + '.obj'),
            'sid': sid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)

        if False:
            for i in range(4):
                img = np.uint8((np.transpose(res['img'][i].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
                rot = render_data['calib'][i,:3, :3]
                trans = render_data['calib'][i,:3, 3:4]
                pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
                pts = 0.5 * (pts.numpy().T + 1.0) * self.opt.loadSize
                for p in pts:
                    img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)

                plt.imshow(img, cmap='gray')
                plt.show()

        return res

    def __getitem__(self, index):
        return self.get_item(index)
