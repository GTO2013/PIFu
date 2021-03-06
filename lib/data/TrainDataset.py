from torch.utils.data import Dataset
import torch.nn.functional as F
from .ParallelDataLoader import loadData, loadDataParallel

from src.datasetInterfaces import datasetInterfaceUnity, datasetInterfaceProcessed
from src.utils import viewUtils
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch
from PIL.ImageFilter import GaussianBlur
import cv2
import matplotlib.pyplot as plt
import logging
import math
from ..train_util import make_rotate
from src.pix2pix.options.test_options import TestOptions
from src.pix2pix.models import create_model

log = logging.getLogger('trimesh')
log.setLevel(40)


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'
        self.is_train = (phase == 'train')
        self.is_eval = (phase == 'eval')

        sub_dir = "training" if self.is_train else "test"

        # Path setup
        self.root = os.path.join(self.opt.dataroot, sub_dir)
        self.B_MIN = np.array([-0.55, -0.55, -0.55])
        self.B_MAX = np.array([0.55, 0.55, 0.55])

        self.load_size = self.opt.loadSize
        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color
        self.num_sample_normals = self.opt.num_sample_normals

        self.yaw_list = list(range(0, 181, 90))
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

        self.pertubed_points_dic = None
        self.points_surface_dic = None
        self.sdf_dic = None
        self.normals_dic = None
        self.edges_dic = None

        self.views = None
        self.gan_model = None

        if self.opt.use_gan_input:
            opt_gan = TestOptions().parse(["--input_nc", "3", "--output_nc", "3",
                                          "--preprocess", "none",
                                          "--epoch", str(self.opt.gan_epoch),
                                          "--direction", "AtoB",
                                          "--batch_size", "1",
                                          "--gpu_ids", "-1",
                                          "--name", "blueprint2ND",
                                          "--model", "pix2pix",
                                          "--dataset_mode", "template"])

            opt_gan.num_threads = 0
            opt_gan.batch_size = 1
            opt_gan.serial_batches = True
            opt_gan.no_flip = True
            opt_gan.display_id = -1
            opt_gan.test = True
            opt_gan.checkpoints_dir = r"C:\Blueprint2Car\checkpoints"

            self.gan_model = create_model(opt_gan)  # create a model given opt.model and other options
            self.gan_model.setup(opt_gan)  # regular setup: load and print networks; create schedulers

        if not self.is_eval:
            points, sdf, pointsNormals, normals, edges = loadDataParallel(self.root, self.subjects, self.opt)

            self.pertubed_points_dic = points
            self.points_surface_dic = pointsNormals
            self.sdf_dic = sdf
            self.normals_dic = normals
            self.edges_dic = edges

    def get_subjects(self):
        all_subjects = []
        list_all = os.listdir(self.root)

        if self.opt.max_train_size > 0:
            list_all = list_all[:min(len(list_all), self.opt.max_train_size)]

        for subject in list_all:
            try:
                path = os.path.join(self.root, subject, 'sdf.npy')
                if os.path.exists(path):
                    test_datatype = np.load(path, allow_pickle=True)
                    if test_datatype.dtype is np.dtype(np.float32):
                        if os.path.exists(os.path.join(self.root, subject, 'top_Normals.npy' if self.opt.use_normal_input else 'top_Blueprint.npy')):
                            datasetInterfaceUnity.getBoundingBox(os.path.join(self.root, subject))
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

        if self.opt.use_normal_input:
            views = datasetInterfaceProcessed.getViewsNormals(sample_path)
        else:
            views = datasetInterfaceProcessed.getViewsBlueprint(sample_path)

        if not self.opt.use_normal_input and not self.is_eval:
            views = viewUtils.augmentViews(views, self.opt.random_scale, self.opt.loadSize, bb)

        if self.opt.use_gan_input or self.opt.use_normal_input:
            views = viewUtils.resizeViews(views, self.opt.loadSize)

        if self.opt.use_gan_input:
            views = viewUtils.colorcodeViews(views)

        if self.opt.use_normal_input:
            views = viewUtils.trimViewsByBB(views, bb)

        #viewUtils.showViews(views)

        self.bounding_box = bb
        self.views = views

    def get_render(self, subject, num_views):
        """
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
        """

        if num_views == 4:
            yaw_list = [0, 90, 90, 180]
        elif num_views == 3:
            yaw_list = [0, 90, 180]
        else:
            yaw_list = [90]
        
        calib_list = []
        render_list = []
        extrinsic_list = []
        normal_list = []
        depth_list = []

        normal_views = None
        depth_views = None

        if self.opt.render_normals:
            normal_views = datasetInterfaceProcessed.getViewsNormals(os.path.join(self.root, subject))
            normal_views = viewUtils.resizeViews(normal_views, self.opt.loadSize)
            normal_views = viewUtils.trimViewsByBB(normal_views, self.bounding_box)

            depth_views = datasetInterfaceProcessed.getViewsDepth(os.path.join(self.root, subject))
            depth_views = viewUtils.resizeViews(depth_views, self.opt.loadSize)
            depth_views = viewUtils.trimViewsByBB(depth_views, self.bounding_box)

        # random flip
        flip = False
        #if self.is_train and self.opt.random_flip and np.random.rand() > 0.5:
            #flip = True

        scale = viewUtils.getMaxDimension(self.views)

        for idx, yaw in enumerate(yaw_list):
            pitch = 0
            if num_views == 4 and idx == 2:
                pitch = 90

            poseName = self.angles_to_name(yaw, pitch)

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
            uv_intrinsic[0, 0] = 1.0 / float(scale // 2)
            uv_intrinsic[1, 1] = 1.0 / float(scale // 2)
            uv_intrinsic[2, 2] = 1.0 / float(scale // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            render = Image.fromarray(self.views[poseName], mode='RGB' if self.opt.use_normal_input else 'L')

            if self.opt.use_normal_input:
                render = render.convert('L')

            if self.opt.render_normals:
                normal = Image.fromarray(normal_views[poseName], mode='RGB')
                normal_img_tensor = transforms.ToTensor()(normal)
                #normal_img_tensor = F.normalize(normal_img_tensor, dim=0, eps=1e-8)
                depth_list.append(np.expand_dims(depth_views[poseName], 0))
                normal_list.append(normal_img_tensor)

            #render = render.resize((self.load_size, self.load_size), Image.BILINEAR)

            if True:#(yaw == 90 and pitch == 0) or (yaw == 90 and pitch == 90):
                render = ImageOps.mirror(render)

                #if flip:
                #    scale_intrinsic[0, 0] *= -1
                #    render = transforms.RandomHorizontalFlip(p=1.0)(render)

            if self.is_train:
                # Pad images
                #pad_size = int(0.1 * self.load_size)
                #render = ImageOps.expand(render, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(scale // 2)
                trans_intrinsic[1, 3] = -dy / float(scale // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                #render = render.crop((x1, y1, x1 + tw, y1 + th))

                if not self.opt.use_normal_input:
                    render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic))
            extrinsic = torch.Tensor(extrinsic)

            if not self.opt.use_gan_input:
                render = self.to_tensor(render)
                #plt.imshow(render.permute(1,2,0), cmap='gray')
                #plt.show()

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        if not self.opt.render_normals:
            depth_list = None
            normal_list = None

        self.views = None

        return {
            'img': render_list,
            'img_nml': normal_list,
            'img_depth': depth_list,
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
        }

    def select_sampling_method(self, subject):
        if not self.is_train and self.opt.same_test_data:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)

        bb = self.bounding_box
        minBB = np.array([bb['min'][0] - 0.05, bb['min'][1] - 0.05, bb['min'][2] - 0.05]) - 0.5
        maxBB = np.array([bb['max'][0] + 0.05, bb['max'][1] + 0.05, bb['max'][2] + 0.05]) - 0.5

        normals_points = None
        normals = None
        edges = None

        if subject in self.pertubed_points_dic:
            points = self.pertubed_points_dic[subject]
            sdf = self.sdf_dic[subject]

            if self.opt.use_normal_loss or self.opt.use_edge_loss:
                normals_points = self.points_surface_dic[subject]

            if self.opt.use_normal_loss:
                normals = self.normals_dic[subject]

            if self.opt.use_edge_loss:
                edges = self.edges_dic[subject]
        else:
            points, sdf, normals_points, normals, edges = loadData(self.root, subject, self.opt)

        #Occupany sampling
        num = self.num_sample_inout

        if self.opt.regression:
            sdf = np.clip(sdf, -self.opt.reg_distance, self.opt.reg_distance)
            sdf = sdf * (1/self.opt.reg_distance)
            sdf = -sdf * 0.5 + 0.5
            maskInside = sdf >= 0.5
            maskOutside = sdf < 0.5
        else:
            maskInside = sdf
            maskOutside = np.logical_not(sdf)

        inside = sdf[maskInside]
        outside = sdf[maskOutside]

        insideIdx = np.random.choice(inside.shape[0], min(inside.shape[0], num//2), replace=False)
        outsideIdx = np.random.choice(outside.shape[0], min(outside.shape[0], num - insideIdx.shape[0]), replace=False)

        samples = np.concatenate([points[maskInside][insideIdx], points[maskOutside][outsideIdx]], axis=0)
        labels = np.concatenate([inside[insideIdx], outside[outsideIdx]], axis=0)

        samples = samples.T
        labels = np.expand_dims(labels.T, 0)

        #Normal sampling
        if self.opt.use_normal_loss or self.opt.use_edge_loss:
            indexNormal = np.random.choice(normals_points.shape[0], self.num_sample_normals, replace=False)
            normals_points = normals_points[indexNormal].T
            normals_points = torch.Tensor(normals_points).float()

        if self.opt.use_normal_loss:
            normals = normals[indexNormal].T
            normals = torch.Tensor(normals).float()
            # Make sure they are normalized
            normals = F.normalize(normals, dim=0, eps=1e-8)

        if self.opt.use_edge_loss:
            edges = edges[indexNormal].T
            edges = torch.Tensor(edges).float()

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()

        return {
            'samples': samples,
            'labels': labels,
            'samples_normals': normals_points,
            'normals': normals,
            'edges': edges,
            'b_min': minBB,
            'b_max': maxBB
        }

    def applyGANToViews(self, render_data):
        new_views = []
        #back,side,top,front
        for idx, view in enumerate(render_data['img']):
            A = view
            #A = Image.fromarray(view, mode="RGB")
            A = self.to_tensor(A).unsqueeze(0)

            with torch.no_grad():
                self.gan_model.real_A = A
                self.gan_model.forward()

            img = self.gan_model.fake_B.cpu().numpy()[0, :, :, :]
            img = np.transpose(img, [1, 2, 0])
            new_views.append(img)

        views_dict = {'back': new_views[0], 'side': new_views[1], 'top': new_views[2], 'front': new_views[3]}
        #new_views = views_dict
        new_views = viewUtils.trimViewsByBB(views_dict, self.bounding_box)
        #viewUtils.showViews(new_views)

        for key in new_views:
            #new_views[key] = np.transpose(new_views[key], [1, 2, 0])
            new_views[key] = np.transpose(new_views[key], [2,0,1])
            new_views[key] = torch.Tensor(new_views[key])

        return {"img": [new_views['back'], new_views['side'], new_views['top'], new_views['front']]}

    def get_item(self, index):
        sid = index % len(self.subjects)

        subject = self.subjects[sid]
        self.set_views_from_path(subject)

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

        if self.opt.use_gan_input:
            gan_output = self.applyGANToViews(render_data)
            res.update(gan_output)

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

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res

    def __getitem__(self, index):
        return self.get_item(index)
