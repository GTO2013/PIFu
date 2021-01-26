import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *
from lib.data import EvalDataset
from lib.custom_collate import MultiViewCollator

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
#opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize

        # set cuda
        device_ids = [int(i) for i in opt.gpu_ids.split(",")]
        cuda = torch.device('cuda:%d' % device_ids[0])

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if False and opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        self.coll = MultiViewCollator(self.opt)
        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def eval(self, views, bounding_box, save_path, use_octree=True):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''

        dataset = EvalDataset(self.opt)
        extents = np.array(bounding_box['max']) - np.array(bounding_box['min'])
        min_bb = -extents/2 - 0.05
        max_bb = extents/2 + 0.05

        dataset.setBoundingBox(min_bb, max_bb)
        dataset.set_views(views)

        data = self.coll([dataset[0]])
        data = move_to_gpu(data, self.cuda)

        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            #save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, dataset[0]['name'])
            if self.netC:
                return gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                return gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)