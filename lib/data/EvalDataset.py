from .TrainDataset import TrainDataset
from src.utils import viewUtils
import numpy as np

class EvalDataset(TrainDataset):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt):
        super(EvalDataset, self).__init__(opt, phase='eval')
        self.opt = opt

    def __len__(self):
        return 1

    def setBoundingBox(self, bounding_box):
        self.bounding_box = bounding_box

        extents = np.array(bounding_box['max']) - np.array(bounding_box['min'])
        min_bb = -extents/2 - 0.01
        max_bb = extents/2 + 0.01

        self.B_MIN = min_bb
        self.B_MAX = max_bb

    def set_views(self, views):
        self.views = views

        if self.opt.use_gan_input:
            self.views = viewUtils.resizeViews(self.views, self.opt.loadSize)
        #self.views = viewUtils.colorcodeViews(self.views)

    def get_item(self, index):
        subject = 'real_blueprint_test'

        res = {
            'name': subject,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
            'samples': None,
            'labels': None,
            'samples_normals': None,
            'normals': None,
            'edges':None
        }

        render_data = self.get_render(subject, num_views=self.num_views)
        res.update(render_data)

        if self.opt.use_gan_input:
            gan_output = self.applyGANToViews(render_data)
            res.update(gan_output)

        return res

    def __getitem__(self, index):
        return self.get_item(index)
