from lib.train_util import *
from lib.model import *
from lib.model import HGFilters
from lib.data import EvalDataset
from lib.custom_collate import MultiViewCollator

class Evaluator:
    def __init__(self, opt, gan_filter_path = None):
        self.opt = opt
        self.load_size = self.opt.loadSize

        # set cuda
        device_ids = [int(i) for i in opt.gpu_ids.split(",")]

        if device_ids[0] != -1:
            cuda = torch.device('cuda:%d' % device_ids[0])
        else:
            cuda = torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if gan_filter_path is not None:
            #opt.use_normal_input = False
            opt.num_stack = 3
            new_filter = HGFilters.HGFilter(opt).to(device=cuda)
            new_filter.load_state_dict(torch.load(gan_filter_path, map_location=cuda))
            netG.image_filter = new_filter

        self.coll = MultiViewCollator(self.opt)

        self.cuda = cuda
        self.netG = netG

    def eval(self, views, bounding_box, save_path, use_octree=True, num_samples=1000, extr_value = 0.5, predict_normal = False, dual_contouring=True):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''

        dataset = EvalDataset(self.opt)
        dataset.setBoundingBox(bounding_box)
        dataset.set_views(views)

        data = self.coll([dataset[0]])
        data = move_to_gpu(data, self.cuda)

        with torch.no_grad():
            self.netG.eval()
            verts, faces, normals = gen_mesh(self.opt, self.netG, self.cuda, data, save_path, extr_value = extr_value, use_octree=use_octree,
                                              predict_vertex_normals = predict_normal, num_samples=num_samples,
                                             dual_contouring=dual_contouring)
            return verts, faces, normals
