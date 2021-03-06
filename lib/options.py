import argparse
import json, os

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataroot', type=str, default='./data', help='path to images (data folder)')
        g_data.add_argument('--tensorboard_path', type=str, default='./trainedModels/logs_pifu/', help='path to images (data folder)')
        g_data.add_argument('--loadSize', type=int, default=512, help='load size of input image')
        g_data.add_argument('--use_normal_input', action='store_true')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='multiview_pifu', help='name of the experiment')
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not')
        g_exp.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')
        g_exp.add_argument('--render_normals', action='store_true')
        g_exp.add_argument('--super_res', action='store_true')
        g_exp.add_argument("--regression", action='store_true')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        g_train.add_argument('--num_threads', default=1, type=int, help='#threads for loading data')
        g_train.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--same_test_data', action='store_true', help='if true, always use the same test data')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        
        g_train.add_argument('--batch_size', type=int, default=1, help='input batch size')
        #g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--learning_rate', type=float, default=1e-4, help='adam learning rate') # -4 before
        g_train.add_argument('--learning_rateC', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=40, help='num epoch to train')
        g_train.add_argument('--predict_normal', action='store_true')

        g_train.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply')
       
        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')
        
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma', type=float, default=.005, help='perturbation standard deviation for positions')
        g_sample.add_argument('--reg_distance', type=float, default=.005, help='regression distance threshold')

        g_sample.add_argument('--sample_on_surface', default=False, action='store_true', help='Sample on surface for occ')
        g_sample.add_argument('--use_normal_loss', default=False, action='store_true', help='Use normal loss or not')
        g_sample.add_argument('--use_edge_loss', default=False, action='store_true', help='Use edge loss or not')
        g_sample.add_argument('--occ_loss_weight', type=float, default=1, help='occ loss weight')
        g_sample.add_argument('--normal_loss_weight', type=float, default=0.5, help='normal loss weight')
        g_sample.add_argument('--edge_loss_weight', type=float, default=0.25, help='edge loss weight')
        g_sample.add_argument('--num_sample_normals', type=int, default=5000, help='# of sampling points')
        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')
        g_sample.add_argument('--num_sample_color', type=int, default=0, help='# of sampling points')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group', help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--norm_color', type=str, default='instance',
                             help='instance normalization or batch normalization or group normalization')

        # hg filter specify
        g_model.add_argument('--use_unet', action='store_true', help='Use a unet instead')
        g_model.add_argument('--use_gan_input', action='store_true', help='Use the input of the GAN')
        g_model.add_argument('--gan_epoch', type=int, default=135, help='GAN Epoch to be used')

        g_model.add_argument('--num_stack', type=int, default=2, help='# of hourglass')
        #g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass') #3 before
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')
        g_model.add_argument('--hourglass_dim_internal', type=int, default='128', help='256 | 512')
        g_model.add_argument('--skip_downsample', action='store_true')

        # Classification General
        g_model.add_argument('--mlp_type', type=str, default='conv1d', help='type of classifier to use')
        g_model.add_argument('--mlp_dim', nargs='+', default=[0, 512, 512, 256, 128, 1], type=int, help='# of dimensions of mlp')
        #g_model.add_argument('--mlp_dim', nargs='+', default=[0, 1024, 512, 256, 128, 1], type=int,help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_color', nargs='+', default=[513, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')

        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')

        # for train
        parser.add_argument('--random_flip', action='store_true', help='if random flip')
        parser.add_argument('--random_trans', action='store_true', help='if random flip')
        parser.add_argument('--random_scale', action='store_true', help='if random flip')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[10, 25, 60, 80],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--color_loss_type', type=str, default='l1', help='mse | l1')

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--num_gen_mesh_test', type=int, default=1, help='how many meshes to generate during testing')

        # path
        parser.add_argument('--decoder_base', type=str, default='', help='path to load a pretrained decoder')
        parser.add_argument('--checkpoints_path', type=str, default='./trainedModels', help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./generated3DModels', help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')
        parser.add_argument('--single', type=str, default='', help='single data for training')
        parser.add_argument('--max_train_size', type=int, default=-1, help='max number of training samples')

        #for single image reconstruction
        parser.add_argument('--img_path', type=str, help='path for input image')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')

        # special tasks
        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def setNameFromOptions(self, opt):
        baseName = opt.name
        type_name = "OCC"

        if opt.predict_normal:
            type_name = "NORMAL"
        elif opt.render_normals:
            type_name = "RENDER"

        input_type = "nml" if opt.use_normal_input else "bp"
        if opt.use_gan_input:
            input_type = "gan"

        filter_hg = str(opt.hourglass_dim)
        sample_count = str(opt.num_sample_inout)
        nml_loss = "nml_loss" if opt.use_normal_loss else ""
        edge_loss = 'edge_loss' if opt.use_edge_loss else ""
        skip_ds = "sds" if opt.skip_downsample else ""
        super_res = "superRes" if opt.super_res else ""
        unet = "unet" if opt.use_unet else "hg"
        mlp_type = opt.mlp_type
        mlp_sizes = '_'.join(str(x) for x in opt.mlp_dim)

        return '_'.join(str(x) for x in [baseName, type_name, unet, input_type, super_res, filter_hg, sample_count,
                                         nml_loss, edge_loss, skip_ds, mlp_type])

    def saveOptToFile(self, opt):
        savePath = '%s/%s/options.txt' % (opt.checkpoints_path, opt.name)
        with open(savePath, "w") as f:
            json.dump(opt.__dict__, f, indent=2)

        print("Saved options to {0}".format(savePath))

    def loadOptFromFile(self, name, checkPointsPath = "./trainedModels"):
        loadPath = '%s/%s/options.txt' % (checkPointsPath, name)

        if os.path.exists(loadPath):
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            opt = parser.parse_args("")

            with open(loadPath, 'r') as f:
                opt.__dict__ = json.load(f)

            return opt
        else:
            return None

    def parse(self):
        opt = self.gather_options()

        #Set first mlp dim according to filter sizes
        if opt.use_unet:
            opt.mlp_dim[0] = 512+3
        else:
            opt.mlp_dim[0] = opt.hourglass_dim * opt.num_views + 3

        #if opt.super_res:
        #    opt.mlp_dim[0] = opt.mlp_dim[0] +  opt.hourglass_dim//2 * opt.num_views

        if opt.predict_normal:
            opt.mlp_dim[-1] = 3

        #if opt.max_train_size != -1:
            #opt.no_gen_mesh = True

        opt.name = self.setNameFromOptions(opt)
        return opt
