import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
from lib.model import LossNetwork
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.data.TrainDataset import TrainDataset
import torch.nn.functional as F

from lib.options import BaseOptions
from lib.train_util import *
from lib.model import *
from lib.train_metrics import calc_error
import apps.tensorboard_wrapper as tb
from lib.custom_collate import MultiViewCollator,move_to_gpu,reshape_sample_tensor
import matplotlib.pyplot as plt
import cv2

# get options
opt = BaseOptions().parse()

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def train(opt):
    # set cuda
    device_ids = [int(i) for i in opt.gpu_ids.split(",")]
    cuda = torch.device('cuda:%d' % device_ids[0])

    tb.initWriter(opt)

    train_dataset = TrainDataset(opt, phase='train')
    test_dataset = TrainDataset(opt, phase='test')

    print("Loading training data...")
    coll = MultiViewCollator(opt)
    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,collate_fn=coll,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))
    # NOTE: batch size should be 1 and use all the points for evaluation
    #test_data_loader = DataLoader(test_dataset,
    #                              batch_size=1, shuffle=True, collate_fn=coll,
    #                              num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_dataset))

    # create net
    print("Num GPUs: " + str(torch.cuda.device_count()))

    name = "multiview_pifu_OCC_hg_bp_64_5000_nml_loss_sds_mlp"
    model_opt = BaseOptions().loadOptFromFile(name= name)

    if len(device_ids) > 1:
        netG = MyDataParallel(HGPIFuNet(model_opt, train_dataset.projection_mode), device_ids=device_ids).to(device=cuda)
    else:
        netG = HGPIFuNet(model_opt, train_dataset.projection_mode).to(device=cuda)

    model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, name, 20)
    netG.load_state_dict(torch.load(model_path, map_location=cuda))

    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate
    print('Using Network: ', netG.name)
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s/training' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s/test' % (opt.results_path, opt.name), exist_ok=True)

    BaseOptions().saveOptToFile(opt)
    vgg_loss = LossNetwork.VGGPerceptualLoss(False).to(device=cuda)
    scaler = torch.cuda.amp.GradScaler()
    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()

        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            train_data = move_to_gpu(train_data, cuda)
            image_idx = 0

            #features = netG.filter(train_data['images'])
            sampled_points = 0
            num_points = 50000
            points = train_data['samples_depth'][image_idx]
            total_points = points.shape[2]

            current_normals_list = []
            while False and sampled_points < total_points:
                num_points = min(num_points, total_points - sampled_points)
                slice = points[:,:,sampled_points:num_points+sampled_points]
                #netG.query(slice, train_data['calib'], train_data['size'])
                #pred = netG.get_preds()
                #new_nmls = torch.autograd.grad(outputs=pred, inputs=[slice[0:1,:,:]], grad_outputs=torch.ones_like(slice[0:1,:,:]),create_graph=True, retain_graph=True)
                #new_nmls = netG.forward(reshape_sample_tensor(slice, opt.num_views), train_data['calib'],train_data['size'], fd_type='forward')
                points_query = reshape_sample_tensor(slice, opt.num_views)
                new_nmls = netG.calc_normal(features, points_query, train_data['calib'], train_data['size'], fd_type='forward')
                current_normals_list.append(new_nmls[image_idx::opt.num_views, :, :])
                sampled_points += num_points

            #points_query = reshape_sample_tensor(points, opt.num_views)
            points_query = reshape_sample_tensor(points, opt.num_views)

            gt_normals_img = train_data['normal_images'][image_idx].to(dtype=torch.float32)
            gt_normals_img_mask = (gt_normals_img > 0).to(dtype=torch.float32)

            sdf, normals, errorPred = netG.forward(train_data['images'], train_data['samples'], train_data['calib'], train_data['size'],
                                                labels=train_data['labels'], points_nml=points_query, labels_nml=None)


            img_orig_shape = train_data['normal_images'][image_idx].shape
            pred_normals_img = torch.reshape(normals, (-1, 3, img_orig_shape[2], img_orig_shape[3]))
            pred_normals_img = pred_normals_img * 0.5 + 0.5
            pred_normals_img = F.normalize(pred_normals_img, dim=1, eps=1e-8)
            pred_normals_img *= gt_normals_img_mask

            #plt.imshow(gt_normals_img[0].detach().cpu().numpy().transpose(1,2,0))
            #plt.show()
            preview = np.concatenate([pred_normals_img[0].detach().cpu().numpy().transpose(1,2,0), gt_normals_img[0].detach().cpu().numpy().transpose(1,2,0)], axis=1)
            cv2.imshow("prediction", preview)
            cv2.waitKey(1)
            #plt.imshow(pred_normals_img[0].detach().cpu().numpy().transpose(1,2,0))
            #plt.show()

            #mse = torch.nn.MSELoss()
            #error = mse(pred_normals_img, gt_normals_img) + errorPred['Err(cmb)'].mean()

            error = vgg_loss(pred_normals_img, gt_normals_img) + errorPred['Err(cmb)'].mean()*2
            scaler.scale(error.mean()).backward()
            scaler.step(optimizerG)
            scaler.update()
            optimizerG.zero_grad()
            netG.zero_grad()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err(VGG): {4:.06f} | LR: {5:.06f} | dataT: {6:.05f} | netT: {7:.05f} | ETA: {8:02d}:{9:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader), error, lr, iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))

            if not opt.debug and train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            if train_idx % opt.freq_save_ply == 0:
                save_path = '%s/%s/nml.ply' % (opt.results_path, opt.name)
                normals = normals.detach().cpu().numpy()
                points = points.detach().cpu().numpy()
                save_samples_rgb(save_path, points[0].T, (normals[0].T + 1) / 2)

            iter_data_time = time.time()

        # update learning rate
        if isinstance(optimizerG, torch.optim.RMSprop):
            lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

        if len(device_ids) > 1:
            netG.device_ids = [device_ids[0]]

        #### test
        with torch.no_grad():
            set_eval()

            if not opt.no_gen_mesh:
                print('generate mesh (test) ...')
                test_data = None
                train_data = None
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    #test_data = test_dataset[random.randint(0, len(test_dataset) - 1)]
                    test_data = test_dataset[6]
                    test_data_batched = coll([test_data])
                    test_data_batched = move_to_gpu(test_data_batched, cuda)

                    save_path = '%s/%s/test/test_eval_epoch%d_%s.obj' % (opt.results_path, opt.name, epoch, test_data['name'])
                    mesh_test = gen_mesh(opt, netG, cuda, test_data_batched, save_path)

                print('generate mesh (train) ...')
                train_dataset.is_train = False
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    train_data = train_dataset[random.randint(0, len(test_dataset) - 1)]
                    train_data_batched = coll([train_data])
                    train_data_batched = move_to_gpu(train_data_batched, cuda)

                    save_path = '%s/%s/training/train_eval_epoch%d_%s.obj' % (opt.results_path, opt.name, epoch, train_data['name'])
                    mesh_train = gen_mesh(opt, netG, cuda, train_data_batched, save_path)
                train_dataset.is_train = True

                #tb.updateAfterEpoch(epoch, train_errors, test_errors, train_data['img'], test_data['img'])

        if len(device_ids) > 1:
            netG.device_ids = device_ids

if __name__ == '__main__':
    train(opt)

