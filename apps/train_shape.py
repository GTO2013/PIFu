import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.options import BaseOptions
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.train_metrics import calc_error
import apps.tensorboard_wrapper as tb
from lib.custom_collate import MultiViewCollator,move_to_gpu

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

    if len(device_ids) > 1:
        netG = MyDataParallel(HGPIFuNet(opt, train_dataset.projection_mode), device_ids=device_ids).to(device=cuda)
    else:
        netG = HGPIFuNet(opt, train_dataset.projection_mode).to(device=cuda)

    #netG = HGPIFuNet(opt, projection_mode).to(device=cuda)

    #optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate
    print('Using Network: ', netG.name)
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        netG.load_state_dict(torch.load(model_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s/training' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s/test' % (opt.results_path, opt.name), exist_ok=True)

    BaseOptions().saveOptToFile(opt)

    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()

        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()
            optimizerG.zero_grad()

            train_data = move_to_gpu(train_data,cuda)
            res, nmls, error = netG.forward(train_data['images'], train_data['samples'], train_data['calib'], imgSizes=train_data['size'],
                                  labels=train_data['labels'], points_nml=train_data['samples_normals'], labels_nml=train_data['normals'])

            error['Err(cmb)'].mean().backward()

            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                normal_loss = 0
                if opt.use_normal_loss:
                    normal_loss = error['Err(nml)'].mean().item()
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err (Cmb): {4:.06f} | Err(Occ): {5:.06f} |  Err(Nml): {6:.06f} | LR: {7:.06f} | dataT: {8:.05f} | netT: {9:.05f} | ETA: {10:02d}:{11:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader), error['Err(cmb)'].mean().item(), error['Err(occ)'].mean().item(), normal_loss, lr, iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))

            if not opt.debug and train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            if train_idx % opt.freq_save_ply == 0:
                save_path = '%s/%s/pred.ply' % (opt.results_path, opt.name)
                r = res[0].cpu()
                points = train_data['samples'][0].transpose(0, 1).cpu()
                save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())

            iter_data_time = time.time()

        # update learning rate
        if isinstance(optimizerG, torch.optim.RMSprop):
            lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

        if len(device_ids) > 1:
            netG.device_ids = [device_ids[0]]

        #### test
        with torch.no_grad():
            set_eval()

            if not opt.no_num_eval:
                test_losses = {}
                print('calc error (test) ...')
                test_errors = calc_error(coll, netG, cuda, test_dataset, 100)
                print('eval test MSE: {0:06f} IOU: {1:06f} prec: {2:06f} recall: {3:06f}'.format(*test_errors))
                MSE, IOU, prec, recall = test_errors
                test_losses['MSE(test)'] = MSE
                test_losses['IOU(test)'] = IOU
                test_losses['prec(test)'] = prec
                test_losses['recall(test)'] = recall

                print('calc error (train) ...')
                train_dataset.is_train = False
                train_errors = calc_error(coll,netG, cuda, train_dataset, 100)
                train_dataset.is_train = True
                print('eval train MSE: {0:06f} IOU: {1:06f} prec: {2:06f} recall: {3:06f}'.format(*train_errors))
                MSE, IOU, prec, recall = train_errors
                test_losses['MSE(train)'] = MSE
                test_losses['IOU(train)'] = IOU
                test_losses['prec(train)'] = prec
                test_losses['recall(train)'] = recall

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

                tb.updateAfterEpoch(epoch, train_errors, test_errors, train_data['img'], test_data['img'])

        if len(device_ids) > 1:
            netG.device_ids = device_ids

if __name__ == '__main__':
    train(opt)

