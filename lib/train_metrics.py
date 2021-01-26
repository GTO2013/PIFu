import torch
import numpy as np
from tqdm import tqdm
from lib.custom_collate import move_to_gpu

def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt

def calc_error(coll,net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []

        for idx in tqdm(range(num_tests)):
            #ToDO: Normal Loss!
            train_data = move_to_gpu(coll([dataset[idx]]),cuda)

            res, nmls, error = net.forward(train_data['images'], train_data['samples'], train_data['calib'], imgSizes=train_data['size'],
                                      labels=train_data['labels'], points_nml=train_data['samples_normals'], labels_nml=train_data['normals'])

            IOU, prec, recall = compute_acc(res, train_data['labels'])

            erorr_arr.append(error['Err(occ)'].mean().item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def calc_error_color(opt, netG, netC, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)

            #if opt.num_views > 1:
                #color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)

            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)