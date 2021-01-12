import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/run_cleaned_wheels')

def updateDuringEpoch(train_idx, error_cmb, error_occ, error_nml, netTime):
    writer.add_scalar('current/Loss combined', error_cmb, train_idx)
    writer.add_scalar('current/Loss occupancy', error_occ, train_idx)
    writer.add_scalar('current/Loss normal', error_nml, train_idx)
    writer.add_scalar('current/Network Time', netTime, train_idx)

def updateAfterEpoch(epoch, train_errors, test_errors, mesh_train, mesh_test):
    MSE_test, IOU_test, prec_test, recall_test = test_errors
    writer.add_scalar('test/MSE', MSE_test, epoch)
    writer.add_scalar('test/IOU', IOU_test, epoch)
    writer.add_scalar('test/prec', prec_test, epoch)
    writer.add_scalar('test/recall', recall_test, epoch)

    MSE_train, IOU_train, prec_train, recall_train = train_errors
    writer.add_scalar('train/MSE', MSE_train, epoch)
    writer.add_scalar('train/IOU', IOU_train, epoch)
    writer.add_scalar('train/prec', prec_train, epoch)
    writer.add_scalar('train/recall', recall_train, epoch)

    if mesh_train[0] is not None:
        mesh_train[0] = np.expand_dims(mesh_train[0], 0)
        mesh_train[1] = np.expand_dims(mesh_train[1], 0)
        colors = np.full_like(mesh_train[0], 128)
        writer.add_mesh('training/mesh', global_step=epoch, vertices=mesh_train[0].copy(), faces = mesh_train[1].copy(), colors=colors.copy())
        for idx, img in enumerate(mesh_train[2]):
            writer.add_image('training/image_{0}'.format(idx), img[0], global_step=epoch)

    if mesh_test[0] is not None:
        mesh_test[0] = np.expand_dims(mesh_test[0], 0)
        mesh_test[1] = np.expand_dims(mesh_test[1], 0)
        colors = np.full_like(mesh_test[0], 128)
        writer.add_mesh('test/mesh', global_step=epoch, vertices=mesh_test[0].copy(), faces = mesh_train[1].copy(), colors=colors.copy())

        for idx, img in enumerate(mesh_test[2]):
            writer.add_image('test/image_{0}'.format(idx), img[0], global_step=epoch)
def close():
    writer.close()