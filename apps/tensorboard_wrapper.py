import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

writer = None

def initWriter(opt):
    global writer

    #Dont create a new dir when we are only testing
    if opt.max_train_size == -1 and not opt.debug:
        path = os.path.join(opt.tensorboard_path,'/{0}/{1}'.format(opt.name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        writer = SummaryWriter(log_dir=path)

def updateAfterEpoch(epoch, train_errors, test_errors, images_train, images_test):
    if writer is not None:
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

        for idx, img in enumerate(images_train):
            writer.add_image('training/image_{0}'.format(idx), img, global_step=epoch)
        for idx, img in enumerate(images_test):
            writer.add_image('test/image_{0}'.format(idx), img, global_step=epoch)

        if False:
            if mesh_train[0] is not None:
                mesh_train[0] = np.expand_dims(mesh_train[0], 0)
                mesh_train[1] = np.expand_dims(mesh_train[1], 0)
                colors = np.full_like(mesh_train[0], 128)
                writer.add_mesh('training/mesh', global_step=epoch, vertices=mesh_train[0].copy(), faces = mesh_train[1].copy(), colors=colors.copy())

            if mesh_test[0] is not None:
                mesh_test[0] = np.expand_dims(mesh_test[0], 0)
                mesh_test[1] = np.expand_dims(mesh_test[1], 0)
                colors = np.full_like(mesh_test[0], 128)
                writer.add_mesh('test/mesh', global_step=epoch, vertices=mesh_test[0].copy(), faces = mesh_train[1].copy(), colors=colors.copy())
def close():
    writer.close()