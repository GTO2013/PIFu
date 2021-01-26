import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier, Fourier, SurfaceClassifierLinear, GaussianFourierFeatureTransform
from .SimpleEncoder import SimpleEncoder
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from .UnetFilter import UNet
from ..net_util import init_net
from siren import SIREN
from ..custom_loss import CustomBCELoss
import matplotlib.pyplot as plt


class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self, opt,projection_mode='orthogonal', criteria={'occ': CustomBCELoss(False, 0.5), 'nml': nn.MSELoss()}
                 #error_term = nn.SmoothL1Loss(beta=0.001)
                 #error_term=nn.MSELoss(),
                 #'nml': nn.CosineSimilarity()
                 ):

        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=criteria)

        self.name = 'multi_pifu_normals'

        self.opt = opt
        self.num_views = self.opt.num_views
        self.criteria = criteria

        self.image_filter1 = SimpleEncoder(opt)
        #self.image_filter1 = HGFilter(opt)
        #self.image_filter2 = HGFilter(opt)
        #self.image_filter3 = HGFilter(opt)
        #self.image_filter4 = HGFilter(opt)

        #self.image_filter = UNet(in_channels=3, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upsample')

        if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier':
            fourier = self.opt.mlp_type == 'mlp_fourier'
            self.surface_classifier = SurfaceClassifierLinear(filter_channels=self.opt.mlp_dim, opt=opt, use_fourier = fourier, num_views=self.opt.num_views, no_residual=self.opt.no_residual, last_op=nn.Sigmoid())
        elif self.opt.mlp_type == 'conv1d':
            self.surface_classifier = SurfaceClassifier(filter_channels=self.opt.mlp_dim[:-1], num_views=self.opt.num_views, no_residual=self.opt.no_residual)
            self.surface_classifier_final = SurfaceClassifier(filter_channels=[512,1024,256,128,1], num_views=self.opt.num_views, no_residual=self.opt.no_residual, last_op=nn.Sigmoid())
        elif self.opt.mlp_type == 'siren':
            self.surface_classifier = SIREN(layers=self.opt.mlp_dim[:-1], in_features=self.opt.mlp_dim[0], out_features=1, w0=1.0, w0_initial=30.0, initializer='siren', c=6)
        else:
            raise RuntimeError("MLP Type is unknown!")

        #self.gauss = Fourier(opt)
        #self.gauss = GaussianFourierFeatureTransform(3,64)
        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.nmls = None
        self.labels_nml = None

        self.intermediate_preds_list = []
        self.images = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list = []
        self.images = images

        for idx, img in enumerate(images):
            #img = self.gauss(img)

            #if idx == 0:
            #    feat, _, _ = self.image_filter1(img)
            #elif idx == 1:
            #    feat, _, _ = self.image_filter2(img)
            #elif idx == 2:
            #    feat, _, _ = self.image_filter3(img)
            #elif idx == 3:
            #    feat, _, _ = self.image_filter4(img)

            feat = self.image_filter1(img)

            # If it is not in training, only produce the last im_feat
            #if not self.training:
            #feat = [feat[-1]]

            self.im_feat_list.append(feat)

        #self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)

        #if not self.training:
            #self.im_feat_list = [self.im_feat_list[-1]]

    def calc_pred(self, points, calibs, imgSizes, transforms=None):
        xyz = self.projection(points.repeat(self.opt.num_views, 1, 1), calibs, transforms)

        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        point_local_feat_list = []

        # We do this for every image per batch
        for idx, im_feat_list in enumerate(self.im_feat_list):
            for im_feat in im_feat_list:
                # We need to adjust the UV coordinates now because each image has a different size
                currentXY = xy[idx::self.num_views]
                sizes = imgSizes[idx::self.num_views].unsqueeze(2)

                #skip = torch.cat([self.images[idx], im_feat], dim=1)

                #pixel_local = self.index(self.images[idx], currentXY).cpu().detach().numpy()
                #img_local = np.swapaxes(self.images[idx][0].cpu().detach().numpy(),0,2)
                #new_img = np.zeros_like(img_local)
                #new_xy = currentXY.cpu().detach().numpy()[0] * np.reshape(new_img.shape[:2],(2,1))
                #new_xy = ((new_xy+1)/2).astype(np.uint32)
                #new_img[new_xy.T[:,0], new_xy.T[:,1]] = pixel_local[0].T

                #plt.imshow(new_img, cmap='gray')
                #plt.show()

                point_local_feat = self.index(im_feat, currentXY * sizes)
                currentZ = z[idx::self.num_views]/2
                point_local_feat = torch.cat([point_local_feat, currentZ], dim=1)
                point_local_feat_list.append(point_local_feat)

        currentNumStacks = len(self.im_feat_list[0])
        pred = None

        for i in range(currentNumStacks):
            multi = torch.cat(point_local_feat_list, dim=1)
            #multi = torch.cat(point_local_feat_list, dim=0)
            #multi = torch.cat([multi, points], dim=1)
            #points_trimmed = points[::self.num_views, :, :]
            #points_fourier = self.gauss(points_trimmed)
            #multi = torch.cat([multi, points_trimmed], dim=1)

            # Reshape for SIREN
            if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier' or self.opt.mlp_type == 'siren':
                batchsize = multi.shape[0]
                multi = multi.permute(0,2,1)
                #multi = multi.reshape((multi.shape[0], multi.shape[1], multi.shape[2]))

            pred = self.surface_classifier(multi)

            # SIREN Reshape
            if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier' or self.opt.mlp_type == 'siren':
                pred = pred.reshape((batchsize, 1, -1))

        return pred

    def calc_pred_exp(self, points, calibs, imgSizes, transforms=None):
        xyz = self.projection(points.repeat(self.opt.num_views, 1, 1), calibs, transforms)

        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        point_local_feat_list = []

        # We do this for every image per batch
        for idx, im_feat_list in enumerate(self.im_feat_list):
            # We need to adjust the UV coordinates now because each image has a different size
            currentXY = xy[idx::self.num_views]
            sizes = imgSizes[idx::self.num_views].unsqueeze(2)

            for im_feat in im_feat_list:
                # skip = torch.cat([self.images[idx], im_feat], dim=1)

                # pixel_local = self.index(self.images[idx], currentXY).cpu().detach().numpy()
                # img_local = np.swapaxes(self.images[idx][0].cpu().detach().numpy(),0,2)
                # new_img = np.zeros_like(img_local)
                # new_xy = currentXY.cpu().detach().numpy()[0] * np.reshape(new_img.shape[:2],(2,1))
                # new_xy = ((new_xy+1)/2).astype(np.uint32)
                # new_img[new_xy.T[:,0], new_xy.T[:,1]] = pixel_local[0].T

                # plt.imshow(new_img, cmap='gray')
                # plt.show()

                point_local_feat = self.index(im_feat, currentXY * sizes)
                point_local_feat_list.append(point_local_feat)
                currentZ = z[idx::self.num_views]
                point_local_feat = torch.cat([point_local_feat, currentZ], dim=1)
                point_local_feat_list.append(point_local_feat)


        multi = torch.cat(point_local_feat_list, dim=0)

        # Reshape for SIREN
        if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier' or self.opt.mlp_type == 'siren':
            batchsize = multi.shape[0]
            multi = multi.permute(0, 2, 1)
            # multi = multi.reshape((multi.shape[0], multi.shape[1], multi.shape[2]))

        pred = self.surface_classifier(multi)
        pred = self.surface_classifier_final(pred.reshape((points.shape[0], 512, -1)))

        # SIREN Reshape
        if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier' or self.opt.mlp_type == 'siren':
            pred = pred.reshape((batchsize, 1, -1))

        return pred

    def calc_normal(self, points, calibs, imgSizes, transforms=None, labels=None, delta=0.0001, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference
            fd_type: finite difference type (forward/backward/central)
        '''
        pdx = points.clone()
        pdx[:, 0, :] += delta
        pdy = points.clone()
        pdy[:, 1, :] += delta
        pdz = points.clone()
        pdz[:, 2, :] += delta

        if labels is not None:
            self.labels_nml = labels

        points_all = torch.stack([points, pdx, pdy, pdz], 3)
        points_all = points_all.view(*points.size()[:2], -1)

        pred = self.calc_pred(points_all, calibs, imgSizes, transforms)
        pred = pred.view(pred.shape[0], 1, -1, 4)  # (B, 1, N, 4)

        # divide by delta is omitted since it's normalized anyway
        dfdx = pred[:, :, :, 1] - pred[:, :, :, 0]
        dfdy = pred[:, :, :, 2] - pred[:, :, :, 0]
        dfdz = pred[:, :, :, 3] - pred[:, :, :, 0]

        #here was a minus
        nml = -torch.cat([dfdx, dfdy, dfdz], 1)
        nml = F.normalize(nml, dim=1, eps=1e-8)
        self.nmls = nml

    def query(self, points, calibs, imgSizes = None, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''

        if labels is not None:
            self.labels = labels

        self.intermediate_preds_list = [self.calc_pred(points,calibs, imgSizes, transforms)]
        self.preds = self.intermediate_preds_list[0]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self, gamma=0):
        '''
        Hourglass has its own intermediate supervision scheme
        '''

        error = {'Err(occ)': 0, 'Err(cmb)': 0}

        for preds in self.intermediate_preds_list:
            #error['Err(occ)'] += self.criteria['occ'](preds, self.labels, gamma)
            error['Err(occ)'] += self.criteria['occ'](preds, self.labels).unsqueeze(0)

        error['Err(occ)'] /= len(self.intermediate_preds_list)

        if self.opt.use_normal_loss and self.nmls is not None and self.labels_nml is not None:
            error['Err(nml)'] = (self.criteria['nml'](self.nmls, self.labels_nml).unsqueeze(0))
            error['Err(cmb)'] = error['Err(occ)'] + error['Err(nml)']*0.01
        else:
            error['Err(cmb)'] = error['Err(occ)']

        return error

    def forward(self, images, points, calibs, imgSizes=None, transforms=None, labels=None, points_nml=None, labels_nml=None):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, imgSizes = imgSizes, transforms=transforms, labels=labels)

        if self.opt.use_normal_loss and points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calibs, imgSizes, labels=labels_nml)

            if self.opt.debug:
                gt_labels_rgb = (labels_nml[0].cpu().detach().numpy() + 1) / 0.5
                pred_labels_rgb = (self.nmls[0].cpu().detach().numpy() + 1) / 0.5
                save_samples_rgb('pointclouds/normals_gt.ply', points_nml[0].cpu().detach().numpy().T, gt_labels_rgb.T)
                save_samples_rgb('pointclouds/normals_pred.ply', points_nml[0].cpu().detach().numpy().T, pred_labels_rgb.T)

        # get the prediction
        res = self.get_preds()

        # get the error
        error = self.get_error()

        return res, self.nmls,  error