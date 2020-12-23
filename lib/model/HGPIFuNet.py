import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier, SurfaceClassifierLinear
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from .UnetFilter import UNet
from ..net_util import init_net


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

    def __init__(self, opt,projection_mode='orthogonal', criteria={'occ': nn.MSELoss(), 'nml': nn.MSELoss()}
                 #error_term = nn.SmoothL1Loss(beta=0.001)
                 #error_term=nn.MSELoss(),
                 ):

        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=criteria)

        self.name = 'multi_pifu_normals'

        self.opt = opt
        self.num_views = self.opt.num_views
        self.criteria = criteria

        self.image_filter1 = HGFilter(opt)
        self.image_filter2 = HGFilter(opt)
        self.image_filter3 = HGFilter(opt)
        self.image_filter4 = HGFilter(opt)


        #self.image_filter = UNet(in_channels=3, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upsample')

        self.surface_classifier = SurfaceClassifierLinear(
            filter_channels=self.opt.mlp_dim,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.nmls = None
        self.labels_nml = None

        self.intermediate_preds_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list = []

        for idx, img in enumerate(images):
            if idx == 0:
                feat, _, _ = self.image_filter1(img)
            elif idx == 1:
                feat, _, _ = self.image_filter2(img)
            elif idx == 2:
                feat, _, _ = self.image_filter3(img)
            elif idx == 3:
                feat, _, _ = self.image_filter4(img)
            #feat = self.image_filter(img).unsqueeze(0)

            # If it is not in training, only produce the last im_feat
            #if not self.training:
            feat = [feat[-1]]

            self.im_feat_list.append(feat)

        #self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)

        #if not self.training:
            #self.im_feat_list = [self.im_feat_list[-1]]

    def calc_normal(self, points, calibs, imgSizes, transforms=None, labels=None, delta=0.004, fd_type='forward'):
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

        xyz = self.projection(points_all, calibs, transforms)
        xy = xyz[:, :2, :]

        #im_feat = self.im_feat_list[-1]
        #sp_feat = self.spatial_enc(xyz, calibs=calibs)

        point_local_feat_list = []
        for idx, im_feat_list in enumerate(self.im_feat_list):
            for im_feat in im_feat_list:
                # [B, Feat_i + z, N]
                # point_local_feat = self.index(im_feat, xy)

                #We need to adjust the UV coordinates now because each image has a different size
                currentXY = xy[idx::self.num_views]

                sizes = imgSizes[idx::self.num_views].unsqueeze(2)
                currentXY = currentXY * sizes

                point_local_feat = self.index(im_feat, currentXY)
                point_local_feat_list.append(point_local_feat)

        currentNumStacks = len(self.im_feat_list[0])

        preds = []
        for i in range(currentNumStacks):
            multi = torch.cat(point_local_feat_list[i::currentNumStacks], dim=1)
            multi = torch.cat([multi, points_all[::self.num_views, :, :]], dim=1)

            pred = self.surface_classifier(multi)
            preds.append(pred)

        pred = preds[-1]
        pred = pred.view(multi.shape[0], 1, -1, 4)  # (B, 1, N, 4)

        #point_local_feat_list = [self.index(im_feat, xy), sp_feat]
        #point_local_feat = torch.cat(point_local_feat_list, 1)

        #pred = self.mlp(point_local_feat)[0]
        #pred = pred.view(*pred.size()[:2], -1, 4)  # (B, 1, N, 4)

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

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        point_local_feat_list = []

        #We do this for every image per batch
        for idx, im_feat_list in enumerate(self.im_feat_list):
            for im_feat in im_feat_list:
                # [B, Feat_i + z, N]
                # point_local_feat = self.index(im_feat, xy)

                #We need to adjust the UV coordinates now because each image has a different size
                currentXY = xy[idx::self.num_views]

                sizes = imgSizes[idx::self.num_views].unsqueeze(2)
                currentXY = currentXY * sizes

                point_local_feat = self.index(im_feat, currentXY)
                point_local_feat_list.append(point_local_feat)

        self.intermediate_preds_list = []
        currentNumStacks = len(self.im_feat_list[0])

        for i in range(currentNumStacks):
            multi = torch.cat(point_local_feat_list[i::currentNumStacks], dim=1)
            multi = torch.cat([multi, points[::self.num_views, :, :]], dim=1)

            pred = self.surface_classifier(multi)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def queryBackup(self, points, calibs, transforms=None, labels=None):
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

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        #z = xyz[:, 2:3, :]

        #in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        #z_feat = self.normalizer(z, calibs=calibs)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        point_local_feat_list = []
        for idx, im_feat in enumerate(self.im_feat_list):
            # [B, Feat_i + z, N]
            #point_local_feat = self.index(im_feat, xy)
            point_local_feat = self.index(np.expand_dims(im_feat[idx], axis=0), np.expand_dims(xy[idx], axis=0))
            point_local_feat_list.append(point_local_feat)


            #if self.opt.skip_hourglass:
                #point_local_feat_list.append(tmpx_local_feature)

            #point_local_feat = torch.cat(point_local_feat_list, 1)

            total = point_local_feat
            #total = torch.cat([point_local_feat, points[::self.num_views, :, :]], 1)

            # out of image plane is always set to 0
            #print(point_local_feat.shape)

            multi = total.view(points.shape[0]//self.num_views, -1, points.shape[2])
            multi = torch.cat([multi, points[::self.num_views, :, :]], dim=1)
            #print(multi.shape)
            pred = self.surface_classifier(multi)
            #pred = in_img[:, None].float() * self.surface_classifier(multi)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

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
            error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml).unsqueeze(0)
            error['Err(cmb)'] = error['Err(occ)'] + error['Err(nml)']
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

        # get the prediction
        res = self.get_preds()

        # get the error
        error = self.get_error()

        return res, error