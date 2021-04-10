from .SurfaceClassifier import SurfaceClassifier, Fourier, SurfaceClassifierLinear, GaussianFourierFeatureTransform
from .HGFilters import *
from .UnetFilter import UNet
from ..net_util import init_net
from siren import SIREN
from ..geometry import orthogonal, index
from ..custom_loss import CustomBCELoss
import matplotlib.pyplot as plt

class HGPIFuNet(nn.Module):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """

    def __init__(self, opt,  criteria={'occ': nn.MSELoss(),
                                                                    'nml': nn.CosineSimilarity(),
                                                                    'edges': nn.MSELoss()}
                 #error_term = nn.SmoothL1Loss(beta=0.001)
                 #error_term=nn.MSELoss(),
                 #'nml': nn.CosineSimilarity()
                 #CustomBCELoss(False, 0.5)
                 ):
        super(HGPIFuNet, self).__init__()

        self.name = 'multi_pifu_normals'
        self.opt = opt
        self.num_views = self.opt.num_views
        self.index = index
        self.projection = orthogonal
        self.criteria = criteria

        if opt.use_unet:
            self.image_filter = UNet(in_channels=1, depth=5, wf=7, padding=True, batch_norm=True, up_mode='upsample')
        else:
            self.image_filter = HGFilter(opt)

        last_op = torch.tanh if opt.predict_normal else torch.sigmoid

        if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier':
            fourier = self.opt.mlp_type == 'mlp_fourier'
            self.surface_classifier = SurfaceClassifierLinear(filter_channels=self.opt.mlp_dim, opt=opt, use_fourier = fourier, num_views=self.opt.num_views, no_residual=self.opt.no_residual, last_op=last_op)
        elif self.opt.mlp_type == 'conv1d':
            self.surface_classifier = SurfaceClassifier(filter_channels=self.opt.mlp_dim, num_views=self.opt.num_views, no_residual=self.opt.no_residual, last_op=last_op)
        elif self.opt.mlp_type == 'siren':
            self.surface_classifier = SIREN(layers=self.opt.mlp_dim[:-1], in_features=self.opt.mlp_dim[0], out_features=self.opt.mlp_dim[-1], w0=1.0, w0_initial=30.0, initializer='siren', c=6)
        else:
            raise RuntimeError("MLP Type is unknown!")

        #self.gauss = Fourier(opt)
        #self.gauss = GaussianFourierFeatureTransform(3,64)
        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        features = []

        with torch.cuda.amp.autocast():
            for idx, img in enumerate(images):
                feat = self.image_filter(img)
                #feat = checkpoint.checkpoint(self.image_filter1,img)
                #feat = [feat[-1]]

                #if feat_high is not None:
                #    feat.append(feat_high)
                features.append(feat)
                #self.im_feat_list.append([feat])

        return features

    def calc_normal_edges(self, features, points, calibs, imgSizes, transforms=None, delta=0.0001, fd_type='central'):
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
        #If forward --> original points <---> + delta
        #If backward --> original points <---> - delta
        #If central -> +delta <---> -delta
        #if calc_edges --> original points <---> + delta  <-----> original points <---> - delta
        nml = None
        edges = None

        calc_jobs = dict()

        if self.opt.use_edge_loss:
            calc_jobs['forward'] = 0
            calc_jobs['center'] = 0
            calc_jobs['backward'] = 0
        elif self.opt.use_normal_loss:
            if fd_type == 'forward':
                calc_jobs['forward'] = 0
                calc_jobs['center'] = 0
            elif fd_type == 'backward':
                calc_jobs['backward'] = 0
                calc_jobs['center'] = 0
            elif fd_type == 'central':
                calc_jobs['forward'] = 0
                calc_jobs['backward'] = 0

        for key in calc_jobs:
            if key == 'center':
                points_all = torch.stack([points], 3)
                points_all = points_all.view(*points.size()[:2], -1)

                pred = self.query(features, points_all, calibs, imgSizes, transforms)
                pred = pred.view(pred.shape[0], 1, -1, 1)  # (B, 1, N, 4)
                calc_jobs['center'] = pred

            elif key == 'forward':
                pdx = points.clone()
                pdx[:, 0, :] = pdx[:, 0, :] + delta
                pdy = points.clone()
                pdy[:, 1, :] = pdy[:, 1, :] + delta
                pdz = points.clone()
                pdz[:, 2, :] = pdz[:, 2, :] + delta

                points_all = torch.stack([pdx, pdy, pdz], 3)
                points_all = points_all.view(*points.size()[:2], -1)

                pred = self.query(features, points_all, calibs, imgSizes, transforms)
                pred = pred.view(pred.shape[0], 1, -1, 3)  # (B, 1, N, 3)
                dfdx = pred[:, :, :, 0]
                dfdy = pred[:, :, :, 1]
                dfdz = pred[:, :, :, 2]

                calc_jobs['forward'] = [dfdx, dfdy, dfdz]

            elif key == 'backward':
                pdx = points.clone()
                pdx[:, 0, :] = pdx[:, 0, :] - delta
                pdy = points.clone()
                pdy[:, 1, :] = pdy[:, 1, :] - delta
                pdz = points.clone()
                pdz[:, 2, :] = pdz[:, 2, :] - delta

                points_all = torch.stack([pdx, pdy, pdz], 3)
                points_all = points_all.view(*points.size()[:2], -1)

                pred = self.query(features, points_all, calibs, imgSizes, transforms)
                pred = pred.view(pred.shape[0], 1, -1, 3)  # (B, 1, N, 3)
                dfdx = pred[:, :, :, 0]
                dfdy = pred[:, :, :, 1]
                dfdz = pred[:, :, :, 2]

                calc_jobs['backward'] = [dfdx, dfdy, dfdz]

        if fd_type == 'forward':
            nml = -torch.cat([
                                calc_jobs['forward'][0] - calc_jobs['center'][:, :, :, 0],
                                calc_jobs['forward'][1] - calc_jobs['center'][:, :, :, 0],
                                calc_jobs['forward'][2] - calc_jobs['center'][:, :, :, 0]], 1)
        elif fd_type == 'backward':
            nml = -torch.cat([
                                calc_jobs['center'][:, :, :, 0] - calc_jobs['backward'][0],
                                calc_jobs['center'][:, :, :, 0] - calc_jobs['backward'][1],
                                calc_jobs['center'][:, :, :, 0] - calc_jobs['backward'][2]], 1)
        elif fd_type == 'central':
            nml = -torch.cat([calc_jobs['forward'][0], calc_jobs['forward'][1], calc_jobs['forward'][2]], 1) \
                  + torch.cat([calc_jobs['backward'][0], calc_jobs['backward'][1], calc_jobs['backward'][2]], 1)

        nml = F.normalize(nml, dim=1, eps=1e-8)

        if self.opt.use_edge_loss:
            nml_fwd = -torch.cat([
                                calc_jobs['forward'][0] - calc_jobs['center'][:, :, :, 0],
                                calc_jobs['forward'][1] - calc_jobs['center'][:, :, :, 0],
                                calc_jobs['forward'][2] - calc_jobs['center'][:, :, :, 0]], 1)
            nml_bkw = -torch.cat([
                                calc_jobs['center'][:, :, :, 0] - calc_jobs['backward'][0],
                                calc_jobs['center'][:, :, :, 0] - calc_jobs['backward'][1],
                                calc_jobs['center'][:, :, :, 0] - calc_jobs['backward'][2]], 1)
            edges = nml_fwd - nml_bkw
            edges = torch.linalg.norm(edges, dim=1, keepdim=True)*100
            #print("Calculated Edges Min: {0}, Max: {1}".format(torch.min(edges), torch.max(edges)))
        return nml, edges

    def query(self, features, points, calibs, imgSizes = None, transforms=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''

        xyz = self.projection(torch.repeat_interleave(points, self.num_views, dim=0), calibs, transforms)

        xy = xyz[:, :2, :]
        # z = xyz[:, 2:3, :]

        point_local_feat_list = []

        # We do this for every image per batch
        for idx, im_feat in enumerate(features):
            # We need to adjust the UV coordinates now because each image has a different size
            currentXY = xy[idx::self.num_views]
            sizes = imgSizes[idx::self.num_views].unsqueeze(2)

            # pixel_local = self.index(self.images[idx], currentXY).cpu().detach().numpy()
            # img_local = np.swapaxes(self.images[idx][0].cpu().detach().numpy(),0,2)
            # new_img = np.zeros_like(img_local)
            # new_xy = currentXY.cpu().detach().numpy()[0] * np.reshape(new_img.shape[:2],(2,1))
            # new_xy = ((new_xy+1)/2).astype(np.uint32)
            # new_img[new_xy.T[:,0], new_xy.T[:,1]] = pixel_local[0].T

            # plt.imshow(new_img, cmap='gray')
            # plt.show()
            point_local_feat_list.append(self.index(im_feat, (currentXY * sizes).to(im_feat.dtype)))

        point_local_feat_list.append(points)
        multi = torch.cat(point_local_feat_list, dim=1)

        # Reshape for SIREN
        if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier' or self.opt.mlp_type == 'siren':
            multi = multi.permute(0, 2, 1)

        if multi.requires_grad:
            pred = checkpoint.checkpoint(self.surface_classifier, multi)
        else:
            pred = self.surface_classifier(multi)

        # SIREN Reshape
        if self.opt.mlp_type == 'mlp' or self.opt.mlp_type == 'mlp_fourier' or self.opt.mlp_type == 'siren':
            pred = pred.permute(0,2,1)

        if self.opt.predict_normal:
            pred = F.normalize(pred, dim=1, eps=1e-8)

        return pred

    def get_error(self, pred, labels, nmls=None, labels_nml=None, edges=None, labels_edges=None):
        error = {'Err(occ)': 0, 'Err(cmb)': 0, 'Err(nml)': 0, 'Err(edges)': 0}

        if self.opt.predict_normal:
            error['Err(nml)'] = (1 - self.criteria['nml'](pred, labels_nml).unsqueeze(0))
            error['Err(cmb)'] = error['Err(nml)']
        else:
            error['Err(occ)'] += self.criteria['occ'](pred, labels).unsqueeze(0)

            if self.opt.use_normal_loss and nmls is not None and labels_nml is not None:
                error['Err(nml)'] = (1 - self.criteria['nml'](nmls, labels_nml).unsqueeze(0))

            if self.opt.use_edge_loss and edges is not None and labels_edges is not None:
                error['Err(edges)'] = self.criteria['edges'](edges, labels_edges).unsqueeze(0)

            error['Err(cmb)'] = error['Err(occ)'] * self.opt.occ_loss_weight + error['Err(nml)'] *\
                                self.opt.normal_loss_weight + error['Err(edges)'] * self.opt.edge_loss_weight

        return error

    def forward(self, images, points, calibs, imgSizes=None, transforms=None, labels=None, points_surface=None,
                labels_nml=None, labels_edges=None):

        # Get image feature
        features = self.filter(images)

        # Phase 2: point query
        pred = self.query(features=features, points=points_surface if self.opt.predict_normal else points,
                          calibs=calibs, imgSizes=imgSizes, transforms=transforms)

        nmls = None
        edges = None

        if not self.opt.predict_normal and points_surface is not None:
            nmls, edges = self.calc_normal_edges(features, points_surface, calibs, imgSizes)

        # get the error
        error = self.get_error(pred, labels, nmls, labels_nml, edges, labels_edges)

        return pred, nmls, edges, error
