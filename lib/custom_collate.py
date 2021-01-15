from .mesh_util import *
from .sample_util import *
from .geometry import *
from PIL import Image

class MultiViewCollator(object):
    def __init__(self, opt):
        self.opt = opt
    def __call__(self, batches):
        return self.prepareBatches(batches, self.opt)

    def make_divisible(self, x, div):
        return int(((x // div) + 1) * div)

    def adjustImageSizesInBatch(self, batches, div = 4):
        numBatches = len(batches)
        numImages = len(batches[0]['img'])

        img_tensor_list = []

        for i in range(numImages):
            maxWidth = 0
            maxHeight = 0

            for j in range(numBatches):
                img = batches[j]['img'][i]
                width = img.shape[2]
                height = img.shape[1]
                maxHeight = max(height, maxHeight)
                maxWidth = max(width, maxWidth)

            diffWidth = 0
            diffHeight = 0
            newWidth = maxWidth
            newHeight = maxHeight

            if newWidth % div != 0:
                newWidth = self.make_divisible(newWidth, div)
                diffWidth = (newWidth - maxWidth) // 2

            if newHeight % div != 0:
                newHeight = self.make_divisible(newHeight, div)
                diffHeight = (newHeight - maxHeight) // 2

            list_img = []
            for j in range(numBatches):
                img = batches[j]['img'][i]
                width = img.shape[2]
                height = img.shape[1]

                channels = batches[j]['img'][i].shape[0]
                if channels == 3:
                    newImg = np.zeros((channels, newHeight, newWidth), np.float32)
                else:
                    newImg = np.ones((channels, newHeight, newWidth), np.float32)

                newImg[:, diffHeight:height + diffHeight, diffWidth:width + diffWidth] = img

                list_img.append(newImg)

            tensor = torch.Tensor(np.stack(list_img, axis=0))
            img_tensor_list.append(tensor)

        return img_tensor_list

    def prepareBatches(self, batches, opt):
        image_tensor_list = []
        sizes = []

        #Pad images to be the same across batches and be divisible by factor x so convolution doesnt fail
        for img in self.adjustImageSizesInBatch(batches):
            image_tensor_list.append(img)
            size = torch.Tensor([opt.loadSize / img.shape[3], opt.loadSize / img.shape[2]])
            sizes.append(size)

        train_data = {'bounding_boxes':[], 'images':image_tensor_list, 'calib': [], 'samples': [], 'labels': [], 'samples_normals': [], 'size': [], 'normals': []}

        for batch in batches:
            train_data['bounding_boxes'].append({'b_min':batch['b_min'],'b_max':batch['b_max']})
            train_data['calib'].append(batch['calib'].unsqueeze(0))

            if batch['samples'] is not None:
                train_data['samples'].append(batch['samples'].unsqueeze(0))
            if batch['labels'] is not None:
                train_data['labels'].append(batch['labels'].unsqueeze(0))

            if batch['samples_normals'] != None:
                train_data['samples_normals'].append(batch['samples_normals'].unsqueeze(0))
            else:
                train_data['samples_normals'] = None

            if batch['normals'] != None:
                train_data['normals'].append(batch['normals'].unsqueeze(0))
            else:
                train_data['normals'] = None

        train_data['calib'] = torch.cat(train_data['calib'], dim=0)
        train_data['size'] = torch.stack(sizes, dim=0).repeat(len(train_data['calib']), 1)

        if train_data['samples'] != []:
            train_data['samples'] = torch.cat(train_data['samples'], dim=0)

        if train_data['labels'] != []:
            train_data['labels'] = torch.cat(train_data['labels'], dim=0)

        if train_data['samples_normals'] != None:
            train_data['samples_normals'] = torch.cat(train_data['samples_normals'], dim=0)

            if opt.num_views > 1:
               train_data['samples_normals'] = reshape_sample_tensor(train_data['samples_normals'], opt.num_views)

        if train_data['normals'] != None:
            train_data['normals'] = torch.cat(train_data['normals'], dim=0)

        train_data['calib'] = reshape_multiview_calib_tensor(train_data['calib'])

        if opt.num_views > 1 and train_data['samples'] != []:
            train_data['samples'] = reshape_sample_tensor(train_data['samples'], opt.num_views)

        return train_data

def move_to_gpu(train_data, cuda):
    for key in train_data:
        if key == 'images':
            for idx, img in enumerate(train_data[key]):
                train_data[key][idx] = img.to(device=cuda)
        elif key == 'bounding_boxes':
            pass
        elif train_data[key] is not None and len(train_data[key]) > 0:
            train_data[key] = train_data[key].to(device=cuda)

    return train_data

def reshape_multiview_calib_tensor(calib_tensor):
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return calib_tensor

def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor

def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor