import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.projection import get_imgcoord_from_worldcoord_mat, get_worldcoord_from_imgcoord_mat

import matplotlib.pyplot as plt


def create_coord_map(img_size, with_r=False):
    H, W = img_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
    grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
    ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    if with_r:
        grid_r = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
        ret = torch.cat([ret, grid_r], dim=1)
    return ret


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.Rgrid_shape = dataset.img_shape, dataset.Rgrid_shape
        self.coord_map = create_coord_map(self.Rgrid_shape)
        self.intrinsic_matrices, self.extrinsic_matrices, self.worldcoord_unit = \
            dataset.base.intrinsic_matrices, dataset.base.extrinsic_matrices, dataset.base.worldcoord_unit
        self.img_upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        self.imgcoord_from_Rimggrid_mat = np.diag([dataset.img_reduce, dataset.img_reduce, 1]) @ \
                                          dataset.base.img_xy_from_xy_mat
        # world grid change to xy indexing
        world_zoom_mat = np.diag([dataset.world_reduce, dataset.world_reduce, 1])
        self.Rworldgrid_from_worldcoord_mat = np.linalg.inv(
            dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        self.world_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                              nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                              nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        pass

    def forward(self, imgs, z=0.0, visualize=False):
        img_features, img_results = self.get_img_feats(imgs, visualize=visualize)
        world_result, world_features = self.get_world_res(img_features, z / self.worldcoord_unit, visualize=visualize)
        return world_result, img_results

    def get_img_feats(self, imgs, visualize=False):
        img_features, img_results = [], []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to('cuda:1'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, self.img_upsample_shape, mode='bilinear')
            img_features.append(img_feature)
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            img_results.append(img_res)
            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
        return img_features, img_results

    def get_world_res(self, img_features, z, visualize=False):
        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.intrinsic_matrices[cam],
                                                                          self.extrinsic_matrices[cam], z)
                                         for cam in range(self.num_cam)]
        # Rworldgrid(xy)_from_Rimggrid(xy)
        proj_mats = [torch.from_numpy(self.Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam] @
                                      self.imgcoord_from_Rimggrid_mat) for cam in range(self.num_cam)]
        B = img_features[0].shape[0]
        world_features = []
        for cam in range(self.num_cam):
            proj_mat = proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            world_feature = kornia.warp_perspective(img_features[cam].to('cuda:0'), proj_mat, self.Rgrid_shape)
            world_features.append(world_feature.to('cuda:0'))
            if visualize:
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        world_result = self.world_classifier(world_features.to('cuda:0'))
        if visualize:
            plt.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            plt.show()
            plt.imshow(torch.norm(world_result[0].detach(), dim=0).cpu().numpy())
            plt.show()
        return world_result, world_features


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, world_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    img_feats, img_results = model.get_img_feats(imgs, visualize=False)
    world_result0, world_feats0 = model.get_world_res(img_feats, z=0, visualize=True)
    # world_result1, world_feats1 = model.get_world_res(img_feats, z=1.8, visualize=True)
    pass


if __name__ == '__main__':
    test()
