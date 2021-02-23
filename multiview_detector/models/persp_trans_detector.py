import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, mobilenet_v2
import kornia
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat

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


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18', z=0, reduction=None, bottleneck_dim=0, outfeat_dim=0):
        super().__init__()
        self.Rimg_shape, self.Rworld_shape = dataset.Rimg_shape, dataset.Rworld_shape
        self.coord_map = create_coord_map(self.Rworld_shape).to('cuda:0')

        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        imgcoord_from_Rimggrid_mat = np.diag([dataset.img_reduce, dataset.img_reduce, 1]) @ \
                                     dataset.base.img_xy_from_xy_mat
        # world grid change to xy indexing
        world_zoom_mat = np.diag([dataset.world_reduce, dataset.world_reduce, 1])
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(
            dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam],
                                                                          dataset.base.extrinsic_matrices[cam],
                                                                          z / dataset.base.worldcoord_unit)
                                         for cam in range(dataset.num_cam)]
        # Rworldgrid(xy)_from_Rimggrid(xy)
        self.proj_mats = torch.stack([torch.from_numpy(Rworldgrid_from_worldcoord_mat @
                                                       worldcoord_from_imgcoord_mats[cam] @
                                                       imgcoord_from_Rimggrid_mat)
                                      for cam in range(dataset.num_cam)]).to('cuda:0')

        self.reduction = reduction
        self.use_cuda1 = 'cuda:0' if self.reduction == 'sum' or torch.cuda.device_count() == 1 else 'cuda:1'
        if arch == 'vgg11':
            base = vgg11(pretrained=True).features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(pretrained=True,
                                                replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            out_channel = 512
        elif arch == 'mobilenet':
            base = mobilenet_v2(pretrained=True).features
            split = 12
            out_channel = 1280
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        self.base_pt1 = base[:split].to(self.use_cuda1)
        self.base_pt2 = base[split:].to('cuda:0')

        if bottleneck_dim:
            self.bottleneck = nn.Conv2d(out_channel, bottleneck_dim, 3, padding=1).to('cuda:0')
            out_channel = bottleneck_dim
        else:
            self.bottleneck = nn.Sequential()

        # img heads
        self.img_heatmap = output_head(out_channel, outfeat_dim, 1).to('cuda:0')
        self.img_offset = output_head(out_channel, outfeat_dim, 2).to('cuda:0')
        self.img_wh = output_head(out_channel, outfeat_dim, 2).to('cuda:0')

        # world feat
        if self.reduction is None:
            out_channel = out_channel * dataset.num_cam + 2
        elif self.reduction == 'sum':
            out_channel = out_channel + 2
        else:
            raise Exception
        self.world_feat = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(),
                                        nn.Conv2d(256, 128, 3, padding=4, dilation=4)).to('cuda:0')

        # world heads
        self.world_heatmap = output_head(128, outfeat_dim, 1).to('cuda:0')
        self.world_offset = output_head(128, outfeat_dim, 2).to('cuda:0')

        # init
        self.img_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.img_offset)
        fill_fc_weights(self.img_wh)
        self.world_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.world_offset)
        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        imgs_feat = self.base_pt1(imgs.to(self.use_cuda1))
        imgs_feat = self.base_pt2(imgs_feat.to('cuda:0'))
        imgs_feat = self.bottleneck(imgs_feat)
        imgs_feat = F.interpolate(imgs_feat, self.Rimg_shape, mode='bilinear')
        if visualize:
            for cam in range(N):
                plt.imshow(torch.norm(imgs_feat[cam * B].detach(), dim=0).cpu().numpy())
                plt.show()

        # img heads
        _, C, H, W = imgs_feat.shape
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)

        # world feat
        H, W = self.Rworld_shape
        world_feat = kornia.warp_perspective(imgs_feat, self.proj_mats.repeat(B, 1, 1, 1).view(B * N, 3, 3).float(),
                                             self.Rworld_shape).view(B, N, C, H, W)
        # world_feats = []
        # for i in range(B * N):
        #     proj_mat = torch.from_numpy(np.linalg.inv(proj_mats[i])).to('cuda:0')
        #     coeff = (proj_mat / proj_mat[2, 2]).view(-1)[:8]
        #     img_feat = torch.zeros([C, H, W], device=imgs_feat.device).to('cuda:0')
        #     img_feat[:, :self.Rimg_shape[0], :self.Rimg_shape[1]] = imgs_feat[i].to('cuda:0')
        #     world_feats.append(perspective(img_feat, coeff))
        # world_feat = torch.stack(world_feats)
        if visualize:
            for cam in range(N):
                plt.imshow(torch.norm(world_feat[0, cam].detach(), dim=0).cpu().numpy())
                plt.show()
        if self.reduction is None:
            world_feat = world_feat.view(B, N * C, H, W)
        elif self.reduction == 'sum':
            world_feat = world_feat.sum(dim=1)
        else:
            raise Exception
        world_feat = torch.cat([world_feat, self.coord_map.repeat([B, 1, 1, 1])], 1)
        world_feat = self.world_feat(world_feat.to('cuda:0'))

        # world heads
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)

        if visualize:
            plt.imshow(torch.norm(world_feat[0].detach(), dim=0).cpu().numpy())
            plt.show()
            plt.imshow(torch.norm(world_heatmap[0].detach(), dim=0).cpu().numpy())
            plt.show()
        return (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh)


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.utils.decode import ctdet_decode

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform, train=False)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, world_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset, arch='resnet18', reduction='sum')
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(imgs, visualize=False)
    xysc = ctdet_decode(world_heatmap, world_offset)
    pass


if __name__ == '__main__':
    test()
