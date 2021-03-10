import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, mobilenet_v2
import kornia
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat
from multiview_detector.models.conv_world_feat import ConvWorldFeat, DeformConvWorldFeat
from multiview_detector.models.trans_world_feat import TransformerWorldFeat, DeformTransWorldFeat

import matplotlib.pyplot as plt


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


class MVDeTr(nn.Module):
    def __init__(self, dataset, arch='resnet18', z=0, world_feat_arch='conv', reduction=None,
                 bottleneck_dim=128, hidden_dim=128, outfeat_dim=0, droupout=0.5):
        super().__init__()
        self.Rimg_shape, self.Rworld_shape = dataset.Rimg_shape, dataset.Rworld_shape

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

        self.use_cuda1 = 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1'
        if arch == 'vgg11':
            base = vgg11(pretrained=True).features
            base[-1] = nn.Identity()
            base[-4] = nn.Identity()
            base_dim = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(pretrained=True,
                                                replace_stride_with_dilation=[False, True, True]).children())[:-2])
            base_dim = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        self.base = base.to(self.use_cuda1)

        if bottleneck_dim:
            self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 1),
                                            nn.Dropout2d(droupout)).to(self.use_cuda1)
            base_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1).to('cuda:0')
        self.img_offset = output_head(base_dim, outfeat_dim, 2).to('cuda:0')
        self.img_wh = output_head(base_dim, outfeat_dim, 2).to('cuda:0')

        # world feat
        if world_feat_arch == 'conv':
            self.world_feat = ConvWorldFeat(dataset.num_cam, dataset.Rworld_shape,
                                            base_dim, hidden_dim, reduction).to('cuda:0')
        elif world_feat_arch == 'trans':
            self.world_feat = TransformerWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim).to('cuda:0')
        elif world_feat_arch == 'deform_conv':
            self.world_feat = DeformConvWorldFeat(dataset.num_cam, dataset.Rworld_shape,
                                                  base_dim, hidden_dim).to('cuda:0')
        elif world_feat_arch == 'deform_trans':
            self.world_feat = DeformTransWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim).to('cuda:0')
        else:
            raise Exception

        # world heads
        self.world_heatmap = output_head(hidden_dim, outfeat_dim, 1).to('cuda:0')
        self.world_offset = output_head(hidden_dim, outfeat_dim, 2).to('cuda:0')

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
        imgs_feat = self.base(imgs.to(self.use_cuda1))
        imgs_feat = self.bottleneck(imgs_feat).to('cuda:0')
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
                                             self.Rworld_shape, align_corners=False).view(B, N, C, H, W)
        if visualize:
            for cam in range(N):
                plt.imshow(torch.norm(world_feat[0, cam].detach(), dim=0).cpu().numpy())
                plt.show()
        world_feat = self.world_feat(world_feat)

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

    transform = T.Compose([T.Resize([540, 960]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform, train=False,
                           img_reduce=16)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, world_gt, imgs_gt, frame = next(iter(dataloader))
    model = MVDeTr(dataset, arch='resnet18', world_feat_arch='trans')
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(imgs, visualize=True)
    xysc = ctdet_decode(world_heatmap, world_offset)
    pass


if __name__ == '__main__':
    test()
