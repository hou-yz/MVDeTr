import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg11
import torchvision.transforms as T
import kornia
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap
from multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points
from multiview_detector.models.conv_world_feat import ConvWorldFeat, DeformConvWorldFeat
from multiview_detector.models.trans_world_feat import TransformerWorldFeat, DeformTransWorldFeat, \
    DeformTransWorldFeat_aio
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


def create_reference_map(dataset, n_points=4, downsample=2, visualize=False):
    H, W = dataset.Rworld_shape  # H,W; N_row,N_col
    H, W = H // downsample, W // downsample
    ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32),
                                  torch.linspace(0.5, W - 0.5, W, dtype=torch.float32))
    ref = torch.stack((ref_x, ref_y), -1).reshape([-1, 2])
    if n_points == 4:
        zs = [0, 0, 0, 0]
    elif n_points == 8:
        zs = [-0.4, -0.2, 0, 0, 0.2, 0.4, 1, 1.8]
    else:
        raise Exception
    ref_maps = torch.zeros([H * W, dataset.num_cam, n_points, 2])
    world_zoom_mat = np.diag([dataset.world_reduce * downsample, dataset.world_reduce * downsample, 1])
    Rworldgrid_from_worldcoord_mat = np.linalg.inv(
        dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)
    for cam in range(dataset.num_cam):
        mat_0 = Rworldgrid_from_worldcoord_mat @ get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam],
                                                                                  dataset.base.extrinsic_matrices[cam])
        for i, z in enumerate(zs):
            mat_z = Rworldgrid_from_worldcoord_mat @ get_worldcoord_from_imgcoord_mat(
                dataset.base.intrinsic_matrices[cam],
                dataset.base.extrinsic_matrices[cam],
                z / dataset.base.worldcoord_unit)
            img_pts = project_2d_points(np.linalg.inv(mat_z), ref)
            ref_maps[:, cam, i, :] = torch.from_numpy(project_2d_points(mat_0, img_pts))
        pass
        if visualize:
            fig, ax = plt.subplots()
            field_x = (ref_maps[:, cam, 3, 0] - ref_maps[:, cam, 1, 0]).reshape([H, W])
            field_y = (ref_maps[:, cam, 3, 1] - ref_maps[:, cam, 1, 1]).reshape([H, W])
            ax.streamplot(ref_x.numpy(), ref_y.numpy(), field_x.numpy(), field_y.numpy())
            ax.set_aspect('equal', 'box')
            ax.invert_yaxis()
            plt.show()

    ref_maps[:, :, :, 0] /= W
    ref_maps[:, :, :, 1] /= H
    return ref_maps


class MVDeTr(nn.Module):
    def __init__(self, dataset, arch='resnet18', z=0, world_feat_arch='conv',
                 bottleneck_dim=128, outfeat_dim=64, droupout=0.5):
        super().__init__()
        self.Rimg_shape, self.Rworld_shape = dataset.Rimg_shape, dataset.Rworld_shape
        self.img_reduce = dataset.img_reduce

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
        # Rworldgrid(xy)_from_imgcoord(xy)
        self.proj_mats = torch.stack([torch.from_numpy(Rworldgrid_from_worldcoord_mat @
                                                       worldcoord_from_imgcoord_mats[cam])
                                      for cam in range(dataset.num_cam)])

        if arch == 'vgg11':
            self.base = vgg11(pretrained=True).features
            self.base[-1] = nn.Identity()
            self.base[-4] = nn.Identity()
            base_dim = 512
        elif arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
            base_dim = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        if bottleneck_dim:
            self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 1), nn.Dropout2d(droupout))
            base_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.img_offset = output_head(base_dim, outfeat_dim, 2)
        self.img_wh = output_head(base_dim, outfeat_dim, 2)
        # self.img_id = output_head(base_dim, outfeat_dim, len(dataset.pid_dict))

        # world feat
        if world_feat_arch == 'conv':
            self.world_feat = ConvWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim)
        elif world_feat_arch == 'trans':
            self.world_feat = TransformerWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim)
        elif world_feat_arch == 'deform_conv':
            self.world_feat = DeformConvWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim)
        elif world_feat_arch == 'deform_trans':
            n_points = 4
            reference_points = create_reference_map(dataset, n_points).repeat([dataset.num_cam, 1, 1, 1])
            self.world_feat = DeformTransWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim,
                                                   n_points=n_points, stride=2, reference_points=reference_points)
        elif world_feat_arch == 'aio':
            self.world_feat = DeformTransWorldFeat_aio(dataset.num_cam, dataset.Rworld_shape, base_dim)
        else:
            raise Exception

        # world heads
        self.world_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.world_offset = output_head(base_dim, outfeat_dim, 2)
        # self.world_id = output_head(base_dim, outfeat_dim, len(dataset.pid_dict))

        # init
        self.img_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.img_offset)
        fill_fc_weights(self.img_wh)
        self.world_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.world_offset)
        pass

    def forward(self, imgs, M, visualize=False):
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)

        inverse_affine_mats = torch.inverse(M.view([B * N, 3, 3]))
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        imgcoord_from_Rimggrid_mat = inverse_affine_mats @ \
                                     torch.from_numpy(np.diag([self.img_reduce, self.img_reduce, 1])
                                                      ).view(1, 3, 3).repeat(B * N, 1, 1).float()
        # Rworldgrid(xy)_from_Rimggrid(xy)
        proj_mats = self.proj_mats.repeat(B, 1, 1, 1).view(B * N, 3, 3).float() @ imgcoord_from_Rimggrid_mat

        if visualize:
            denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            proj_imgs = kornia.warp_perspective(T.Resize(self.Rimg_shape)(imgs), proj_mats.to(imgs.device),
                                                self.Rworld_shape, align_corners=False). \
                view(B, N, 3, self.Rworld_shape[0], self.Rworld_shape[1])
            for cam in range(N):
                visualize_img = T.ToPILImage()(denorm(imgs.detach())[cam * B])
                visualize_img.save(f'../../imgs/augimg{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()
                visualize_img = T.ToPILImage()(denorm(proj_imgs.detach())[0, cam])
                plt.imshow(visualize_img)
                plt.show()

        imgs_feat = self.base(imgs)
        imgs_feat = self.bottleneck(imgs_feat)
        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(imgs_feat[cam * B].detach(), dim=0).cpu())
                visualize_img.save(f'../../imgs/augimgfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()

        # img heads
        _, C, H, W = imgs_feat.shape
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)

        # world feat
        H, W = self.Rworld_shape
        world_feat = kornia.warp_perspective(imgs_feat, proj_mats.to(imgs.device),
                                             self.Rworld_shape, align_corners=False).view(B, N, C, H, W)
        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
                visualize_img.save(f'../../imgs/projfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()
        world_feat = self.world_feat(world_feat, visualize=visualize)

        # world heads
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)
        # world_id = self.world_id(world_feat)

        if visualize:
            visualize_img = array2heatmap(torch.norm(world_feat[0].detach(), dim=0).cpu())
            visualize_img.save(f'../../imgs/worldfeatall.png')
            plt.imshow(visualize_img)
            plt.show()
            visualize_img = array2heatmap(torch.sigmoid(world_heatmap.detach())[0, 0].cpu())
            visualize_img.save(f'../../imgs/worldres.png')
            plt.imshow(visualize_img)
            plt.show()
        return (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh)


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.utils.decode import ctdet_decode

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=False, augmentation=False)
    create_reference_map(dataset, 4)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    model = MVDeTr(dataset, world_feat_arch='deform_trans').cuda()
    # model.load_state_dict(torch.load(
    #     '../../logs/wildtrack/augFCS_deform_trans_lr0.001_baseR0.1_neck128_out64_alpha1.0_id0_drop0.5_dropcam0.0_worldRK4_10_imgRK12_10_2021-04-09_22-39-28/MultiviewDetector.pth'))
    imgs, world_gt, imgs_gt, affine_mats, frame = next(iter(dataloader))
    imgs = imgs.cuda()
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(imgs, affine_mats)
    xysc = ctdet_decode(world_heatmap, world_offset)
    pass


if __name__ == '__main__':
    test()
