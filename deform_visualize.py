import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import torch
import kornia
import torchvision.transforms as T
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils import projection
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.datasets import *

if __name__ == '__main__':
    multi = 10
    freq = 40
    dataset = frameDataset(Wildtrack('/home/houyz/Data/Wildtrack'), train=False)
    imgs, world_gt, imgs_gt, affine_mats, frame = dataset.__getitem__(0)
    world_gt_location = torch.zeros(np.prod(dataset.Rworld_shape))
    world_gt_location[world_gt['idx'][world_gt['reg_mask']]] = 1
    world_gt_location = (F.interpolate(world_gt_location.view([1, 1, ] + dataset.Rworld_shape), scale_factor=0.5,
                                       mode='bilinear') > 0).squeeze()
    plt.imshow(world_gt_location)
    plt.show()

    model = MVDeTr(dataset, world_feat_arch='deform_trans').cuda()
    model.load_state_dict(torch.load('logs/wildtrack/augFCS_deform_trans_lr0.001_baseR0.1_neck128_out64_'
                                     'alpha1.0_id0_drop0.5_dropcam0.0_worldRK4_10_imgRK12_10_2021-04-09_22-39-28/'
                                     'MultiviewDetector.pth'))
    # add breakpoint to L112@'multiview_detector/models/ops/modules/ms_deform_attn.py',
    # run 'torch.save(sampling_locations.detach().cpu(),'sampling_locations1')' to save the sampling locations,
    # and run 'torch.save(attention_weights.detach().cpu(),'attention_weights1')' to save the attention weights
    if 0:
        (world_heatmap, _), (_, _, _) = model(imgs.unsqueeze(0).cuda(), affine_mats.unsqueeze(0))
        plt.imshow(world_heatmap.squeeze().detach().cpu().numpy())
        plt.show()
    # N, Len_q, n_heads, n_levels, n_points, 2
    sampling_locations1 = torch.load('sampling_locations1')
    attention_weights1 = torch.load('attention_weights1')
    # len_q = n*h*w
    world_shape = (np.array(dataset.Rworld_shape) // 2).tolist()
    # ij indexing for gt locations in range [60,180]
    # [ 33,  62] -> 33*180+62=6002
    # [ 39,  101] -> 39*180+101=7121
    denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    pil_transform = T.Compose([T.Resize(dataset.img_shape), T.ToPILImage()])
    for cam in range(dataset.num_cam):
        img = pil_transform(denorm(imgs)[cam])
        world_img = T.ToTensor()(img).unsqueeze(0)
        img_mask = torch.ones_like(world_img)
        world_img = kornia.warp_perspective(world_img, dataset.world_from_img[[cam]], dataset.worldgrid_shape,
                                            align_corners=False)[0]
        world_mask = kornia.warp_perspective(img_mask, dataset.world_from_img[[cam]], dataset.worldgrid_shape,
                                             align_corners=False)[0, 0].bool().numpy()
        world_img = np.array(T.ToPILImage()(world_img))
        world_mask_grid = np.zeros_like(world_mask, dtype=bool)
        world_mask_grid[:, ::freq] = 1
        world_mask_grid[::freq, :] = 1
        world_mask = world_mask * world_mask_grid
        world_img[world_mask] = [0, 0, 255]

        # xy indexing -> ij indexing
        attend_points1 = (sampling_locations1[0, 6002, :, cam, :, [1, 0]].reshape([-1, 2]).clip(0, 1 - 1e-3) *
                          torch.tensor(world_shape)).int().long()
        attend_points2 = (sampling_locations1[0, 7121, :, cam, :, [1, 0]].reshape([-1, 2]).clip(0, 1 - 1e-3) *
                          torch.tensor(world_shape)).int().long()
        weight1 = attention_weights1[0, 6002, :, cam].reshape([-1])
        weight1 = (weight1 / weight1.max()) ** 0.2
        weight2 = attention_weights1[0, 7121, :, cam].reshape([-1])
        weight2 = (weight2 / weight2.max()) ** 0.2
        world_mask_points1 = torch.zeros(world_shape)
        world_mask_points2 = torch.zeros(world_shape)
        world_mask_points_og = torch.zeros(world_shape)
        world_mask_points1[attend_points1[:, 0], attend_points1[:, 1]] = weight1
        world_mask_points2[attend_points2[:, 0], attend_points2[:, 1]] = weight2
        world_mask_points_og[33, 62] = 1
        world_mask_points_og[39, 101] = 1

        world_mask_points1 = F.interpolate(world_mask_points1.view([1, 1] + world_shape),
                                           dataset.worldgrid_shape)
        world_mask_points2 = F.interpolate(world_mask_points2.view([1, 1] + world_shape),
                                           dataset.worldgrid_shape)
        world_mask_points_og = F.interpolate(world_mask_points_og.view([1, 1] + world_shape),
                                             dataset.worldgrid_shape)
        idx = world_mask_points1.squeeze().bool()
        world_img[idx] = torch.tensor([[255, 192, 0]]) * world_mask_points1.squeeze()[idx].view([-1, 1]) + \
                         world_img[idx] * (1 - world_mask_points1.squeeze()[idx].view([-1, 1]).numpy())
        idx = world_mask_points2.squeeze().bool()
        world_img[idx] = torch.tensor([[0, 176, 80]]) * world_mask_points2.squeeze()[idx].view([-1, 1]) + \
                         world_img[idx] * (1 - world_mask_points2.squeeze()[idx].view([-1, 1]).numpy())
        world_img[world_mask_points_og.squeeze().bool()] = [255, 0, 0]

        world_img = Image.fromarray(world_img)
        world_img.save(f'imgs/world_grid_visualize_C{cam + 1}.png')
        plt.imshow(world_img)
        plt.show()

        world_grid = np.zeros(np.array(dataset.worldgrid_shape) * multi + np.array([1, 1]))
        world_grid[:, ::freq * multi] = 1
        world_grid[::freq * multi, :] = 1
        world_grid = np.array(np.where(world_grid)) / multi
        world_coord = dataset.base.get_worldcoord_from_worldgrid(world_grid)
        img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.base.intrinsic_matrices[cam],
                                                              dataset.base.extrinsic_matrices[cam])
        img_coord = (img_coord).astype(int)
        img_coord = img_coord[:, np.where((img_coord[0] > 0) & (img_coord[1] > 0) &
                                          (img_coord[0] < 1920) & (img_coord[1] < 1080))[0]]
        img = np.array(img)
        img[img_coord[1], img_coord[0]] = [0, 0, 255]
        img_mask_points1 = kornia.warp_perspective(world_mask_points1, dataset.img_from_world[[cam]],
                                                   dataset.img_shape, align_corners=False).squeeze()
        img_mask_points2 = kornia.warp_perspective(world_mask_points2, dataset.img_from_world[[cam]],
                                                   dataset.img_shape, align_corners=False).squeeze()
        img_mask_points_og = kornia.warp_perspective(world_mask_points_og, dataset.img_from_world[[cam]],
                                                     dataset.img_shape, align_corners=False).squeeze()

        idx = img_mask_points1.bool()
        img[idx] = torch.tensor([[255, 192, 0]]) * img_mask_points1[idx].view([-1, 1]) + \
                   img[idx] * (1 - img_mask_points1[idx].view([-1, 1]).numpy())
        idx = img_mask_points2.bool()
        img[idx] = torch.tensor([[0, 176, 80]]) * img_mask_points2[idx].view([-1, 1]) + \
                   img[idx] * (1 - img_mask_points2[idx].view([-1, 1]).numpy())
        img[img_mask_points_og.bool()] = [255, 0, 0]
        img = Image.fromarray(img)
        img.save(f'imgs/img_grid_visualize_C{cam + 1}.png')
        plt.imshow(img)
        plt.show()

    pass
