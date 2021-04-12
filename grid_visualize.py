import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import torch
import kornia
import torchvision.transforms as T
from multiview_detector.utils import projection
from multiview_detector.datasets import *

if __name__ == '__main__':
    dataset = frameDataset(Wildtrack('/home/houyz/Data/Wildtrack'))
    multi = 10
    freq = 40
    for cam in range(dataset.num_cam):
        img = Image.open(f'/home/houyz/Data/Wildtrack/Image_subsets/C{cam + 1}/00000025.png')
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
        world_img = Image.fromarray(world_img)
        draw = ImageDraw.Draw(world_img)
        draw.line([(480, 160), (480, 280), (600, 280), (600, 160), (480, 160)], fill=(255, 192, 0), width=5)
        draw.line([(680, 80), (680, 200), (800, 200), (800, 80), (680, 80)], fill=(0, 176, 80), width=5)
        world_mask_conv1, world_mask_conv2 = Image.new('1', dataset.worldgrid_shape[::-1]), \
                                             Image.new('1', dataset.worldgrid_shape[::-1])
        draw = ImageDraw.Draw(world_mask_conv1)
        draw.line([(480, 160), (480, 280), (600, 280), (600, 160), (480, 160)], fill=1, width=5)
        draw = ImageDraw.Draw(world_mask_conv2)
        draw.line([(680, 80), (680, 200), (800, 200), (800, 80), (680, 80)], fill=1, width=5)
        world_mask_conv1, world_mask_conv2 = T.ToTensor()(world_mask_conv1).unsqueeze(0), \
                                             T.ToTensor()(world_mask_conv2).unsqueeze(0)

        # overlay = Image.new('RGBA', world_img.size, (0,0,0,0,))
        # draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        # draw.rectangle(((llx, lly), (urx, ury)), fill=TINT_COLOR + (OPACITY,))
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
        img_mask_conv1 = kornia.warp_perspective(world_mask_conv1, dataset.img_from_world[[cam]], dataset.img_shape,
                                                 align_corners=False)[0, 0].bool().numpy()
        img_mask_conv2 = kornia.warp_perspective(world_mask_conv2, dataset.img_from_world[[cam]], dataset.img_shape,
                                                 align_corners=False)[0, 0].bool().numpy()
        img[img_mask_conv1] = [255, 192, 0]
        img[img_mask_conv2] = [0, 176, 80]
        img = Image.fromarray(img)
        img.save(f'imgs/img_grid_visualize_C{cam + 1}.png')
        plt.imshow(img)
        plt.show()

    pass
