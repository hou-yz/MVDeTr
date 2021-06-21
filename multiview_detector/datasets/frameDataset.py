import os
import json
import time
from operator import itemgetter
import copy

import numpy as np
from PIL import Image
import kornia
from torchvision.datasets import VisionDataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from multiview_detector.utils.projection import *
from multiview_detector.utils.image_utils import draw_umich_gaussian, random_affine
import matplotlib.pyplot as plt


def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=np.bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, reID=False, world_reduce=4, img_reduce=12,
                 world_kernel_size=10, img_kernel_size=10,
                 train_ratio=0.9, top_k=100, force_download=True,
                 semi_supervised=0.0, dropout=0.0, augmentation=False):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        # reduce = input/output
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.semi_supervised = semi_supervised * train
        self.dropout = dropout
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])
        self.augmentation = augmentation

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.world_from_img, self.img_from_world = self.get_world_imgs_trans()
        world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
        self.imgs_region = kornia.warp_perspective(world_masks, self.img_from_world, self.img_shape, 'nearest',
                                                   align_corners=False)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.keeps = {}
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        num_keep, num_all = 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                keep = np.mean(np.array(frame_range) < frame) < self.semi_supervised if self.semi_supervised else 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                if keep:
                    for pedestrian in all_pedestrians:
                        grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                        if pedestrian['personID'] not in self.pid_dict:
                            self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                        num_all += 1
                        num_keep += keep
                        num_world_bbox += keep
                        if self.base.indexing == 'xy':
                            world_pts.append((grid_x, grid_y))
                        else:
                            world_pts.append((grid_y, grid_x))
                        world_pids.append(self.pid_dict[pedestrian['personID']])
                        for cam in range(self.num_cam):
                            if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                                img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                      (pedestrian['views'][cam]))
                                img_pids[cam].append(self.pid_dict[pedestrian['personID']])
                                num_imgs_bbox += 1
                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))
                self.keeps[frame] = keep

        print(f'all: pid: {len(self.pid_dict)}, frame: {num_frame}, keep ratio: {num_keep / num_all:.3f}\n'
              f'recorded: world bbox: {num_world_bbox / num_frame:.1f}, '
              f'imgs bbox per cam: {num_imgs_bbox / num_frame / self.num_cam:.1f}')
        # gt in mot format for evaluation
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        pass

    def get_world_imgs_trans(self, z=0):
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        # world grid change to xy indexing
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(self.base.worldcoord_from_worldgrid_mat @
                                                       self.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                          self.base.extrinsic_matrices[cam],
                                                                          z / self.base.worldcoord_unit)
                                         for cam in range(self.num_cam)]
        # worldgrid(xy)_from_img(xy)
        proj_mats = [Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam] @ self.base.img_xy_from_xy_mat
                     for cam in range(self.num_cam)]
        world_from_img = torch.tensor(np.stack(proj_mats))
        # img(xy)_from_worldgrid(xy)
        img_from_world = torch.tensor(np.stack([np.linalg.inv(proj_mat) for proj_mat in proj_mats]))
        return world_from_img.float(), img_from_world.float()

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID']).squeeze()
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def __getitem__(self, index, visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for i in range(len(img_x_s)):
                x, y = img_x_s[i], img_y_s[i]
                if x > 0 and y > 0:
                    ax.add_patch(Circle((x, y), 10))
            plt.show()
            img0 = img.copy()
            for bbox in img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        frame = list(self.world_gt.keys())[index]
        # imgs
        imgs, imgs_gt, affine_mats, masks = [], [], [], []
        for cam in range(self.num_cam):
            img = np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
            img_bboxs, img_pids = self.imgs_gt[frame][cam]
            if self.augmentation:
                img, img_bboxs, img_pids, M = random_affine(img, img_bboxs, img_pids)
            else:
                M = np.eye(3)
            imgs.append(self.transform(img))
            affine_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (img_bboxs[:, 0] + img_bboxs[:, 2]) / 2, img_bboxs[:, 3]
            img_w_s, img_h_s = (img_bboxs[:, 2] - img_bboxs[:, 0]), (img_bboxs[:, 3] - img_bboxs[:, 1])

            img_gt = get_gt(self.Rimg_shape, img_x_s, img_y_s, img_w_s, img_h_s, v_s=img_pids,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            imgs_gt.append(img_gt)
            if visualize:
                plt_visualize()

        imgs = torch.stack(imgs)
        affine_mats = torch.stack(affine_mats)
        # inverse_M = torch.inverse(
        #     torch.cat([affine_mats, torch.tensor([0, 0, 1]).view(1, 1, 3).repeat(self.num_cam, 1, 1)], dim=1))[:, :2]
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt]) for key in imgs_gt[0]}
        # imgs_gt['heatmap_mask'] = self.imgs_region if self.keeps[frame] else torch.zeros_like(self.imgs_region)
        # imgs_gt['heatmap_mask'] = kornia.warp_perspective(imgs_gt['heatmap_mask'], affine_mats, self.img_shape,
        #                                                   align_corners=False)
        # imgs_gt['heatmap_mask'] = F.interpolate(imgs_gt['heatmap_mask'], self.Rimg_shape, mode='bilinear',
        #                                         align_corners=False).bool().float()
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            drop_cam = np.random.randint(0, self.num_cam)
            keep_cams[drop_cam] = 0
            for key in imgs_gt:
                imgs_gt[key][drop_cam] = 0
        # world gt
        world_pt_s, world_pid_s = self.world_gt[frame]
        world_gt = get_gt(self.Rworld_shape, world_pt_s[:, 0], world_pt_s[:, 1], v_s=world_pid_s,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        return imgs, world_gt, imgs_gt, affine_mats, frame

    def __len__(self):
        return len(self.world_gt.keys())


def test(test_projection=False):
    from torch.utils.data import DataLoader
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=True, augmentation=False)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), train=True)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=True, semi_supervised=.1)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), train=True, semi_supervised=.1)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=True, semi_supervised=0.5)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), train=True, semi_supervised=0.5)
    min_dist = np.inf
    for world_gt in dataset.world_gt.values():
        x, y = world_gt[0][:, 0], world_gt[0][:, 1]
        if x.size and y.size:
            xy_dists = ((x - x[:, None]) ** 2 + (y - y[:, None]) ** 2) ** 0.5
            np.fill_diagonal(xy_dists, np.inf)
            min_dist = min(min_dist, np.min(xy_dists))
            pass
    dataloader = DataLoader(dataset, 2, True, num_workers=0)
    # imgs, world_gt, imgs_gt, M, frame = next(iter(dataloader))
    t0 = time.time()
    imgs, world_gt, imgs_gt, M, frame = dataset.__getitem__(0, visualize=False)
    print(time.time() - t0)

    pass
    if test_projection:
        import matplotlib.pyplot as plt
        from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
        world_grid_maps = []
        xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
        H, W = xx.shape
        image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
        for cam in range(dataset.num_cam):
            world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(),
                                                          dataset.base.intrinsic_matrices[cam],
                                                          dataset.base.extrinsic_matrices[cam])
            world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
            world_grid_map = np.zeros(dataset.worldgrid_shape)
            for i in range(H):
                for j in range(W):
                    x, y = world_grids[i, j]
                    if dataset.base.indexing == 'xy':
                        if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                            world_grid_map[int(y), int(x)] += 1
                    else:
                        if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
                            world_grid_map[int(x), int(y)] += 1
            world_grid_map = world_grid_map != 0
            plt.imshow(world_grid_map)
            plt.show()
            world_grid_maps.append(world_grid_map)
            pass
        plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
        plt.show()
        pass


if __name__ == '__main__':
    test(True)
