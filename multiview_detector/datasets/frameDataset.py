import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *
from multiview_detector.utils.gaussian import draw_umich_gaussian
import matplotlib.pyplot as plt


def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=np.uint8)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
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
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=None, reID=False,
                 world_reduce=4, img_reduce=12, world_kernel_size=10, img_kernel_size=10,
                 train_ratio=0.9, top_k=100, force_download=True):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = list(map(lambda x: x // self.img_reduce, self.img_shape))

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_x_s, world_y_s, world_pid_s = [], [], []
                img_x_s, img_y_s = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                img_w_s, img_h_s = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                img_pid_s = [[] for _ in range(self.num_cam)]
                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if self.base.indexing == 'xy':
                        world_x_s.append(grid_x)
                        world_y_s.append(grid_y)
                    else:
                        world_x_s.append(grid_y)
                        world_y_s.append(grid_x)
                    world_pid_s.append(pedestrian['personID'] + 1)
                    for cam in range(self.num_cam):
                        x_foot = max(min((pedestrian['views'][cam]['xmin'] + pedestrian['views'][cam]['xmax']) // 2,
                                         self.img_shape[1] - 1), 0)
                        y_foot = min(pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        height = max(pedestrian['views'][cam]['ymax'] - pedestrian['views'][cam]['ymin'], 0)
                        width = max(pedestrian['views'][cam]['xmax'] - pedestrian['views'][cam]['xmin'], 0)
                        if pedestrian['views'][cam]['xmax'] > 0 and pedestrian['views'][cam]['ymax'] > 0:
                            # and pedestrian['views'][cam]['xmin'] < self.img_shape[1] and \
                            #     pedestrian['views'][cam]['ymin'] < self.img_shape[0]:
                            img_x_s[cam].append(x_foot)
                            img_y_s[cam].append(y_foot)
                            img_w_s[cam].append(width)
                            img_h_s[cam].append(height)
                            img_pid_s[cam].append(pedestrian['personID'] + 1)
                self.world_gt[frame] = (world_x_s, world_y_s, world_pid_s)
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # xywh
                    self.imgs_gt[frame][cam] = (img_x_s[cam], img_y_s[cam], img_w_s[cam], img_h_s[cam], img_pid_s[cam])

        # gt in mot format for evaluation
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        pass

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

    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        world_x_s, world_y_s, world_pid_s = self.world_gt[frame]
        world_gt = get_gt(self.Rworld_shape, world_x_s, world_y_s, v_s=world_pid_s,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        imgs_gt = {'heatmap': [], 'reg_mask': [], 'idx': [], 'pid': [], 'offset': [], 'wh': []}
        for cam in range(self.num_cam):
            img_x_s, img_y_s, img_w_s, img_h_s, img_pid_s = self.imgs_gt[frame][cam]
            img_gt = get_gt(self.Rimg_shape, img_x_s, img_y_s, img_w_s, img_h_s, v_s=img_pid_s,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            for key in img_gt.keys():
                imgs_gt[key].append(img_gt[key])
        for key in imgs_gt.keys():
            imgs_gt[key] = torch.stack(imgs_gt[key])
        return imgs, world_gt, imgs_gt, frame

    def __len__(self):
        return len(self.world_gt.keys())


def test(test_proection=False):
    from torch.utils.data import DataLoader
    from multiview_detector.datasets.Wildtrack import Wildtrack
    # from multiview_detector.datasets.MultiviewX import MultiviewX

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
    dataloader = DataLoader(dataset, 2, False, num_workers=0)
    imgs, world_gt, imgs_gt, frame = next(iter(dataloader))

    pass
    if test_proection:
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
    test()
