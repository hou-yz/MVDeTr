import os

os.environ['OMP_NUM_THREADS'] = '1'
from PIL import Image, ImageDraw
import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from multiview_detector.datasets import frameDataset, Wildtrack, MultiviewX
from multiview_detector.utils.image_utils import img_color_denormalize


def _traget_transform(target, kernel):
    with torch.no_grad():
        target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
    return target


def test(dataset_name='multiviewx'):
    if dataset_name == 'multiviewx':
        result_fpath = '/home/houyz/Code/MVDeTr/logs/multiviewx/augFCS_deform_trans_lr0.001_baseR0.1_neck128_out64_alpha1.0_id0_drop0.5_dropcam0.0_worldRK4_10_imgRK12_10_2021-04-09_22-39-33/test.txt'
        dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), False, )
    elif dataset_name == 'wildtrack':
        result_fpath = '/home/houyz/Code/MVDeTr/logs/wildtrack/augFCS_deform_trans_lr0.0005_baseR1_neck128_out64_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK12_10_2021-04-09_21-39-53/test.txt'
        dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), False, )
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    grid_size = list(map(lambda x: x * 3, dataset.Rworld_shape))
    bbox_by_pos_cam = dataset.base.read_pom()
    results = np.loadtxt(result_fpath)
    denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    reshape_transform = T.Resize(dataset.img_shape)

    # video = cv2.VideoWriter(f'{dataset_name}_test.avi', cv2.VideoWriter_fourcc(*"MJPG"), 2, (1580, 1060))
    for index in [31]:  # tqdm.tqdm(range(len(dataset))):
        # img_comb = np.zeros([1060, 1580, 3]).astype('uint8')
        map_res = np.zeros(dataset.Rworld_shape)
        imgs, map_gt, imgs_gt, affine_mats, frame = dataset.__getitem__(index)
        imgs = reshape_transform(denorm(imgs))
        res_map_grid = results[results[:, 0] == frame, 1:]
        # for ij in res_map_grid:
        #     i, j = (ij / dataset.world_reduce).astype(int)
        #     if dataset.base.indexing == 'xy':
        #         i, j = j, i
        #         map_res[i, j] = 1
        #     else:
        #         map_res[i, j] = 1
        # map_res = _traget_transform(torch.from_numpy(map_res).unsqueeze(0).unsqueeze(0).float(),
        #                             dataset.world_kernel)
        # map_res = F.interpolate(map_res, grid_size).squeeze().numpy()
        # map_res = np.uint8(255 * map_res)
        # map_res = cv2.applyColorMap(map_res, cv2.COLORMAP_JET)
        # cv2.imwrite('world')
        # map_res = cv2.putText(map_res, 'Ground Plane', (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #                       1, (87, 59, 233), 2, cv2.LINE_AA)

        # img_comb[580:580 + grid_size[0], 500:500 + grid_size[1]] = map_res
        # plt.imshow(cv2.cvtColor(img_comb.astype('uint8'), cv2.COLOR_BGR2RGB))
        # plt.show()

        res_posID = dataset.base.get_pos_from_worldgrid(res_map_grid.transpose())
        # gt_map_grid = map_gt[0].nonzero().cpu().numpy() * dataset.world_reduce
        # gt_posID = dataset.base.get_pos_from_worldgrid(gt_map_grid.transpose())

        for cam in range(dataset.num_cam):
            img = (imgs[cam].cpu().numpy().transpose([1, 2, 0]) * 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for posID in res_posID:
                bbox = bbox_by_pos_cam[posID][cam]
                if bbox is not None:
                    # bbox = tuple(map(lambda x: int(x / 4), bbox))
                    cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
            pass

            cv2.imwrite(f'imgs/cam{cam + 1}.png', img)
            # img = cv2.putText(img, f'Camera {cam + 1}', (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
            #                   1, (87, 59, 233), 2, cv2.LINE_AA)
            # i, j = cam // 3, cam % 3
            # img_comb[i * 290:i * 290 + 270, j * 500:j * 500 + 480] = img

        # video.write(img_comb)
        # plt.imshow(cv2.cvtColor(img_comb.astype('uint8'), cv2.COLOR_BGR2RGB))
        # plt.show()
        pass
    # video.release()


if __name__ == '__main__':
    test('multiviewx')
    # test('wildtrack')
