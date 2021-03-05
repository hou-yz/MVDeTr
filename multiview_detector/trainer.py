import time
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.losses import *
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, denormalize, cls_thres=0.4, alpha=1.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss()
        self.regress_loss = RegL1Loss()
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha

    def train(self, epoch, dataloader, optimizer, scaler, scheduler=None, log_interval=100):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, world_gt, imgs_gt, _) in enumerate(dataloader):
            # with autocast():
            (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data)
            B, N, K = imgs_gt['reg_mask'].shape
            loss_w_hm = self.focal_loss(world_heatmap, world_gt['heatmap'])
            loss_w_off = self.regress_loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
            loss_img_hm = self.focal_loss(imgs_heatmap, imgs_gt['heatmap'])
            loss_img_off = self.regress_loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
            loss_img_wh = self.regress_loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])

            w_loss = loss_w_hm + loss_w_off
            img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1
            loss = w_loss + img_loss / N * self.alpha
            # loss = self.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device))

            t_f = time.time()
            t_forward += t_f - t_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            losses += loss.item()

            t_b = time.time()
            t_backward += t_b - t_f

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, '
                      f'Time: {t_epoch:.1f}, maxima: {world_heatmap.max():.3f}')
                pass
        return losses / len(dataloader)

    def test(self, dataloader, res_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        res_list = []
        t0 = time.time()
        for batch_idx, (data, world_gt, imgs_gt, frame) in enumerate(dataloader):
            # with autocast():
            with torch.no_grad():
                (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data)
                B, N, K = imgs_gt['reg_mask'].shape
                loss_w_hm = self.focal_loss(world_heatmap, world_gt['heatmap'])
                loss_w_off = self.regress_loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
                loss_img_hm = self.focal_loss(imgs_heatmap, imgs_gt['heatmap'])
                loss_img_off = self.regress_loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                loss_img_wh = self.regress_loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])

                w_loss = loss_w_hm + loss_w_off
                img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1
                loss = w_loss + img_loss / N * self.alpha
            # loss = self.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device))

            losses += loss.item()

            if res_fpath is not None:
                xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(),
                                   reduce=dataloader.dataset.world_reduce)
                # xys = mvdet_decode(world_heatmap.detach().cpu(), reduce=dataloader.dataset.world_reduce)
                grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
                if dataloader.dataset.base.indexing == 'xy':
                    positions = grid_xy
                else:
                    positions = grid_xy[:, :, [1, 0]]

                for b in range(B):
                    ids = scores[b].squeeze() > self.cls_thres
                    pos, s = positions[b, ids], scores[b, ids, 0]
                    ids, count = nms(pos, s, 20, np.inf)
                    res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                    res_list.append(res)

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            # visualizing the heatmap for world
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt1 = fig.add_subplot(212, title="target")
            subplt0.imshow(world_heatmap.cpu().detach().numpy().squeeze())
            subplt1.imshow(world_gt['heatmap'].squeeze())
            plt.savefig(os.path.join(self.logdir, 'world.jpg'))
            plt.close(fig)
            # visualizing the heatmap for per-view estimation
            heatmap0_foot = imgs_heatmap[0].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        if res_fpath is not None:
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')
            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(dataloader.dataset.gt_fpath),
                                                     dataloader.dataset.base.__name__)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
        else:
            moda = 0

        print(f'Test, loss: {losses / len(dataloader):.6f}, Time: {t_epoch:.3f}')

        return losses / len(dataloader), moda
