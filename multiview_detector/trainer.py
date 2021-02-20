import time
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.evaluation.evaluate import evaluate
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
        self.criterion = GaussianMSE()
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha

    def train(self, epoch, dataloader, optimizer, scaler, cyclic_scheduler=None, log_interval=100):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, world_gt, imgs_gt, _) in enumerate(dataloader):
            # with autocast():
            world_res, imgs_res = self.model(data)
            B, N = imgs_gt.shape[:2]
            loss = self.criterion(world_res, world_gt.to(world_res.device), dataloader.dataset.world_kernel) + \
                   self.criterion(imgs_res.view([B * N] + list(imgs_res.shape[2:])),
                                  imgs_gt.view([B * N] + list(imgs_gt.shape[2:])).to(imgs_res.device),
                                  dataloader.dataset.img_kernel) / N * self.alpha
            t_f = time.time()
            t_forward += t_f - t_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            losses += loss.item()
            pred = (world_res > self.cls_thres).int().to(world_gt.device)
            true_positive = (pred.eq(world_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = world_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, '
                      f'prec: {precision_s.avg * 100:.1f}%, recall: {recall_s.avg * 100:.1f}%, \tTime: {t_epoch:.1f} '
                      f'(f{t_forward / (batch_idx + 1):.3f}+b{t_backward / (batch_idx + 1):.3f}), maxima: {world_res.max():.3f}')
                pass
        return losses / len(dataloader), precision_s.avg * 100

    def test(self, dataloader, res_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, world_gt, imgs_gt, frame) in enumerate(dataloader):
            # with autocast():

            with torch.no_grad():
                world_res, imgs_res = self.model(data)
                B, N = imgs_gt.shape[:2]
                loss = self.criterion(world_res, world_gt.to(world_res.device), dataloader.dataset.world_kernel) + \
                       self.criterion(imgs_res.view([B * N] + list(imgs_res.shape[2:])),
                                      imgs_gt.view([B * N] + list(imgs_gt.shape[2:])).to(imgs_res.device),
                                      dataloader.dataset.img_kernel) / N * self.alpha
            losses += loss.item()
            pred = (world_res > self.cls_thres).int().to(world_gt.device)
            true_positive = (pred.eq(world_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = world_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            if res_fpath is not None:
                world_grid_res = world_res.detach().cpu().squeeze()
                v_s = world_grid_res[world_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = torch.nonzero(world_grid_res > self.cls_thres, as_tuple=False)
                if dataloader.dataset.base.indexing == 'xy':
                    grid_index = grid_ij[:, [1, 0]]
                else:
                    grid_index = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, (grid_index.float() + 0.5) *
                                               dataloader.dataset.world_reduce, v_s], dim=1))

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt1 = fig.add_subplot(212, title="target")
            subplt0.imshow(world_res.cpu().detach().numpy().squeeze())
            subplt1.imshow(self.criterion._traget_transform(world_res, world_gt, dataloader.dataset.map_kernel)
                           .cpu().detach().numpy().squeeze())
            plt.savefig(os.path.join(self.logdir, 'world.jpg'))
            plt.close(fig)

            # visualizing the heatmap for per-view estimation
            heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
            heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(dataloader.dataset.gt_fpath),
                                                     dataloader.dataset.base.__name__)
            print('moda: {:.1f}%, modp: {:.1f}%, prec: {:.1f}%, recall: {:.1f}%'.
                  format(moda, modp, precision, recall))

        print('Test, loss: {:.6f}, prec: {:.1f}%, recall: {:.1f}%, \tTime: {:.3f}'.format(
            losses / (len(dataloader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(dataloader), precision_s.avg * 100, moda
