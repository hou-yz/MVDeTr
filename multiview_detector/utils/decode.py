import torch
import torch.nn as nn
import torch.nn.functional as F
from multiview_detector.utils.tensor_utils import _gather_feat, _transpose_and_gather_feat


def _nms(heatmap, kernel_size=3):
    # kernel_size = kernel_size * 2 + 1
    hmax = F.max_pool2d(heatmap, (kernel_size, kernel_size), stride=1, padding=(kernel_size - 1) // 2)
    keep = (hmax == heatmap).float()
    return heatmap * keep


'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''


def _topk(scores, top_K):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), top_K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), top_K)
    topk_clses = (topk_ind / top_K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, top_K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, top_K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, top_K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heatmap, offset=None, wh=None, id=None, top_K=100):
    B, C, H, W = heatmap.shape

    scoremap = torch.sigmoid(heatmap)
    # perform nms on heatmaps
    scoremap = _nms(scoremap)

    scores, inds, _, ys, xs = _topk(scoremap, top_K=top_K)
    xy = torch.stack([xs, ys], dim=2)
    if offset is not None:
        offset = _transpose_and_gather_feat(offset, inds)
        offset = offset.view(B, top_K, 2)
        xy = xy + offset
    else:
        xy = xy + 0.5
    scores = scores.view(B, top_K, 1)

    # xywh
    if wh is None:
        detections = torch.cat([xy, scores], dim=2)
    else:
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(B, top_K, 2)
        detections = torch.cat([xy, wh, scores], dim=2)

    if id is not None:
        id = torch.argmax(id, dim=1)
        id = _transpose_and_gather_feat(id, inds)
        id = id.view(B, top_K, 2)
        detections = torch.cat([detections, id], dim=2)
    return detections


def mvdet_decode(scoremap, offset=None, reduce=4):
    B, C, H, W = scoremap.shape
    # scoremap = _nms(scoremap)

    xy = torch.nonzero(torch.ones_like(scoremap[:, 0])).view([B, H * W, 3])[:, :, [2, 1]].float()
    if offset is not None:
        offset = offset.permute(0, 2, 3, 1).reshape(B, H * W, 2)
        xy = xy + offset
    else:
        xy = xy + 0.5
    xy *= reduce
    scores = scoremap.permute(0, 2, 3, 1).reshape(B, H * W, 1)

    return torch.cat([xy, scores], dim=2)
