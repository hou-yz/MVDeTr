import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from multiview_detector.models.transformer import TransformerEncoderLayer, TransformerEncoder


def create_pos_embedding(img_size, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi
    H, W = img_size
    not_mask = torch.ones([1, H, W])
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos


class TransformerWorldFeat(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, use_multicam=False,
                 hidden_dim=256, dropout=0.1, nhead=8, dim_feedforward=512):
        super(TransformerWorldFeat, self).__init__()
        self.use_multicam = use_multicam
        self.pos_embedding = create_pos_embedding(np.array(Rworld_shape) // 4, hidden_dim // 2)
        self.merge_linear = nn.Sequential(nn.Conv2d(base_dim * num_cam, hidden_dim, 1), nn.ReLU())
        self.merged_downsample = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(),
                                               nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(), )
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, dropout=dropout, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.merged_encoder = TransformerEncoder(encoder_layer, 3)
        self.merged_upsample = nn.Sequential(nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, 2, 1, 1), nn.ReLU(),
                                             nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, 2, 1, 1), nn.ReLU(), )

    def forward(self, x):
        B, N, C, H, W = x.shape
        multicam_feat = x.view(B, N * C, H, W)
        shortcut = merged_feat = self.merge_linear(multicam_feat)
        merged_feat = self.merged_downsample(merged_feat)
        B, C, H, W = merged_feat.shape
        merged_feat = merged_feat.flatten(2).permute(2, 0, 1)
        pos_embedding = self.pos_embedding.repeat(B, 1, 1, 1).flatten(2).permute(2, 0, 1).to(x.device)
        merged_feat = self.merged_encoder(merged_feat, pos=pos_embedding)
        merged_feat = merged_feat.permute(1, 2, 0).view(B, C, H, W)
        merged_feat = self.merged_upsample(merged_feat)
        # return merged_feat + shortcut
        return merged_feat


def test():
    in_feat = torch.zeros([1, 7, 128, 120, 360])
    model = TransformerWorldFeat(7, [120 // 4, 360 // 4], 128, False)
    out_feat = model(in_feat)
    pass


if __name__ == '__main__':
    test()
