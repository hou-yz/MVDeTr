import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from multiview_detector.models.transformer import TransformerEncoderLayer, TransformerEncoder
from multiview_detector.models.deformable_transformer import DeformableTransformerEncoderLayer, \
    DeformableTransformerEncoder
from multiview_detector.models.ops.modules import MSDeformAttn
from multiview_detector.utils.image_utils import array2heatmap
import matplotlib.pyplot as plt


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
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, dropout=0.1, nhead=8, dim_feedforward=512):
        super(TransformerWorldFeat, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(base_dim * num_cam, hidden_dim, 3, 2, 1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(), )

        self.pos_embedding = create_pos_embedding(np.ceil(np.array(Rworld_shape) / 4).astype(int),
                                                  hidden_dim // 2)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, dropout=dropout, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layer, 3)

        self.upsample = nn.Sequential(nn.Upsample(np.ceil(np.array(Rworld_shape) / 2).astype(int).tolist(),
                                                  mode='bilinear'),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(),
                                      nn.Upsample(Rworld_shape, mode='bilinear'),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(), )

    def forward(self, x, visualize=False):
        B, N, C, H, W = x.shape
        # _, _, H, W = x2.shape
        x = self.downsample(x.view(B, N * C, H, W))
        _, _, H, W = x.shape
        # H*W,B,C*N
        pos_embedding = self.pos_embedding.repeat(B, 1, 1, 1).flatten(2).permute(2, 0, 1).to(x.device)
        x = self.encoder(x.flatten(2).permute(2, 0, 1), pos=pos_embedding)
        merged_feat = self.upsample(x.permute(1, 2, 0).view(B, C, H, W))
        return merged_feat


class DeformTransWorldFeat(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, dropout=0.1, nhead=8, dim_feedforward=512,
                 n_points=4, stride=2, reference_points=None):
        super(DeformTransWorldFeat, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 3, stride, 1), nn.ReLU(), )

        encoder_layer = DeformableTransformerEncoderLayer(hidden_dim, dim_feedforward, dropout,
                                                          n_levels=num_cam, n_heads=nhead, n_points=n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, 3, reference_points)
        self.pos_embedding = create_pos_embedding(np.array(Rworld_shape) // stride, hidden_dim // 2)
        self.lvl_embedding = nn.Parameter(torch.Tensor(num_cam, hidden_dim))

        self.merge_linear = nn.Sequential(nn.Conv2d(hidden_dim * num_cam, hidden_dim, 1), nn.ReLU())
        self.upsample = nn.Sequential(nn.Upsample(Rworld_shape, mode='bilinear', align_corners=False),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(), )
        self._reset_parameters()

    def forward(self, x, visualize=False):
        B, N, C, H, W = x.shape
        x = self.downsample(x.view(B * N, C, H, W))
        _, _, H, W = x.shape

        src_flatten = x.view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous().view([B, N * H * W, C])
        lvl_pos_embed_flatten = (self.pos_embedding.to(x.device).flatten(2).transpose(1, 2).unsqueeze(1) +
                                 self.lvl_embedding.view([B, N, 1, C])).view([B, N * H * W, C])
        spatial_shapes = torch.as_tensor(np.array([[H, W]] * N), dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones([B, N, 2], device=x.device)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten)

        if visualize:
            for cam in range(N):
                world_feat = memory.view(B, N, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
                visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
                visualize_img.save(f'../../imgs/worldfeat{cam + 1}.png')
                plt.imshow(visualize_img)
                plt.show()
        merged_feat = self.merge_linear(memory.view(B, N, H, W, C).permute(0, 1, 4, 2, 3).contiguous().
                                        view(B, N * C, H, W))
        merged_feat = self.upsample(merged_feat)
        return merged_feat

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.lvl_embedding)


class DeformTransWorldFeat_aio(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, dropout=0.1, nhead=8, dim_feedforward=512):
        super(DeformTransWorldFeat_aio, self).__init__()
        self.merge = nn.Sequential(nn.Conv2d(base_dim * num_cam, hidden_dim, 1), nn.ReLU(), )
        encoder_layer = DeformableTransformerEncoderLayer(hidden_dim, dim_feedforward, n_levels=1, n_heads=nhead)
        self.encoder = DeformableTransformerEncoder(encoder_layer, 3)
        self.pos_embedding = create_pos_embedding(Rworld_shape, hidden_dim // 2)
        self.output = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), )

        self._reset_parameters()

    def forward(self, x, visualize=False):
        B, N, C, H, W = x.shape
        x = self.merge(x.view(B, N * C, H, W))
        B, C, H, W = x.shape
        src_flatten = x.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view([B, H * W, C])
        spatial_shapes = torch.as_tensor(np.array([[H, W]]), dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones([B, 1, 2], device=x.device)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              self.pos_embedding.to(x.device).flatten(2).transpose(1, 2))

        merged_feat = memory.view(B, H, W, C).permute(0, 3, 1, 2).contiguous().view(B, C, H, W)
        merged_feat = self.output(merged_feat)
        return merged_feat

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()


def test():
    H, W = 640 // 4, 1000 // 4
    in_feat = torch.zeros([1, 6, 128, H, W]).cuda()
    # model = TransformerWorldFeat(6, [H, W], 128).cuda()
    model = DeformTransWorldFeat(6, [H, W], 128).cuda()
    out_feat = model(in_feat)
    pass


if __name__ == '__main__':
    test()
