import torch.nn as nn
import torch
import torch.nn.functional as F
from . import misc


class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        """
        K has to be 16
        """
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.GroupNorm(4, 1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, output_channel),
            nn.LeakyReLU(negative_slope=0.2),
        )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        k = 4
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)
            assert idx.shape[1] == k
            idx_base = (
                torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1)
                * num_points_k
            )
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = (
            feature.view(batch_size, k, num_points_q, num_dims)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, f, coor):

        feature_list = []
        coor = coor.transpose(1, 2).contiguous()
        f = f.transpose(1, 2).contiguous()
        f = self.input_trans(f)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)

        f = torch.cat(feature_list, dim=1)

        f = self.layer5(f)

        f = f.transpose(-1, -2)

        return f


def knn_point(nsample, xyz, new_xyz):

    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):

        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:, :, :3]
            rgb = data[:, :, 3:]
        batch_size, num_points, _ = xyz.shape

        center = misc.fps(xyz, self.num_group)

        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(
                batch_size, self.num_group, self.group_size, -1
            ).contiguous()

        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel, point_input_dims=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):

        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)

        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse),
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )
        a = (
            torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float)
            .view(1, self.grid_size)
            .expand(self.grid_size, self.grid_size)
            .reshape(1, -1)
        )
        b = (
            torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float)
            .view(self.grid_size, 1)
            .expand(self.grid_size, self.grid_size)
            .reshape(1, -1)
        )
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size**2)

    def forward(self, feature_global):

        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3)

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1)

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)

        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1)

        fine = self.final_conv(feat) + center
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine


class DiscreteVAE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims

        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.dgcnn_1 = DGCNN(
            encoder_channel=self.encoder_dims, output_channel=self.num_tokens
        )
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))

        self.dgcnn_2 = DGCNN(
            encoder_channel=self.tokens_dims, output_channel=self.decoder_dims
        )
        self.decoder = Decoder(
            encoder_channel=self.decoder_dims, num_fine=self.group_size
        )

    def recon_loss(self, ret, gt):
        whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret

        bs, g, _, _ = coarse.shape

        coarse = coarse.reshape(bs * g, -1, 3).contiguous()
        fine = fine.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()

        loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
        loss_fine_block = self.loss_func_cdl1(fine, group_gt)

        loss_recon = loss_coarse_block + loss_fine_block

        return loss_recon

    def get_loss(self, ret, gt):

        loss_recon = self.recon_loss(ret, gt)

        logits = ret[-1]
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1.0 / self.num_tokens], device=gt.device))
        loss_klv = F.kl_div(
            log_qy,
            log_uniform.expand(log_qy.size(0), log_qy.size(1)),
            None,
            None,
            "batchmean",
            log_target=True,
        )

        return loss_recon, loss_klv

    def forward(self, inp, temperature=1.0, hard=False, **kwargs):
        neighborhood, center = self.group_divider(inp)
        logits = self.encoder(neighborhood)
        logits = self.dgcnn_1(logits, center)
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)
        sampled = torch.einsum("b g n, n c -> b g c", soft_one_hot, self.codebook)
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)

        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, logits)
        return ret
