from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, transform_frame
import math
from modules.grid_sample import bilinear_grid_sample
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d


class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks 
                        from K TPS transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_channels,
                 scale_factor=0.25, bg=False, multi_mask=True, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()

        self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor

        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))

        up = []
        self.up_nums = int(math.log(1 / scale_factor, 2))
        self.occlusion_num = 4

        channel = [hourglass_output_size[-1] // (2 ** i) for i in range(self.up_nums)]
        for i in range(self.up_nums):
            up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
        self.up = nn.ModuleList(up)

        channel = [hourglass_output_size[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]]
        for i in range(self.up_nums):
            channel.append(hourglass_output_size[-1] // (2 ** (i + 1)))

        occlusion = []
        for i in range(self.occlusion_num):
            occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
        self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps
        self.bg = bg
        self.kp_variance = kp_variance

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):

        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(
            heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)

        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        # K TPS transformaions
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving.view(bs, -1, 5, 2)
        kp_2 = kp_source.view(bs, -1, 5, 2)
        # trans = TPS(mode='kp', bs=bs, kp_1=kp_1, kp_2=kp_2)

        grid = make_coordinate_grid([64, 64], type=torch.FloatTensor).unsqueeze(0)
        grid = grid.view(1, 4096, 2)
        grid = torch.from_numpy(grid.cpu().detach().numpy()).to(source_image.device())
        driving_to_source = transform_frame(grid, kp_1, kp_2)

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # affine background transformation
        identity_grid = to_homogeneous(identity_grid)
        identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
        identity_grid = from_homogeneous(identity_grid)
        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        return transformations

    def create_deformed_source_image(self, source_image, transformations):

        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_tps + 1), -1, h, w)
        transformations = transformations.view((bs * (self.num_tps + 1), h, w, -1))
        # deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = bilinear_grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, self.num_tps + 1, -1, h, w))
        return deformed

    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        '''
        drop = (torch.rand(X.shape[0], X.shape[1]) < (1 - P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2], X.shape[3], 1, 1).permute(2, 3, 0, 1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:, 1:, ...] /= (1 - P)
        mask_bool = (drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition

    def forward(self, source_input, kp_driving, kp_source, bg_param):

        source_image = self.down(source_input)

        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)

        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)

        deformed_source = self.create_deformed_source_image(source_image, transformations)

        deformed_source = deformed_source.view(1, -1, 64, 64)
        input = torch.cat([heatmap_representation, deformed_source], dim=1)

        prediction = self.hourglass(input, mode=1)

        contribution_maps = self.maps(prediction[-1])

        contribution_maps = F.softmax(contribution_maps, dim=1)

        # Combine the K+1 transformations
        # Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        # out_dict['deformation'] = deformation  # Optical Flow

        occlusion_map = []
        # if self.multi_mask:
        for i in range(self.occlusion_num - self.up_nums):
            occlusion_map.append(
                torch.sigmoid(self.occlusion[i](prediction[self.up_nums - self.occlusion_num + i])))
        prediction = prediction[-1]
        for i in range(self.up_nums):
            prediction = self.up[i](prediction)
            occlusion_map.append(torch.sigmoid(self.occlusion[i + self.occlusion_num - self.up_nums](prediction)))

        return deformed_source, transformations, contribution_maps, deformation, occlusion_map
        # print(deformation)
        # return deformation


