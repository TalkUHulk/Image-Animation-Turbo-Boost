from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, transform_frame
import math
from modules.grid_sample import bilinear_grid_sample
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d


class DenseMotionInpaintingNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks
                        from K TPS transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_channels,
                 scale_factor=0.25, bg=False, multi_mask=True, kp_variance=0.01):
        super(DenseMotionInpaintingNetwork, self).__init__()

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

        ## ---------------------------
        max_features = 512
        self.num_down_blocks = 3
        # self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(3):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = [max_features, max_features, max_features // 2]
        out_features = [max_features // 2, max_features // 4, max_features // 8]
        for i in range(3):
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        resblock = []
        for i in range(3):
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

        self.num_channels = num_channels

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
        # 切断tracing
        grid = torch.from_numpy(grid.cpu().detach().numpy()).to(device=source_image.device)

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

    def forward(self, source_input, kp_driving, kp_source, bg_param=None):

        source_image = self.down(source_input)

        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)

        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)

        deformed_source = self.create_deformed_source_image(source_image, transformations)

        deformed_source = deformed_source.view(1, -1, 64, 64)
        input = torch.cat([heatmap_representation, deformed_source], dim=1)

        predictions = self.hourglass(input, mode=1)

        contribution_maps = self.maps(predictions[-1])

        contribution_maps = F.softmax(contribution_maps, dim=1)

        # Combine the K+1 transformations
        # Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        occlusion_map0 = torch.sigmoid(self.occlusion[0](predictions[-2]))
        occlusion_map1 = torch.sigmoid(self.occlusion[1](predictions[-1]))

        prediction = predictions[-1]

        prediction = self.up[0](prediction)
        occlusion_map2 = torch.sigmoid(self.occlusion[2](prediction))

        prediction = self.up[1](prediction)
        occlusion_map3 = torch.sigmoid(self.occlusion[3](prediction))

        # inpainting

        deformed_source = self.deform_input(source_input, deformation)

        out = self.first(source_input)

        encoder_map0 = out.clone()
        encoder_map1 = self.down_blocks[0](encoder_map0)
        encoder_map2 = self.down_blocks[1](encoder_map1)
        encoder_map3 = self.down_blocks[2](encoder_map2)

        out = self.deform_input(encoder_map3, deformation)
        out = self.occlude_input(out, occlusion_map0)

        # i = 0
        out = self.resblock[0](out)
        out = self.resblock[1](out)
        out = self.up_blocks[0](out)
        encode_i = self.deform_input(encoder_map2, deformation)
        encode_i = self.occlude_input(encode_i, occlusion_map1)
        out = torch.cat([out, encode_i], 1)

        # i = 1
        out = self.resblock[2](out)
        out = self.resblock[3](out)
        out = self.up_blocks[1](out)
        encode_i = self.deform_input(encoder_map1, deformation)
        encode_i = self.occlude_input(encode_i, occlusion_map2)
        out = torch.cat([out, encode_i], 1)

        # i = 2
        out = self.resblock[4](out)
        out = self.resblock[5](out)
        out = self.up_blocks[2](out)
        encode_i = self.deform_input(encoder_map0, deformation)
        encode_i = self.occlude_input(encode_i, occlusion_map3)

        out = out * (1 - occlusion_map3) + encode_i
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_map3) + deformed_source * occlusion_map3
        return out

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        # if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
        deformation = deformation.permute(0, 2, 3, 1)
        # return F.grid_sample(inp, deformation, align_corners=True)
        return bilinear_grid_sample(inp, deformation, align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        # if not self.multi_mask:
        #     if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
        #         occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear', align_corners=True)
        out = inp * occlusion_map
        return out
