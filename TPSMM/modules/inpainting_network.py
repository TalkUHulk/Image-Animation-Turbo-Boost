import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork
from modules.grid_sample import bilinear_grid_sample


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """

    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask=True, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        # self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = [max_features, max_features, max_features // 2]
        out_features = [max_features // 2, max_features // 4, max_features // 8]
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        resblock = []
        for i in range(num_down_blocks):
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

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

    def forward(self, source_image,
                deformation,
                occlusion_map0, occlusion_map1, occlusion_map2, occlusion_map3):
        # occlusion_map = [occlusion_map0, occlusion_map1, occlusion_map2, occlusion_map3]

        deformed_source = self.deform_input(source_image, deformation)

        out = self.first(source_image)

        encoder_map0 = out

        encoder_map1 = self.down_blocks[0](encoder_map0)

        encoder_map2 = self.down_blocks[1](encoder_map1)
        encoder_map3 = self.down_blocks[2](encoder_map2)

        # output_dict = {}
        # output_dict['contribution_maps'] = dense_motion['contribution_maps']
        # output_dict['deformed_source'] = dense_motion['deformed_source']

        # occlusion_map = dense_motion['occlusion_map']
        # output_dict['occlusion_map'] = occlusion_map

        # deformation = dense_motion['deformation']
        # out_ij = self.deform_input(out.detach(), deformation)

        out = self.deform_input(encoder_map3, deformation)

        # out_ij = self.occlude_input(out_ij, occlusion_map[0].detach())
        out = self.occlude_input(out, occlusion_map0)

        # warped_encoder_maps = []
        # warped_encoder_maps.append(out_ij)
        encoder_map = [encoder_map0, encoder_map1, encoder_map2, encoder_map3]

        # 0
        out = self.resblock[0](out)
        out = self.resblock[1](out)
        out = self.up_blocks[0](out)
        encode_i = self.deform_input(encoder_map2, deformation)
        encode_i = self.occlude_input(encode_i, occlusion_map1)
        out = torch.cat([out, encode_i], 1)

        # 1
        out = self.resblock[2](out)
        out = self.resblock[3](out)
        out = self.up_blocks[1](out)
        encode_i = self.deform_input(encoder_map1, deformation)
        encode_i = self.occlude_input(encode_i, occlusion_map2)
        out = torch.cat([out, encode_i], 1)

        # 2
        out = self.resblock[4](out)
        out = self.resblock[5](out)
        out = self.up_blocks[2](out)
        encode_i = self.deform_input(encoder_map0, deformation)
        encode_i = self.occlude_input(encode_i, occlusion_map3)

        # output_dict["deformed"] = deformed_source
        # output_dict["warped_encoder_maps"] = warped_encoder_maps

        # occlusion_last = occlusion_map3
        # if not self.multi_mask:
        #     occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear', align_corners=True)

        out = out * (1 - occlusion_map3) + encode_i

        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_map3) + deformed_source * occlusion_map3
        # output_dict["prediction"] = out
        return out

    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(out.detach(), occlusion_map[-1].detach()))
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            out_mask = self.occlude_input(out.detach(), occlusion_map[2 - i].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map
