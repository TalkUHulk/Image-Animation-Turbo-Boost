import torch
import torch.nn.functional as F
import itertools
import operator


def gather(input, dim, index):
    indices = [torch.arange(size, device=index.device) for size in index.shape]
    indices = list(torch.meshgrid(*indices))
    indices[dim] = index
    sizes = list(reversed(list(itertools.accumulate(reversed(input.shape), operator.mul))))
    index = sum((index * size for index, size in zip(indices, sizes[1:] + [1])))
    output = input.flatten()[index]
    return output


def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """

    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)

    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0, device=x0.device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1, device=x0.device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0, device=x1.device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1, device=x1.device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0, device=y0.device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1, device=y0.device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0, device=y1.device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1, device=y1.device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    # if (x0 + y0 * padded_w).shape[0] == 1:
    #     x0_y0 = torch.add(x0, y0 * padded_w).repeat(c, 1).unsqueeze(0)
    #     x0_y1 = torch.add(x0, y1 * padded_w).repeat(c, 1).unsqueeze(0)
    #     x1_y0 = torch.add(x1, y0 * padded_w).repeat(c, 1).unsqueeze(0)
    #     x1_y1 = torch.add(x1, y1 * padded_w).repeat(c, 1).unsqueeze(0)
    # else:
    #     x0_y0 = torch.add(x0, y0 * padded_w).unsqueeze(0).repeat(c, 1, 1).transpose(1, 0)
    #     x0_y1 = torch.add(x0, y1 * padded_w).unsqueeze(0).repeat(c, 1, 1).transpose(1, 0)
    #     x1_y0 = torch.add(x1, y0 * padded_w).unsqueeze(0).repeat(c, 1, 1).transpose(1, 0)
    #     x1_y1 = torch.add(x1, y1 * padded_w).unsqueeze(0).repeat(c, 1, 1).transpose(1, 0)

    # if (x0 + y0 * padded_w).shape[0] == 1:
    #     # x0_y0 = (x0 + y0 * padded_w).squeeze().repeat(c).view(1, 256, 4096)
    #     # x0_y1 = (x0 + y1 * padded_w).squeeze().repeat(c).view(1, 256, 4096)
    #     # x1_y0 = (x1 + y0 * padded_w).squeeze().repeat(c).view(1, 256, 4096)
    #     # x1_y1 = (x1 + y1 * padded_w).squeeze().repeat(c).view(1, 256, 4096)
    #     x0_y0 = torch.ones(1, 256, 4096, dtype=x0.dtype)
    #     x0_y1 = torch.ones(1, 256, 4096, dtype=x0.dtype)
    #     x1_y0 = torch.ones(1, 256, 4096, dtype=x0.dtype)
    #     x1_y1 = torch.ones(1, 256, 4096, dtype=x0.dtype)
    # else:
    #     # x0_y0 = torch.stack([(x0 + y0 * padded_w) for _ in range(c)]).transpose(1, 0)
    #     # x0_y1 = torch.stack([(x0 + y1 * padded_w) for _ in range(c)]).transpose(1, 0)
    #     # x1_y0 = torch.stack([(x1 + y0 * padded_w) for _ in range(c)]).transpose(1, 0)
    #     # x1_y1 = torch.stack([(x1 + y1 * padded_w) for _ in range(c)]).transpose(1, 0)
    #     x0_y0 = torch.ones(11, 3, 4096, dtype=x0.dtype)
    #     x0_y1 = torch.ones(11, 3, 4096, dtype=x0.dtype)
    #     x1_y0 = torch.ones(11, 3, 4096, dtype=x0.dtype)
    #     x1_y1 = torch.ones(11, 3, 4096, dtype=x0.dtype)

    # x0_y0 = torch.cat([(x0 + y0 * padded_w).unsqueeze(1) for _ in range(c)], axis=1)
    # x0_y1 = torch.cat([(x0 + y1 * padded_w).unsqueeze(1) for _ in range(c)], axis=1)
    # x1_y0 = torch.cat([(x1 + y0 * padded_w).unsqueeze(1) for _ in range(c)], axis=1)
    # x1_y1 = torch.cat([(x1 + y1 * padded_w).unsqueeze(1) for _ in range(c)], axis=1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)
    # Ia = gather(im_padded, 2, x0_y0)
    # Ib = gather(im_padded, 2, x0_y1)
    # Ic = gather(im_padded, 2, x1_y0)
    # Id = gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)
