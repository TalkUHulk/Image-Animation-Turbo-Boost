import numpy as np

from torch import nn
import torch.nn.functional as F
import torch
from torch.linalg import det, inv


# def inverse_Gauss_Jordan(x):
#     I = torch.eye(8)
#
#     for i in range(8):
#         if x[i, i] == .0:
#             continue
#         for j in range(8):
#             if i != j:
#                 ratio = x[j, i] / x[i, i]
#                 for k in range(8):
#                     x[j, k] = x[j, k] - ratio * x[i, k]
#                     I[j, k] = I[j, k] - ratio * I[i, k]
#     for i in range(8):
#         divisor = x[i][i]
#         for j in range(8):
#             I[i, j] = I[i, j] / divisor
#     return I

# def inverse_Gauss_Jordan(x):
#     I = torch.eye(8)
#     for i in range(8):
#         for j in range(8):
#             if i != j:
#                 ratio = x[j, i] / (x[i, i] + 1e-8)
#                 x[j] = x[j] - ratio * x[i]
#                 I[j] = I[j] - ratio * I[i]
#
#     diag = torch.stack([x[i, i] for i in range(8)], dim=0).view(8, 1)
#     I = I / diag
#     return I

def inverse_Gauss_Jordan(_input):
    x = _input.clone()
    bs = x.shape[0]
    I = torch.eye(8).view(1, 8, 8).repeat(bs, 1, 1)

    index = torch.arange(0, 10).reshape((10, 1)).repeat(1, 8)

    ratio = x[:, 1, 0] / (x[:, 0, 0] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 1, :].scatter_(0, index, x[:, 1, :] - ratio * x[:, 0, :])
    I[:, 1, :].scatter_(0, index, I[:, 1, :] - ratio * I[:, 0, :])

    ratio = x[:, 2, 0] / (x[:, 0, 0] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 2, :].scatter_(0, index, x[:, 2, :] - ratio * x[:, 0, :])
    I[:, 2, :].scatter_(0, index, I[:, 2, :] - ratio * I[:, 0, :])

    ratio = x[:, 3, 0] / (x[:, 0, 0] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 3, :].scatter_(0, index, x[:, 3, :] - ratio * x[:, 0, :])
    I[:, 3, :].scatter_(0, index, I[:, 3, :] - ratio * I[:, 0, :])

    ratio = x[:, 4, 0] / (x[:, 0, 0] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 4, :].scatter_(0, index, x[:, 4, :] - ratio * x[:, 0, :])
    I[:, 4, :].scatter_(0, index, I[:, 4, :] - ratio * I[:, 0, :])

    ratio = x[:, 5, 0] / (x[:, 0, 0] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 5, :].scatter_(0, index, x[:, 5, :] - ratio * x[:, 0, :])
    I[:, 5, :].scatter_(0, index, I[:, 5, :] - ratio * I[:, 0, :])

    ratio = x[:, 6, 0] / (x[:, 0, 0] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 6, :].scatter_(0, index, x[:, 6, :] - ratio * x[:, 0, :])
    I[:, 6, :].scatter_(0, index, I[:, 6, :] - ratio * I[:, 0, :])

    ratio = x[:, 7, 0] / (x[:, 0, 0] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 7, :].scatter_(0, index, x[:, 7, :] - ratio * x[:, 0, :])
    I[:, 7, :].scatter_(0, index, I[:, 7, :] - ratio * I[:, 0, :])

    ratio = x[:, 0, 1] / (x[:, 1, 1] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 0, :].scatter_(0, index, x[:, 0, :] - ratio * x[:, 1, :])
    I[:, 0, :].scatter_(0, index, I[:, 0, :] - ratio * I[:, 1, :])

    ratio = x[:, 2, 1] / (x[:, 1, 1] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 2, :].scatter_(0, index, x[:, 2, :] - ratio * x[:, 1, :])
    I[:, 2, :].scatter_(0, index, I[:, 2, :] - ratio * I[:, 1, :])

    ratio = x[:, 3, 1] / (x[:, 1, 1] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 3, :].scatter_(0, index, x[:, 3, :] - ratio * x[:, 1, :])
    I[:, 3, :].scatter_(0, index, I[:, 3, :] - ratio * I[:, 1, :])

    ratio = x[:, 4, 1] / (x[:, 1, 1] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 4, :].scatter_(0, index, x[:, 4, :] - ratio * x[:, 1, :])
    I[:, 4, :].scatter_(0, index, I[:, 4, :] - ratio * I[:, 1, :])

    ratio = x[:, 5, 1] / (x[:, 1, 1] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 5, :].scatter_(0, index, x[:, 5, :] - ratio * x[:, 1, :])
    I[:, 5, :].scatter_(0, index, I[:, 5, :] - ratio * I[:, 1, :])

    ratio = x[:, 6, 1] / (x[:, 1, 1] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 6, :].scatter_(0, index, x[:, 6, :] - ratio * x[:, 1, :])
    I[:, 6, :].scatter_(0, index, I[:, 6, :] - ratio * I[:, 1, :])

    ratio = x[:, 7, 1] / (x[:, 1, 1] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 7, :].scatter_(0, index, x[:, 7, :] - ratio * x[:, 1, :])
    I[:, 7, :].scatter_(0, index, I[:, 7, :] - ratio * I[:, 1, :])

    ratio = x[:, 0, 2] / (x[:, 2, 2] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 0, :].scatter_(0, index, x[:, 0, :] - ratio * x[:, 2, :])
    I[:, 0, :].scatter_(0, index, I[:, 0, :] - ratio * I[:, 2, :])

    ratio = x[:, 1, 2] / (x[:, 2, 2] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 1, :].scatter_(0, index, x[:, 1, :] - ratio * x[:, 2, :])
    I[:, 1, :].scatter_(0, index, I[:, 1, :] - ratio * I[:, 2, :])

    ratio = x[:, 3, 2] / (x[:, 2, 2] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 3, :].scatter_(0, index, x[:, 3, :] - ratio * x[:, 2, :])
    I[:, 3, :].scatter_(0, index, I[:, 3, :] - ratio * I[:, 2, :])

    ratio = x[:, 4, 2] / (x[:, 2, 2] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 4, :].scatter_(0, index, x[:, 4, :] - ratio * x[:, 2, :])
    I[:, 4, :].scatter_(0, index, I[:, 4, :] - ratio * I[:, 2, :])

    ratio = x[:, 5, 2] / (x[:, 2, 2] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 5, :].scatter_(0, index, x[:, 5, :] - ratio * x[:, 2, :])
    I[:, 5, :].scatter_(0, index, I[:, 5, :] - ratio * I[:, 2, :])

    ratio = x[:, 6, 2] / (x[:, 2, 2] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 6, :].scatter_(0, index, x[:, 6, :] - ratio * x[:, 2, :])
    I[:, 6, :].scatter_(0, index, I[:, 6, :] - ratio * I[:, 2, :])

    ratio = x[:, 7, 2] / (x[:, 2, 2] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 7, :].scatter_(0, index, x[:, 7, :] - ratio * x[:, 2, :])
    I[:, 7, :].scatter_(0, index, I[:, 7, :] - ratio * I[:, 2, :])

    ratio = x[:, 0, 3] / (x[:, 3, 3] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 0, :].scatter_(0, index, x[:, 0, :] - ratio * x[:, 3, :])
    I[:, 0, :].scatter_(0, index, I[:, 0, :] - ratio * I[:, 3, :])

    ratio = x[:, 1, 3] / (x[:, 3, 3] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 1, :].scatter_(0, index, x[:, 1, :] - ratio * x[:, 3, :])
    I[:, 1, :].scatter_(0, index, I[:, 1, :] - ratio * I[:, 3, :])

    ratio = x[:, 2, 3] / (x[:, 3, 3] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 2, :].scatter_(0, index, x[:, 2, :] - ratio * x[:, 3, :])
    I[:, 2, :].scatter_(0, index, I[:, 2, :] - ratio * I[:, 3, :])

    ratio = x[:, 4, 3] / (x[:, 3, 3] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 4, :].scatter_(0, index, x[:, 4, :] - ratio * x[:, 3, :])
    I[:, 4, :].scatter_(0, index, I[:, 4, :] - ratio * I[:, 3, :])

    ratio = x[:, 5, 3] / (x[:, 3, 3] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 5, :].scatter_(0, index, x[:, 5, :] - ratio * x[:, 3, :])
    I[:, 5, :].scatter_(0, index, I[:, 5, :] - ratio * I[:, 3, :])

    ratio = x[:, 6, 3] / (x[:, 3, 3] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 6, :].scatter_(0, index, x[:, 6, :] - ratio * x[:, 3, :])
    I[:, 6, :].scatter_(0, index, I[:, 6, :] - ratio * I[:, 3, :])

    ratio = x[:, 7, 3] / (x[:, 3, 3] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 7, :].scatter_(0, index, x[:, 7, :] - ratio * x[:, 3, :])
    I[:, 7, :].scatter_(0, index, I[:, 7, :] - ratio * I[:, 3, :])

    ratio = x[:, 0, 4] / (x[:, 4, 4] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 0, :].scatter_(0, index, x[:, 0, :] - ratio * x[:, 4, :])
    I[:, 0, :].scatter_(0, index, I[:, 0, :] - ratio * I[:, 4, :])

    ratio = x[:, 1, 4] / (x[:, 4, 4] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 1, :].scatter_(0, index, x[:, 1, :] - ratio * x[:, 4, :])
    I[:, 1, :].scatter_(0, index, I[:, 1, :] - ratio * I[:, 4, :])

    ratio = x[:, 2, 4] / (x[:, 4, 4] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 2, :].scatter_(0, index, x[:, 2, :] - ratio * x[:, 4, :])
    I[:, 2, :].scatter_(0, index, I[:, 2, :] - ratio * I[:, 4, :])

    ratio = x[:, 3, 4] / (x[:, 4, 4] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 3, :].scatter_(0, index, x[:, 3, :] - ratio * x[:, 4, :])
    I[:, 3, :].scatter_(0, index, I[:, 3, :] - ratio * I[:, 4, :])

    ratio = x[:, 5, 4] / (x[:, 4, 4] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 5, :].scatter_(0, index, x[:, 5, :] - ratio * x[:, 4, :])
    I[:, 5, :].scatter_(0, index, I[:, 5, :] - ratio * I[:, 4, :])

    ratio = x[:, 6, 4] / (x[:, 4, 4] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 6, :].scatter_(0, index, x[:, 6, :] - ratio * x[:, 4, :])
    I[:, 6, :].scatter_(0, index, I[:, 6, :] - ratio * I[:, 4, :])

    ratio = x[:, 7, 4] / (x[:, 4, 4] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 7, :].scatter_(0, index, x[:, 7, :] - ratio * x[:, 4, :])
    I[:, 7, :].scatter_(0, index, I[:, 7, :] - ratio * I[:, 4, :])

    ratio = x[:, 0, 5] / (x[:, 5, 5] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 0, :].scatter_(0, index, x[:, 0, :] - ratio * x[:, 5, :])
    I[:, 0, :].scatter_(0, index, I[:, 0, :] - ratio * I[:, 5, :])

    ratio = x[:, 1, 5] / (x[:, 5, 5] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 1, :].scatter_(0, index, x[:, 1, :] - ratio * x[:, 5, :])
    I[:, 1, :].scatter_(0, index, I[:, 1, :] - ratio * I[:, 5, :])

    ratio = x[:, 2, 5] / (x[:, 5, 5] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 2, :].scatter_(0, index, x[:, 2, :] - ratio * x[:, 5, :])
    I[:, 2, :].scatter_(0, index, I[:, 2, :] - ratio * I[:, 5, :])

    ratio = x[:, 3, 5] / (x[:, 5, 5] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 3, :].scatter_(0, index, x[:, 3, :] - ratio * x[:, 5, :])
    I[:, 3, :].scatter_(0, index, I[:, 3, :] - ratio * I[:, 5, :])

    ratio = x[:, 4, 5] / (x[:, 5, 5] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 4, :].scatter_(0, index, x[:, 4, :] - ratio * x[:, 5, :])
    I[:, 4, :].scatter_(0, index, I[:, 4, :] - ratio * I[:, 5, :])

    ratio = x[:, 6, 5] / (x[:, 5, 5] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 6, :].scatter_(0, index, x[:, 6, :] - ratio * x[:, 5, :])
    I[:, 6, :].scatter_(0, index, I[:, 6, :] - ratio * I[:, 5, :])

    ratio = x[:, 7, 5] / (x[:, 5, 5] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 7, :].scatter_(0, index, x[:, 7, :] - ratio * x[:, 5, :])
    I[:, 7, :].scatter_(0, index, I[:, 7, :] - ratio * I[:, 5, :])

    ratio = x[:, 0, 6] / (x[:, 6, 6] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 0, :].scatter_(0, index, x[:, 0, :] - ratio * x[:, 6, :])
    I[:, 0, :].scatter_(0, index, I[:, 0, :] - ratio * I[:, 6, :])

    ratio = x[:, 1, 6] / (x[:, 6, 6] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 1, :].scatter_(0, index, x[:, 1, :] - ratio * x[:, 6, :])
    I[:, 1, :].scatter_(0, index, I[:, 1, :] - ratio * I[:, 6, :])

    ratio = x[:, 2, 6] / (x[:, 6, 6] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 2, :].scatter_(0, index, x[:, 2, :] - ratio * x[:, 6, :])
    I[:, 2, :].scatter_(0, index, I[:, 2, :] - ratio * I[:, 6, :])

    ratio = x[:, 3, 6] / (x[:, 6, 6] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 3, :].scatter_(0, index, x[:, 3, :] - ratio * x[:, 6, :])
    I[:, 3, :].scatter_(0, index, I[:, 3, :] - ratio * I[:, 6, :])

    ratio = x[:, 4, 6] / (x[:, 6, 6] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 4, :].scatter_(0, index, x[:, 4, :] - ratio * x[:, 6, :])
    I[:, 4, :].scatter_(0, index, I[:, 4, :] - ratio * I[:, 6, :])

    ratio = x[:, 5, 6] / (x[:, 6, 6] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 5, :].scatter_(0, index, x[:, 5, :] - ratio * x[:, 6, :])
    I[:, 5, :].scatter_(0, index, I[:, 5, :] - ratio * I[:, 6, :])

    ratio = x[:, 7, 6] / (x[:, 6, 6] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 7, :].scatter_(0, index, x[:, 7, :] - ratio * x[:, 6, :])
    I[:, 7, :].scatter_(0, index, I[:, 7, :] - ratio * I[:, 6, :])

    ratio = x[:, 0, 7] / (x[:, 7, 7] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 0, :].scatter_(0, index, x[:, 0, :] - ratio * x[:, 7, :])
    I[:, 0, :].scatter_(0, index, I[:, 0, :] - ratio * I[:, 7, :])

    ratio = x[:, 1, 7] / (x[:, 7, 7] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 1, :].scatter_(0, index, x[:, 1, :] - ratio * x[:, 7, :])
    I[:, 1, :].scatter_(0, index, I[:, 1, :] - ratio * I[:, 7, :])

    ratio = x[:, 2, 7] / (x[:, 7, 7] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 2, :].scatter_(0, index, x[:, 2, :] - ratio * x[:, 7, :])
    I[:, 2, :].scatter_(0, index, I[:, 2, :] - ratio * I[:, 7, :])

    ratio = x[:, 3, 7] / (x[:, 7, 7] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 3, :].scatter_(0, index, x[:, 3, :] - ratio * x[:, 7, :])
    I[:, 3, :].scatter_(0, index, I[:, 3, :] - ratio * I[:, 7, :])

    ratio = x[:, 4, 7] / (x[:, 7, 7] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 4, :].scatter_(0, index, x[:, 4, :] - ratio * x[:, 7, :])
    I[:, 4, :].scatter_(0, index, I[:, 4, :] - ratio * I[:, 7, :])

    ratio = x[:, 5, 7] / (x[:, 7, 7] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 5, :].scatter_(0, index, x[:, 5, :] - ratio * x[:, 7, :])
    I[:, 5, :].scatter_(0, index, I[:, 5, :] - ratio * I[:, 7, :])

    ratio = x[:, 6, 7] / (x[:, 7, 7] + 1e-8)
    ratio = ratio.view(bs, 1)
    x[:, 6, :].scatter_(0, index, x[:, 6, :] - ratio * x[:, 7, :])
    I[:, 6, :].scatter_(0, index, I[:, 6, :] - ratio * I[:, 7, :])

    # ratio = x[:, 1, 0] / (x[:, 0, 0] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 1] = x[:, 1] - ratio * x[:, 0]
    # I[:, 1] = I[:, 1] - ratio * I[:, 0]
    #
    # ratio = x[:, 2, 0] / (x[:, 0, 0] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 2] = x[:, 2] - ratio * x[:, 0]
    # I[:, 2] = I[:, 2] - ratio * I[:, 0]
    #
    # ratio = x[:, 3, 0] / (x[:, 0, 0] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 3] = x[:, 3] - ratio * x[:, 0]
    # I[:, 3] = I[:, 3] - ratio * I[:, 0]
    #
    # ratio = x[:, 4, 0] / (x[:, 0, 0] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 4] = x[:, 4] - ratio * x[:, 0]
    # I[:, 4] = I[:, 4] - ratio * I[:, 0]
    #
    # ratio = x[:, 5, 0] / (x[:, 0, 0] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 5] = x[:, 5] - ratio * x[:, 0]
    # I[:, 5] = I[:, 5] - ratio * I[:, 0]
    #
    # ratio = x[:, 6, 0] / (x[:, 0, 0] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 6] = x[:, 6] - ratio * x[:, 0]
    # I[:, 6] = I[:, 6] - ratio * I[:, 0]
    #
    # ratio = x[:, 7, 0] / (x[:, 0, 0] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 7] = x[:, 7] - ratio * x[:, 0]
    # I[:, 7] = I[:, 7] - ratio * I[:, 0]
    #
    # ratio = x[:, 0, 1] / (x[:, 1, 1] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 0] = x[:, 0] - ratio * x[:, 1]
    # I[:, 0] = I[:, 0] - ratio * I[:, 1]
    #
    # ratio = x[:, 2, 1] / (x[:, 1, 1] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 2] = x[:, 2] - ratio * x[:, 1]
    # I[:, 2] = I[:, 2] - ratio * I[:, 1]
    #
    # ratio = x[:, 3, 1] / (x[:, 1, 1] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 3] = x[:, 3] - ratio * x[:, 1]
    # I[:, 3] = I[:, 3] - ratio * I[:, 1]
    #
    # ratio = x[:, 4, 1] / (x[:, 1, 1] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 4] = x[:, 4] - ratio * x[:, 1]
    # I[:, 4] = I[:, 4] - ratio * I[:, 1]
    #
    # ratio = x[:, 5, 1] / (x[:, 1, 1] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 5] = x[:, 5] - ratio * x[:, 1]
    # I[:, 5] = I[:, 5] - ratio * I[:, 1]
    #
    # ratio = x[:, 6, 1] / (x[:, 1, 1] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 6] = x[:, 6] - ratio * x[:, 1]
    # I[:, 6] = I[:, 6] - ratio * I[:, 1]
    #
    # ratio = x[:, 7, 1] / (x[:, 1, 1] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 7] = x[:, 7] - ratio * x[:, 1]
    # I[:, 7] = I[:, 7] - ratio * I[:, 1]
    #
    # ratio = x[:, 0, 2] / (x[:, 2, 2] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 0] = x[:, 0] - ratio * x[:, 2]
    # I[:, 0] = I[:, 0] - ratio * I[:, 2]
    #
    # ratio = x[:, 1, 2] / (x[:, 2, 2] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 1] = x[:, 1] - ratio * x[:, 2]
    # I[:, 1] = I[:, 1] - ratio * I[:, 2]
    #
    # ratio = x[:, 3, 2] / (x[:, 2, 2] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 3] = x[:, 3] - ratio * x[:, 2]
    # I[:, 3] = I[:, 3] - ratio * I[:, 2]
    #
    # ratio = x[:, 4, 2] / (x[:, 2, 2] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 4] = x[:, 4] - ratio * x[:, 2]
    # I[:, 4] = I[:, 4] - ratio * I[:, 2]
    #
    # ratio = x[:, 5, 2] / (x[:, 2, 2] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 5] = x[:, 5] - ratio * x[:, 2]
    # I[:, 5] = I[:, 5] - ratio * I[:, 2]
    #
    # ratio = x[:, 6, 2] / (x[:, 2, 2] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 6] = x[:, 6] - ratio * x[:, 2]
    # I[:, 6] = I[:, 6] - ratio * I[:, 2]
    #
    # ratio = x[:, 7, 2] / (x[:, 2, 2] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 7] = x[:, 7] - ratio * x[:, 2]
    # I[:, 7] = I[:, 7] - ratio * I[:, 2]
    #
    # ratio = x[:, 0, 3] / (x[:, 3, 3] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 0] = x[:, 0] - ratio * x[:, 3]
    # I[:, 0] = I[:, 0] - ratio * I[:, 3]
    #
    # ratio = x[:, 1, 3] / (x[:, 3, 3] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 1] = x[:, 1] - ratio * x[:, 3]
    # I[:, 1] = I[:, 1] - ratio * I[:, 3]
    #
    # ratio = x[:, 2, 3] / (x[:, 3, 3] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 2] = x[:, 2] - ratio * x[:, 3]
    # I[:, 2] = I[:, 2] - ratio * I[:, 3]
    #
    # ratio = x[:, 4, 3] / (x[:, 3, 3] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 4] = x[:, 4] - ratio * x[:, 3]
    # I[:, 4] = I[:, 4] - ratio * I[:, 3]
    #
    # ratio = x[:, 5, 3] / (x[:, 3, 3] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 5] = x[:, 5] - ratio * x[:, 3]
    # I[:, 5] = I[:, 5] - ratio * I[:, 3]
    #
    # ratio = x[:, 6, 3] / (x[:, 3, 3] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 6] = x[:, 6] - ratio * x[:, 3]
    # I[:, 6] = I[:, 6] - ratio * I[:, 3]
    #
    # ratio = x[:, 7, 3] / (x[:, 3, 3] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 7] = x[:, 7] - ratio * x[:, 3]
    # I[:, 7] = I[:, 7] - ratio * I[:, 3]
    #
    # ratio = x[:, 0, 4] / (x[:, 4, 4] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 0] = x[:, 0] - ratio * x[:, 4]
    # I[:, 0] = I[:, 0] - ratio * I[:, 4]
    #
    # ratio = x[:, 1, 4] / (x[:, 4, 4] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 1] = x[:, 1] - ratio * x[:, 4]
    # I[:, 1] = I[:, 1] - ratio * I[:, 4]
    #
    # ratio = x[:, 2, 4] / (x[:, 4, 4] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 2] = x[:, 2] - ratio * x[:, 4]
    # I[:, 2] = I[:, 2] - ratio * I[:, 4]
    #
    # ratio = x[:, 3, 4] / (x[:, 4, 4] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 3] = x[:, 3] - ratio * x[:, 4]
    # I[:, 3] = I[:, 3] - ratio * I[:, 4]
    #
    # ratio = x[:, 5, 4] / (x[:, 4, 4] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 5] = x[:, 5] - ratio * x[:, 4]
    # I[:, 5] = I[:, 5] - ratio * I[:, 4]
    #
    # ratio = x[:, 6, 4] / (x[:, 4, 4] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 6] = x[:, 6] - ratio * x[:, 4]
    # I[:, 6] = I[:, 6] - ratio * I[:, 4]
    #
    # ratio = x[:, 7, 4] / (x[:, 4, 4] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 7] = x[:, 7] - ratio * x[:, 4]
    # I[:, 7] = I[:, 7] - ratio * I[:, 4]
    #
    # ratio = x[:, 0, 5] / (x[:, 5, 5] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 0] = x[:, 0] - ratio * x[:, 5]
    # I[:, 0] = I[:, 0] - ratio * I[:, 5]
    #
    # ratio = x[:, 1, 5] / (x[:, 5, 5] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 1] = x[:, 1] - ratio * x[:, 5]
    # I[:, 1] = I[:, 1] - ratio * I[:, 5]
    #
    # ratio = x[:, 2, 5] / (x[:, 5, 5] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 2] = x[:, 2] - ratio * x[:, 5]
    # I[:, 2] = I[:, 2] - ratio * I[:, 5]
    #
    # ratio = x[:, 3, 5] / (x[:, 5, 5] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 3] = x[:, 3] - ratio * x[:, 5]
    # I[:, 3] = I[:, 3] - ratio * I[:, 5]
    #
    # ratio = x[:, 4, 5] / (x[:, 5, 5] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 4] = x[:, 4] - ratio * x[:, 5]
    # I[:, 4] = I[:, 4] - ratio * I[:, 5]
    #
    # ratio = x[:, 6, 5] / (x[:, 5, 5] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 6] = x[:, 6] - ratio * x[:, 5]
    # I[:, 6] = I[:, 6] - ratio * I[:, 5]
    #
    # ratio = x[:, 7, 5] / (x[:, 5, 5] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 7] = x[:, 7] - ratio * x[:, 5]
    # I[:, 7] = I[:, 7] - ratio * I[:, 5]
    #
    # ratio = x[:, 0, 6] / (x[:, 6, 6] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 0] = x[:, 0] - ratio * x[:, 6]
    # I[:, 0] = I[:, 0] - ratio * I[:, 6]
    #
    # ratio = x[:, 1, 6] / (x[:, 6, 6] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 1] = x[:, 1] - ratio * x[:, 6]
    # I[:, 1] = I[:, 1] - ratio * I[:, 6]
    #
    # ratio = x[:, 2, 6] / (x[:, 6, 6] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 2] = x[:, 2] - ratio * x[:, 6]
    # I[:, 2] = I[:, 2] - ratio * I[:, 6]
    #
    # ratio = x[:, 3, 6] / (x[:, 6, 6] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 3] = x[:, 3] - ratio * x[:, 6]
    # I[:, 3] = I[:, 3] - ratio * I[:, 6]
    #
    # ratio = x[:, 4, 6] / (x[:, 6, 6] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 4] = x[:, 4] - ratio * x[:, 6]
    # I[:, 4] = I[:, 4] - ratio * I[:, 6]
    #
    # ratio = x[:, 5, 6] / (x[:, 6, 6] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 5] = x[:, 5] - ratio * x[:, 6]
    # I[:, 5] = I[:, 5] - ratio * I[:, 6]
    #
    # ratio = x[:, 7, 6] / (x[:, 6, 6] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 7] = x[:, 7] - ratio * x[:, 6]
    # I[:, 7] = I[:, 7] - ratio * I[:, 6]
    #
    # ratio = x[:, 0, 7] / (x[:, 7, 7] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 0] = x[:, 0] - ratio * x[:, 7]
    # I[:, 0] = I[:, 0] - ratio * I[:, 7]
    #
    # ratio = x[:, 1, 7] / (x[:, 7, 7] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 1] = x[:, 1] - ratio * x[:, 7]
    # I[:, 1] = I[:, 1] - ratio * I[:, 7]
    #
    # ratio = x[:, 2, 7] / (x[:, 7, 7] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 2] = x[:, 2] - ratio * x[:, 7]
    # I[:, 2] = I[:, 2] - ratio * I[:, 7]
    #
    # ratio = x[:, 3, 7] / (x[:, 7, 7] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 3] = x[:, 3] - ratio * x[:, 7]
    # I[:, 3] = I[:, 3] - ratio * I[:, 7]
    #
    # ratio = x[:, 4, 7] / (x[:, 7, 7] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 4] = x[:, 4] - ratio * x[:, 7]
    # I[:, 4] = I[:, 4] - ratio * I[:, 7]
    #
    # ratio = x[:, 5, 7] / (x[:, 7, 7] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 5] = x[:, 5] - ratio * x[:, 7]
    # I[:, 5] = I[:, 5] - ratio * I[:, 7]
    #
    # ratio = x[:, 6, 7] / (x[:, 7, 7] + 1e-8)
    # ratio = ratio.view(bs, 1)
    # x[:, 6] = x[:, 6] - ratio * x[:, 7]
    # I[:, 6] = I[:, 6] - ratio * I[:, 7]

    diag = torch.stack([x[:, i, i] for i in range(8)], dim=-1).view(bs, 8, 1)
    I = I / diag

    return I


# def cof1(M, index):
#     zs = M[:index[0] - 1, :index[1] - 1]
#     ys = M[:index[0] - 1, index[1]:]
#     zx = M[index[0]:, :index[1] - 1]
#     yx = M[index[0]:, index[1]:]
#     s = torch.cat((zs, ys), axis=1)
#     x = torch.cat((zx, yx), axis=1)
#     return det(torch.cat((s, x), axis=0))
#
#
# def alcof(M, index):
#     return pow(-1, index[0] + index[1]) * cof1(M, index)
#
#
# def adj(M):
#     result = torch.zeros((M.shape[0], M.shape[1]))
#     for i in range(1, M.shape[0] + 1):
#         for j in range(1, M.shape[1] + 1):
#             result[j - 1][i - 1] = alcof(M, [i, j])
#     return result
#
#
# def invmat(M):
#     return 1.0 / det(M) * adj(M)


class TPS:
    '''
    TPS transformation, mode 'kp' for Eq(2) in the paper, mode 'random' for equivariance loss.
    '''

    def __init__(self, mode, bs, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == 'random':
            noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
            self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        elif mode == 'kp':
            kp_1 = kwargs["kp_1"]
            kp_2 = kwargs["kp_2"]
            device = kp_1.device
            kp_type = kp_1.type()
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            K = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2)
            K = K ** 2
            K = K * torch.log(K + 1e-9)

            one1 = torch.ones(self.bs, kp_1.shape[1], kp_1.shape[2], 1).to(device).type(kp_type)
            kp_1p = torch.cat([kp_1, one1], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 3).to(device).type(kp_type)
            P = torch.cat([kp_1p, zero], 2)

            L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
            L = torch.cat([L, P], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 2).to(device).type(kp_type)
            Y = torch.cat([kp_2, zero], 2)
            one = torch.eye(L.shape[2]).expand(L.shape).to(device).type(kp_type) * 0.01
            L = L + one

            # param = torch.matmul(torch.inverse(L), Y)
            param = torch.matmul(torch.stack([inverse_Gauss_Jordan(L[0, i]) for i in range(10)], 0).unsqueeze(0), Y)
            # param = torch.matmul(torch.stack([invmat(L[0, i]) for i in range(10)], 0).unsqueeze(0), Y)

            self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)

            self.control_points = kp_1
            self.control_params = param[:, :, :n, :]
        else:
            raise Exception("Error TPS mode")

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0).to(frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == 'kp':
            shape.insert(1, self.gs)
        grid = self.warp_coordinates(grid).view(*shape)
        return grid

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        control_points = self.control_points.type(coordinates.type()).to(coordinates.device)
        control_params = self.control_params.type(coordinates.type()).to(coordinates.device)

        if self.mode == 'kp':
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]

            distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - control_points.view(self.bs,
                                                                                                  control_points.shape[
                                                                                                      1], -1, 1, 2)

            distances = distances ** 2
            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            transformed = transformed.permute(0, 1, 3, 2) + result

        elif self.mode == 'random':
            theta = theta.unsqueeze(1)
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
            transformed = transformed.squeeze(-1)
            ances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = ances ** 2

            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        else:
            raise Exception("Error TPS mode")

        return transformed


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """

    coordinate_grid = make_coordinate_grid(spatial_size, kp.type()).to(kp.device)
    number_of_leading_dimensions = len(kp.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = kp.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = kp.shape[:number_of_leading_dimensions] + (1, 1, 2)
    kp = kp.view(*shape)

    mean_sub = (coordinate_grid - kp)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = nn.InstanceNorm2d(in_features, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        # print('encoder:' ,outs[-1].shape)
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
            # print('encoder:' ,outs[-1].shape)
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []
        self.out_channels = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_channels.append(block_expansion + in_features)
        # self.out_filters = block_expansion + in_features

    def forward(self, x, mode=0):
        out = x.pop()
        outs = []
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            outs.append(out)
        if mode == 0:
            return out
        else:
            return outs


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_channels = self.decoder.out_channels
        # self.out_filters = self.decoder.out_filters

    def forward(self, x, mode=0):
        return self.decoder(self.encoder(x), mode)


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out


def to_homogeneous(coordinates):
    ones_shape = list(coordinates.shape)
    ones_shape[-1] = 1
    ones = torch.ones(ones_shape).type(coordinates.type())

    return torch.cat([coordinates, ones], dim=-1)


def from_homogeneous(coordinates):
    return coordinates[..., :2] / coordinates[..., 2:3]


def warp_coordinates(coordinates, kp_1, kp_2):
    n = kp_1.shape[2]
    kpt_10 = kp_1.view(1, 10, 5, 1, 2)
    kpt_11 = kp_1.view(1, 10, 1, 5, 2)
    K_norm = torch.norm(kpt_10 - kpt_11, dim=4, p=2)
    K_pow = torch.pow(K_norm, 2)
    K = K_pow * torch.log(K_pow + 1e-9)

    # one1 = torch.ones(1, kp_1.shape[1], kp_1.shape[2], 1)
    # kp_1p = torch.cat([kp_1, one1], 3)
    kp_1p = torch.nn.functional.pad(kp_1, (0, 1, 0, 0), value=1)

    # zero = torch.zeros(1, kp_1.shape[1], 3, 3)
    # P = torch.cat([kp_1p, zero], 2)

    P = torch.nn.functional.pad(kp_1p, (0, 0, 0, 3), value=0)

    L_cat = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
    LP = torch.cat([L_cat, P], 3)

    zero = torch.zeros(1, 10, 3, 2)
    Y = torch.cat([kp_2, zero], 2)
    one = torch.eye(8).expand([1, 10, 8, 8]) * 0.01
    LP += one

    # param = torch.matmul(torch.inverse(L), Y)
    LP_inv = inverse_Gauss_Jordan(LP[0]).unsqueeze(0)

    param = torch.matmul(LP_inv, Y)
    # param = torch.matmul(torch.stack([inverse_Gauss_Jordan(LP[0, i]) for i in range(10)], 0).unsqueeze(0), Y)
    # param = torch.matmul(torch.stack([invmat(LP[0, i]) for i in range(10)], 0).unsqueeze(0), Y)

    control_params = param[:, :, :n, :]
    theta = param[:, :, n:, :].permute(0, 1, 3, 2)

    transformed = torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]

    # distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - kp_1.view(1, kp_1.shape[1], -1, 1, 2)
    distances = coordinates.view(1, 1, 1, 4096, 2) - kpt_10

    distances = torch.pow(distances, 2)
    result = distances.sum(-1)
    result = result * torch.log(result + 1e-9)
    result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
    transformed = transformed.permute(0, 1, 3, 2) + result

    return transformed


def transform_frame(grid, kp_1, kp_2):
    return warp_coordinates(grid, kp_1, kp_2).view(1, 10, 64, 64, 2)


if __name__ == "__main__":
    import multiprocessing
    import onnxruntime


    class Test(torch.nn.Module):
        def __init__(self):
            super(Test, self).__init__()

        def forward(self, grid, kp_1, kp_2):
            return transform_frame(grid, kp_1, kp_2)
        # def forward(self ,x):
        #     return inverse_Gauss_Jordan(x)


    torch.manual_seed(1215)
    model = Test()
    dummy_kp1 = torch.randn(1, 10, 5, 2)
    dummy_kp2 = torch.randn(1, 10, 5, 2)
    dummy_kp3 = torch.randn(10, 8, 8)
    grid = make_coordinate_grid([64, 64], type=torch.FloatTensor).unsqueeze(0)
    grid = grid.view(1, 4096, 2)
    torch.onnx.export(
        model,
        (grid, dummy_kp1, dummy_kp2),
        "tps.onnx",
        input_names=["grid", "kp1", "kp2"],
        output_names=["output_grid"],
        opset_version=11,
    )

    # torch.onnx.export(
    #     model,
    #     (dummy_kp1),
    #     "tps.onnx",
    #     input_names=["kp1"],
    #     output_names=["output_grid"],
    #     opset_version=11,
    # )

    # torch.onnx.export(
    #     model,
    #     (dummy_kp3),
    #     "tps.onnx",
    #     input_names=["kp3"],
    #     output_names=["output_grid"],
    #     opset_version=11,
    # )

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    onnx_model = onnxruntime.InferenceSession("tps.onnx", sess_options, providers=['CPUExecutionProvider'])

    ort_inputs = {
        onnx_model.get_inputs()[0].name: grid.numpy(),
        onnx_model.get_inputs()[1].name: dummy_kp1.numpy(),
        onnx_model.get_inputs()[2].name: dummy_kp2.numpy()}
    # ort_inputs = {onnx_model.get_inputs()[0].name: dummy_kp3.numpy()}
    onnx_output = onnx_model.run([onnx_model.get_outputs()[0].name], ort_inputs)[0]  # 1, 50, 2

    torch_output = model(grid, dummy_kp1, dummy_kp2).detach().numpy()
    # torch_output = model(dummy_kp3).detach().numpy()

    print(torch_output.flatten())
    print(onnx_output.flatten())
    # print(torch_output.flatten() - onnx_output.flatten())
    np.testing.assert_almost_equal(torch_output, onnx_output, decimal=4)
    print("ok")
