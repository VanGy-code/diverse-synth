from abc import abstractmethod
from typing import List, Any, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch import nn


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Networks
    """

    def __init__(self, latent_dim, reduction=4):
        super(SELayer, self).__init__()
        num_hidden = max(latent_dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(latent_dim, num_hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(num_hidden, latent_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(B, C, 1)
        return x * y.expand_as(x)

class ResidualLinearBlock(nn.Module):
    def __init__(self, dim, linear_dim, dim_nums, bn_momentum=0.01):
        super(ResidualLinearBlock, self).__init__()

        self.linear_dim = linear_dim
        self.dim_nums = dim_nums

        self.seq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, momentum=bn_momentum),
            Swish(),
            nn.Linear(dim, dim)
        )
        self.se_layer = SELayer(dim)

    def forward(self, x):
        B, _, _ = x.shape
        y = self.seq(
            x.permute(0, 2, 1).reshape(B * self.dim_nums, self.linear_dim)
        ).reshape(B, self.dim_nums, self.linear_dim).permute(0, 2, 1)
        return x + 0.1 * self.se_layer(y)

class DecoderResidualBlock(nn.Module):
    def __init__(self, dim, n_group, bn_momentum=0.01):
        super(DecoderResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, n_group * dim)
        self.bn1 = nn.BatchNorm1d(n_group * dim, momentum=bn_momentum)
        self.swish1 = Swish()
   
        self.fc2 = nn.Linear(n_group * dim, n_group * dim)
        self.bn2 = nn.BatchNorm1d(n_group * dim, momentum=bn_momentum)
        self.swish2 = Swish()
        self.fc3 = nn.Linear(n_group * dim, dim)

        self.bn3 = nn.BatchNorm1d(dim, momentum=bn_momentum)
        self.se_layer = SELayer(dim)


    def forward(self, x):

        y = self.swish1(self.bn1(self.fc1(x)))

        y = self.swish2(self.bn2(self.fc2(y)))

        y = self.se_layer(self.bn3(self.fc3(y)))
        return x + 0.1 * y


class BaseUNet(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def class2angle(angle_class, residual, num_angle_class):
        """
        Inverse function to angle2class
        :param angle_class:
        :param residual: residual = real_angle - angle_class * num_angle_class
        :param num_angle_class: the number of angle classes
        """
        angle_per_class = 2 * np.pi / float(num_angle_class)
        angle_center = angle_class * angle_per_class
        angle = angle_center + residual
        return angle

    @staticmethod
    def parse_batch(x):
        B, N = x.shape[0], x.shape[1]
        n1 = torch.zeros((B, N, 2)).to(x)
        angle_class = torch.argmax(x[:, :, :8], dim=2)
        angle = BaseUNet.class2angle(angle_class, x[:, :, 8], num_angle_class=8)
        n1[:, :, 0] = torch.cos(angle)
        n1[:, :, 1] = torch.sin(angle)

        x_parsed = torch.cat((n1, x[:, :, 9:]), dim=2)
        assert (x_parsed.shape == torch.Size([B, N, 9]))

        return x_parsed

    @staticmethod
    def disc2translation(Iij, class_id, residual, interval):
        sij = (Iij - 0.5) * 2  # 1 or -1
        tij = sij * (class_id * interval + residual)
        return tij

    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Any) -> Tensor:
        raise NotImplementedError


class BaseVAE(nn.Module):

    def __init__(self):
        super(BaseVAE, self).__init__()

    @staticmethod
    def class2angle(pred_cls, residual, num_class):
        """
        Inverse function to angle to class
        :param pred_cls: (same_shape)
        :param residual: (same_shape)
        :param num_class: num of class (default: 32)

        :return angle: (same_shape). 0~2pi
        """

        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center
        return angle

    @staticmethod
    def linear_assignment_class(distance_matrix, row_counts=None, col_masks=None):
        batch_size = distance_matrix.shape[0]
        num_class = distance_matrix.shape[1]
        num_each_class = distance_matrix.shape[2]

        batch_ind = []
        row_ind_list = []
        col_ind_list = []
        for i in range(batch_size):
            for j in range(num_class):
                distance_mat = distance_matrix[i, j, :, :]
                if row_counts is not None:
                    distance_mat = distance_mat[:row_counts[i, j], :]
                if col_masks is not None:
                    col_idx = torch.nonzero(col_masks[i, j])[:, 0]
                    distance_mat = distance_mat[:, col_idx]

                row_ind, col_ind = linear_sum_assignment(distance_mat.detach().to('cpu').numpy())
                row_ind = list(row_ind + num_each_class * j)
                if col_masks is None:
                    col_ind = list(col_ind + num_each_class * j)
                else:
                    col_ind = list(col_idx[col_ind] + num_each_class * j)

                batch_ind += [i] * len(row_ind)
                row_ind_list += row_ind
                col_ind_list += col_ind

        return batch_ind, row_ind_list, col_ind_list

    @abstractmethod
    def sample(self, latent_dim: Tensor) -> Tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x) -> List[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Any) -> Tensor:
        raise NotImplementedError


class BaseConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn_momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, 1),
            nn.BatchNorm2d(output_dim, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class LeakyReLUConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn_momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, 1),
            nn.BatchNorm2d(output_dim, momentum=bn_momentum),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=3, bn_momentum=0.01):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(output_dim, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class LeakyReLUTransposeConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=3, bn_momentum=0.01):
        super(LeakyReLUTransposeConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(output_dim, momentum=bn_momentum),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.up(x)
