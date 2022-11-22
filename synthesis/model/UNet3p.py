from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from synthesis.model.Common import BaseUNet


class UNetConv2(nn.Module):
    def __init__(self, input_dim, output_dim, is_batch_norm=True, kernel_size=3, stride=1, padding=1, n=1):
        super().__init__()
        self.conv = nn.ModuleList()
        for i in range(1, n + 1):
            if is_batch_norm:
                conv = nn.Sequential(
                    nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
                    nn.ReLU(inplace=True),
                )
            self.conv.append(conv)
            input_dim = output_dim

        try:
            self.reset_parameters()
        except RuntimeError:
            raise RuntimeError

    def reset_parameters(self):
        for m in self.conv.children():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('BatchNorm') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        for i in range(len(self.conv)):
            x = self.conv[i](x)
        return x


class UNetUp(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, is_deconv=False, n_concat=2):
        super().__init__()
        self.conv = UNetConv2(output_dim * n_concat, output_dim, False, kernel_size=kernel_size, stride=stride, padding=padding, n=2)

        if is_deconv:
            self.up = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        try:
            self.reset_parameters()
        except RuntimeError:
            raise RuntimeError

    def reset_parameters(self):
        for m in self.children():
            classname = m.__class__.__name__
            if classname.find('UNetConv2') != -1:
                continue
            elif classname.find('Conv') != -1:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('BatchNorm') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, input0, *inputs):
        # x = self.up(x)
        x = self.up(input0)
        for i in range(len(inputs)):
            x = torch.cat([x, inputs[i]], dim=1)
        return self.conv(x)


class UNet3Plus(BaseUNet):
    def __init__(self, data_config, model_config):
        super().__init__()
        self.input_dim = model_config["input_dim"]
        self.num_class = data_config["num_class"]
        self.num_each_class = data_config["num_each_class"]
        self.half_range = data_config["half_range"]
        self.interval = data_config["interval"]
        self.bn_momentum = model_config["bn_momentum"]
        self.num_bins = int(data_config["half_range"] / data_config["interval"] + 1)
        self.output_dim = 2 + 2 * self.num_bins + 2 + 1 + 3 + 1 + 1 + 1

        # -------------------------- Encoder ------------------------------

        self.conv1 = UNetConv2(
            input_dim=2 * (9 + self.num_class),
            output_dim=64,
            is_batch_norm=True
        )

        self.conv2 = UNetConv2(
            input_dim=64,
            output_dim=128,
            is_batch_norm=True
        )
        
        self.conv3 = UNetConv2(
            input_dim=128,
            output_dim=256,
            is_batch_norm=True
        )

        self.pool3 = nn.MaxPool2d(self.num_each_class, stride=self.num_each_class)

        self.conv4 = UNetConv2(
            input_dim=256,
            output_dim=256,
            is_batch_norm=True,
            n=2
        )

        self.pool4 = nn.MaxPool2d(self.num_class, stride=self.num_class)

        self.conv5 = nn.Sequential(
            UNetConv2(
                input_dim=256,
                output_dim=512,
                kernel_size=1,
                stride=1,
                padding=0,
                is_batch_norm=True,
            ),
            UNetConv2(
                input_dim=512,
                output_dim=256,
                kernel_size=1,
                stride=1,
                padding=0,
                is_batch_norm=True,
            )
        )
        # --------------------------- Decoder ------------------------------
        self.cat_channels = 128
        self.cat_blocks = 3
        self.up_channels = self.cat_channels * self.cat_blocks

        '''stage 4d'''
        # h3 -> (92, 92), hd4 -> (23, 23), pooling 4
        self.h3_pt_hd4 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(256, self.cat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.cat_channels),
            nn.ReLU(inplace=True)
        )

        # h4 -> (23, 23), hd4 -> (23, 23), concatenation
        self.h4_cat_hd4 = nn.Sequential(
            nn.Conv2d(256, self.cat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.cat_channels),
            nn.ReLU(inplace=True)
        )

        # hd5 -> (1 * 1), hd4 -> (23, 23), UpSample
        self.hd5_ut_hd4 = nn.Sequential(
            nn.Upsample(size=(23, 23), mode='bilinear'),
            nn.Conv2d(256, self.cat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.cat_channels),
            nn.ReLU(inplace=True)
        )

        # fusion all
        self.stage_4 = nn.Sequential(
            # Neck
            nn.Conv2d(self.up_channels, self.up_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.up_channels),
            nn.ReLU(inplace=True)
        )

        '''stage 3d'''
        # h3 -> (92, 92), hd3 -> (92, 92), concatenation
        self.h3_cat_hd3 = nn.Sequential(
            nn.Conv2d(256, self.cat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.cat_channels),
            nn.ReLU(inplace=True)
        )

        # hd4 -> (23, 23), hd3 -> (92, 92), UpSample
        self.h4d_ut_hd3 = nn.Sequential(
            nn.Upsample(size=(92, 92), mode='bilinear'),
            nn.Conv2d(self.up_channels, self.cat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.cat_channels),
            nn.ReLU(inplace=True)
        )

        # hd5 -> (1 * 1), hd3 -> (92, 92), UpSample
        self.hd5_ut_hd3 = nn.Sequential(
            nn.Upsample(size=(92, 92), mode='bilinear'),
            nn.Conv2d(256, self.cat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.cat_channels),
            nn.ReLU(inplace=True)
        )

        # fusion all
        self.stage_3 = nn.Sequential(
            nn.Conv2d(self.up_channels, self.up_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.up_channels),
            nn.ReLU(inplace=True)
        )

        self.upscore2 = nn.Upsample(size=(92, 92), mode='bilinear')
        self.upscore3 = nn.Upsample(size=(92, 92), mode='bilinear')

        self.conv_out1 = nn.Conv2d(self.up_channels, self.output_dim, kernel_size=1, stride=1)
        self.conv_out2 = nn.Conv2d(self.up_channels, self.output_dim, kernel_size=1, stride=1)
        self.conv_out3 = nn.Conv2d(256, self.output_dim, kernel_size=1, stride=1)

        try:
            self.reset_parameters()
        except RuntimeError:
            raise RuntimeError

        # loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss(beta=0.5, reduction='mean', size_average=False)
        # BCEwithlogitsloss = BCELoss + Sigmoid
        self.binary_cross_entropy_with_logit_loss = nn.BCEWithLogitsLoss(reduction='none')

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.recombine_data(x)

        # -------------------------- Encoder ------------------------------
        # (B, 2 * (9 + num_class), 92, 92)  -> (B, 64，92, 92)
        x = self.conv1(x)

        # (B, 64 92, 92)  -> (B, 128，92, 92)
        x = self.conv2(x)

        # (B, 128 92, 92)  -> (B, 256，92, 92)
        x1 = self.conv3(x)

        # (B, 256 92, 92)  -> (B, 256，23, 23)
        x = self.pool3(x1)
        # (B, 256 23, 23)  -> (B, 256，23, 23)
        x2 = self.conv4(x)

        # (B, 256 23, 23)  -> (B, 256，1, 1)
        x = self.pool4(x2)
        # (B, 256 1, 1)  -> (B, 256，1, 1)
        x = self.conv5(x)

        # --------------------------- Decoder ------------------------------
        h3_pt_hd4 = self.h3_pt_hd4(x1)
        h4_cat_hd4 = self.h4_cat_hd4(x2)
        hd5_ut_hd4 = self.hd5_ut_hd4(x)
        # hd4 -> (B, UpChannels, 23, 23)
        x2 = self.stage_4(torch.cat((h3_pt_hd4, h4_cat_hd4, hd5_ut_hd4), 1))

        h3_cat_hd3 = self.h3_cat_hd3(x1)
        h4d_ut_hd3 = self.h4d_ut_hd3(x2)
        hd5_ut_hd3 = self.hd5_ut_hd3(x)
        # hd3 -> (B, UpChannels, 92, 92)
        x1 = self.stage_3(torch.cat((h3_cat_hd3, h4d_ut_hd3, hd5_ut_hd3), 1))

        # d -> (B, 53, 92, 92)
        dh = self.conv_out3(x)
        dh = self.upscore3(dh).permute(0, 2, 3, 1)

        d2 = self.conv_out2(x2)
        d2 = self.upscore2(d2).permute(0, 2, 3, 1)

        d1 = self.conv_out1(x1).permute(0, 2, 3, 1)

        # return self.post_process(x)
        return [self.post_process(d1), self.post_process(d2), self.post_process(dh)]

    def recombine_data(self, x):
        B, N = x.shape[0], x.shape[1]
        assert (x.shape[2] == 16)
        # (B, N, 9)
        x = self.parse_batch(x)
        assert (x.shape[2] == 9)

        # (B, N, num_class)
        related_class = torch.zeros((B, N, self.num_class)).to(x.device)
        # Each furniture is related to their own.
        for i in range(self.num_class):
            related_class[:, i * self.num_each_class: (i + 1) * self.num_each_class, i] = 1

        # (B, N, 9 + num_class)
        x = torch.cat((x, related_class), dim=2)

        # (B, N, N, 2 * (9 + num_class))
        x = torch.cat(
            (
                #
                x[:, :, None, :].repeat(1, 1, self.num_class * self.num_each_class, 1),
                x[:, None, :, :].repeat(1, self.num_class * self.num_each_class, 1, 1)
            ),
            dim=3
        )

        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def post_process(self, x):
        """
            :param x:
            :return output:
                # output_dim = 2 + 2 * self.num_bins + 2 + 1 + 3 + 8 + 1
                #     0 -- Ix
                #     1 -- Iy
                #     2:2+num_bins -- cx
                #     2+num_bins:2+2*num_bins -- cy
                #     -15 -- rx
                #     -14 -- ry
                #     -13 -- z
                #     -12:-9 -- rotation_class
                #     -9:-6 -- x_size
                #     -6:-3 -- y_size
                #     -3 -- same_size
                #     -2 -- rel_size
                #     -1 -- rel_indicator (deprecated)
                output_dim = 2 + 2 * self.num_bins + 2 + 1 + 3 + 1 + 1 + 1
                    0 -- Ix
                    1 -- Iy
                    2:2+num_bins -- cx
                    2+num_bins:2+2*num_bins -- cy
                    -9 -- rx
                    -8 -- ry
                    -7 -- z
                    -6:-3 -- rotation_class
                    -3 -- same_size
                    -2 -- rel_size
                    -1 -- rel_indicator (deprecated)
        """
        Ix = x[:, :, :, 0:1]
        Iy = x[:, :, :, 1:2]
        cx = x[:, :, :, 2: 2 + self.num_bins]
        cy = x[:, :, :, 2 + self.num_bins: 2 + 2 * self.num_bins]

        rx = x[:, :, :, -9:-8]
        ry = x[:, :, :, -8:-7]
        z = x[:, :, :, -7:-6]
        rotation_class = x[:, :, :, -6:-3]
        same_size = x[:, :, :, -3:-2]
        rel_size = x[:, :, :, -2:-1]

        mask = x[:, :, :, -1:]

        Ix = (Ix - Ix.permute(0, 2, 1, 3)) / 2
        Iy = (Iy - Iy.permute(0, 2, 1, 3)) / 2
        cx = (cx + cx.permute(0, 2, 1, 3)) / 2
        cy = (cy + cy.permute(0, 2, 1, 3)) / 2
        rx = (rx + rx.permute(0, 2, 1, 3)) / 2
        ry = (ry + ry.permute(0, 2, 1, 3)) / 2
        z = (z - z.permute(0, 2, 1, 3)) / 2
        rotation_class = (rotation_class + rotation_class.permute(0, 2, 1, 3)) / 2
        same_size = (same_size + same_size.permute(0, 2, 1, 3)) / 2

        output = torch.cat((Ix, Iy, cx, cy, rx, ry, z, rotation_class, same_size, rel_size, mask), dim=3)

        return output

    def loss_function(self, **kwargs: Any) -> dict:
        room_type = kwargs['room_type']
        ground_truth = kwargs['ground_truth']
        x = kwargs['reconstruct_x']
        batch_idx = kwargs['batch_idx']
        reconstruct_idx = kwargs['reconstruct_idx']
        ground_truth_idx = kwargs['ground_truth_idx']
        total_loss = kwargs['total_loss']
        loss_dict = {}

        batch_size, max_num_parts = ground_truth.shape[0], ground_truth.shape[1]
        assert (max_num_parts == self.num_class * self.num_each_class)

        error_indicator_x = torch.zeros(1).to(ground_truth.device)
        error_indicator_y = torch.zeros(1).to(ground_truth.device)

        error_translation_class_x = torch.zeros(1).to(ground_truth.device)
        error_translation_class_y = torch.zeros(1).to(ground_truth.device)
        error_translation_residual_x = torch.zeros(1).to(ground_truth.device)
        error_translation_residual_y = torch.zeros(1).to(ground_truth.device)

        error_z = torch.zeros(1).to(ground_truth.device)
        error_rotation_class = torch.zeros(1).to(ground_truth.device)
        error_size_similarity = torch.zeros(1).to(ground_truth.device)
        error_size = torch.zeros(1).to(ground_truth.device)

        num_related = 0
        num_unrelated = 0

        for b in range(batch_size):
            ind = np.where(np.array(batch_idx) == b)[0]
            ground_truth_ind = np.array(ground_truth_idx)[ind].tolist()
            reconstruct_ind = np.array(reconstruct_idx)[ind].tolist()
            if len(reconstruct_idx) == 0:
                continue

            ground_truth_prob = ground_truth[b][ground_truth_ind, :][:, ground_truth_ind]
            reconstruct_x = x[0][b][reconstruct_ind, :][:, reconstruct_ind]

            # mask filter the furniture that are not related to each other
            mask = ground_truth_prob[:, :, -1:]

            reconstruct_x = reconstruct_x * mask
            num_related += torch.sum(mask)

            # z is the distance between object A and object B in z axis
            error_z += self.huber_loss(
                reconstruct_x[:, :, -7],
                ground_truth_prob[:, :, -5]
            )

            error_translation_residual_x += self.huber_loss(
                reconstruct_x[:, :, -9],
                ground_truth_prob[:, :, -7],
            )

            error_translation_residual_y += self.huber_loss(
                reconstruct_x[:, :, -8],
                ground_truth_prob[:, :, -6]
            )

            error_size += self.huber_loss(
                reconstruct_x[:, :, -2],
                ground_truth_prob[:, :, -2]
            )

            first_nonzero_index = torch.nonzero(mask)[0, 0]
            index = torch.where(mask[first_nonzero_index, :, 0])[0]
            n = index.shape[0]

            error_mat_indicator_x = self.binary_cross_entropy_with_logit_loss(
                reconstruct_x[index, :][:, index][None, :, :, 0],
                ground_truth_prob[index, :][:, index][None, :, :, 0]
            )

            error_mat_indicator_y = self.binary_cross_entropy_with_logit_loss(
                reconstruct_x[index, :][:, index][None, :, :, 1],
                ground_truth_prob[index, :][:, index][None, :, :, 1]
            )

            error_mat_translation_class_x = self.cross_entropy_loss(
                reconstruct_x[index, :][:, index][None, :, :, 2: 2 + self.num_bins].permute(0, 3, 1, 2),
                ground_truth_prob[index, :][:, index][None, :, :, 2].long()
            )

            error_mat_translation_class_y = self.cross_entropy_loss(
                reconstruct_x[index, :][:, index][None, :, :, 2 + self.num_bins: 2 + 2 * self.num_bins].permute(0, 3, 1,
                                                                                                                2),
                ground_truth_prob[index, :][:, index][None, :, :, 3].long()
            )

            error_mat_rotation_class = self.cross_entropy_loss(
                reconstruct_x[index, :][:, index][None, :, :, -6:-3].permute(0, 3, 1, 2),
                ground_truth_prob[index, :][:, index][None, :, :, -4].long()
            )

            error_mat_size_similarity = self.binary_cross_entropy_with_logit_loss(
                reconstruct_x[index, :][:, index][None, :, :, -3],
                ground_truth_prob[index, :][:, index][None, :, :, -3]
            )

            valid_mask = (torch.ones((n, n)) - torch.eye(n))[None, ...].to(ground_truth.device)

            error_indicator_x += torch.sum(error_mat_indicator_x * valid_mask)
            error_indicator_y += torch.sum(error_mat_indicator_y * valid_mask)

            if 'bedroom' in room_type:
                error_translation_class_x += torch.sum(error_mat_translation_class_x * valid_mask)
                error_translation_class_y += torch.sum(error_mat_translation_class_y * valid_mask)
            else:
                error_translation_class_x += torch.sum(error_mat_translation_class_x)
                error_translation_class_y += torch.sum(error_mat_translation_class_y)

            error_rotation_class += torch.sum(error_mat_rotation_class * valid_mask)
            error_size_similarity += torch.sum(error_mat_size_similarity * valid_mask)
            num_unrelated += torch.sum(mask) - n

            if self.training:
                for i in range(1, 3):
                    aux_reconstruct_x = x[i][b][reconstruct_ind, :][:, reconstruct_ind]
                    aux_reconstruct_x = aux_reconstruct_x * mask

                    # z is the distance between object A and object B in z axis
                    error_z += self.huber_loss(
                        aux_reconstruct_x[:, :, -7],
                        ground_truth_prob[:, :, -5]
                    ) * 0.3 * i

                    error_translation_residual_x += self.huber_loss(
                        aux_reconstruct_x[:, :, -9],
                        ground_truth_prob[:, :, -7],
                    ) * 0.3 * i

                    error_translation_residual_y += self.huber_loss(
                        aux_reconstruct_x[:, :, -8],
                        ground_truth_prob[:, :, -6]
                    ) * 0.3 * i

                    error_size += self.huber_loss(
                        aux_reconstruct_x[:, :, -2],
                        ground_truth_prob[:, :, -2]
                    ) * 0.3 * i

                    error_mat_indicator_x = self.binary_cross_entropy_with_logit_loss(
                        aux_reconstruct_x[index, :][:, index][None, :, :, 0],
                        ground_truth_prob[index, :][:, index][None, :, :, 0]
                    )

                    error_mat_indicator_y = self.binary_cross_entropy_with_logit_loss(
                        aux_reconstruct_x[index, :][:, index][None, :, :, 1],
                        ground_truth_prob[index, :][:, index][None, :, :, 1]
                    )

                    error_mat_translation_class_x = self.cross_entropy_loss(
                        aux_reconstruct_x[index, :][:, index][None, :, :, 2: 2 + self.num_bins].permute(0, 3, 1, 2),
                        ground_truth_prob[index, :][:, index][None, :, :, 2].long()
                    )

                    error_mat_translation_class_y = self.cross_entropy_loss(
                        aux_reconstruct_x[index, :][:, index][None, :, :, 2 + self.num_bins: 2 + 2 * self.num_bins].permute(
                            0, 3, 1,
                            2),
                        ground_truth_prob[index, :][:, index][None, :, :, 3].long()
                    )

                    error_mat_rotation_class = self.cross_entropy_loss(
                        aux_reconstruct_x[index, :][:, index][None, :, :, -6:-3].permute(0, 3, 1, 2),
                        ground_truth_prob[index, :][:, index][None, :, :, -4].long()
                    )

                    error_mat_size_similarity = self.binary_cross_entropy_with_logit_loss(
                        aux_reconstruct_x[index, :][:, index][None, :, :, -3],
                        ground_truth_prob[index, :][:, index][None, :, :, -3]
                    )

                    error_indicator_x += torch.sum(error_mat_indicator_x * valid_mask) * 0.3 * i
                    error_indicator_y += torch.sum(error_mat_indicator_y * valid_mask) * 0.3 * i

                    if 'bedroom' in room_type:
                        error_translation_class_x += torch.sum(error_mat_translation_class_x * valid_mask) * 0.3 * i
                        error_translation_class_y += torch.sum(error_mat_translation_class_y * valid_mask) * 0.3 * i
                    else:
                        error_translation_class_x += torch.sum(error_mat_translation_class_x) * 0.3 * i
                        error_translation_class_y += torch.sum(error_mat_translation_class_y) * 0.3 * i

                    error_rotation_class += torch.sum(error_mat_rotation_class * valid_mask) * 0.3 * i
                    error_size_similarity += torch.sum(error_mat_size_similarity * valid_mask) * 0.3 * i
                    
        total_loss += (error_z / (num_related + 1e-12))
        loss_dict['z_loss'] = (error_z / (num_related + 1e-12))
        total_loss += ((error_translation_residual_x + error_translation_residual_y) / (num_related + 1e-12))
        loss_dict['distance_residual_loss'] = ((error_translation_residual_x + error_translation_residual_y) / (
                num_related + 1e-12))

        if room_type == 'bedroom':
            loss_dict['distance_class_loss'] = (error_translation_class_x + error_translation_class_y) / (
                        num_unrelated + 1e-12)
            total_loss += (error_translation_class_x + error_translation_class_y) / (num_unrelated + 1e-12) * 0.1
        else:
            loss_dict['distance_class_loss'] = ((error_translation_class_x + error_translation_class_y) / (
                    num_related + 1e-12))
            total_loss += ((error_translation_class_x + error_translation_class_y) / (num_related + 1e-12)) * 0.1

        loss_dict['distance_indicator_loss'] = ((error_indicator_x + error_indicator_y) / (num_unrelated + 1e-12))
        total_loss += ((error_indicator_x + error_indicator_y) / (num_unrelated + 1e-12))
        loss_dict['rotation_class_loss'] = (error_rotation_class / (num_unrelated + 1e-12))
        total_loss += (error_rotation_class / (num_unrelated + 1e-12)) * 0.1

        loss_dict['size_similarity_loss'] = (error_size_similarity / (num_unrelated + 1e-12))
        total_loss += (error_size_similarity / (num_unrelated + 1e-12))

        loss_dict['size_loss'] = (error_size / (num_related + 1e-12))
        total_loss += (error_size / (num_related + 1e-12))
        return total_loss, loss_dict
