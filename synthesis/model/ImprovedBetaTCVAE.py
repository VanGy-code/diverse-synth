import math
import numpy as np
from typing import List, Tuple, Any

import torch
from torch import nn, Tensor
from torch.autograd import Variable

from synthesis.model.Common import BaseVAE, Swish, SELayer
from synthesis.model.SparseLinear import AutoSparseLinear


class ResidualBlock(nn.Module):
    def __init__(self, hidden_format, bn_momentum=0.01):
        super(ResidualBlock, self).__init__()

        self.hidden_format = hidden_format

        self.seq = nn.Sequential(
            nn.Linear(hidden_format[0], hidden_format[0]),
            nn.BatchNorm1d(hidden_format[0], momentum=bn_momentum),
            Swish(),
            nn.Linear(hidden_format[0], hidden_format[0])
        )
        self.se_layer = SELayer(hidden_format[0])

    def forward(self, x):
        B, _, _ = x.shape
        y = self.seq(
            x.permute(0, 2, 1).reshape(B * self.hidden_format[1], self.hidden_format[0])
        ).reshape(B, self.hidden_format[1], self.hidden_format[0]).permute(0, 2, 1)
        return x + 0.1 * self.se_layer(y)


class EncoderBlock(nn.Module):
    def __init__(self, input_format, output_format, kernel_size=4, kernel_mask=None, momentum=0.01):
        super().__init__()
        self.output_format = output_format
        self.kernel_mask = kernel_mask
        self.sparse_linear = AutoSparseLinear(
            input_format=input_format,
            output_format=output_format,
            kernel_size=kernel_size,
            kernel_mask=kernel_mask,
        )

        if self.kernel_mask is None:
            self.kernel_mask = self.sparse_linear.kernel_mask
        self.bn = nn.BatchNorm1d(output_format[0] * output_format[1], momentum=momentum)

        self.swish = Swish()

    def forward(self, x):
        B = x.shape[0]
        y = self.sparse_linear(x).reshape(B, -1)
        y = self.swish(self.bn(y)).reshape(B, self.output_format[0], self.output_format[1])
        return y


class DownSampleBlock(nn.ModuleList):
    def __init__(self, input_format, output_format, momentum=0.01):
        super().__init__()
        self.input_format = input_format
        self.output_format = output_format
        self.linear = nn.Linear(input_format[0], output_format[0])

        self.bn = nn.BatchNorm1d(output_format[0] * output_format[1], momentum=momentum)
        self.swish = Swish()

    def forward(self, x):
        B = x.shape[0]
        y = self.linear(
            x.permute(0, 2, 1).reshape(B * self.input_format[1], self.input_format[0])
        ).reshape(B, self.output_format[1], self.output_format[0]).permute(0, 2, 1).reshape(B, -1)

        y = self.swish(self.bn(y)).reshape(B, self.output_format[0], self.output_format[1])

        return y


class Encoder(nn.Module):
    def __init__(self, data_config, encoder_config, kernel_mask_list=None, latent_dim=256, is_variational=True):
        super().__init__()

        self.is_variational = is_variational
        self.input_dim = encoder_config["input_dim"]
        self.num_class = data_config["num_class"]
        self.num_each_class = data_config["num_each_class"]
        self.encoder_config = encoder_config
        self.latent_dim = latent_dim

        self.kernel_mask_list = kernel_mask_list

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                input_format=(self.num_class, self.input_dim * self.num_each_class),
                output_format=(self.encoder_config["sparse_embedding1"], self.encoder_config["embedding_dim1"]),
                kernel_size=4,
                kernel_mask=kernel_mask_list[0] if kernel_mask_list is not None else None,
                momentum=encoder_config["bn_momentum"]
            ),
            EncoderBlock(
                input_format=(self.encoder_config["linear_embedding1"], self.encoder_config["embedding_dim1"]),
                output_format=(self.encoder_config["sparse_embedding2"], self.encoder_config["embedding_dim2"]),
                kernel_size=4,
                kernel_mask=kernel_mask_list[1] if kernel_mask_list is not None else None,
                momentum=encoder_config["bn_momentum"]
            ),
            EncoderBlock(
                input_format=(self.encoder_config["linear_embedding2"], self.encoder_config["embedding_dim2"]),
                output_format=(self.encoder_config["sparse_embedding3"], self.encoder_config["embedding_dim3"]),
                kernel_size=4,
                kernel_mask=kernel_mask_list[2] if kernel_mask_list is not None else None,
                momentum=encoder_config["bn_momentum"],
            )
        ])

        self.encoder_residual_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_format=(self.encoder_config["linear_embedding1"], self.encoder_config["embedding_dim1"]),
            ),
            ResidualBlock(
                hidden_format=(self.encoder_config["linear_embedding2"], self.encoder_config["embedding_dim2"]),
            ),
            ResidualBlock(
                hidden_format=(self.encoder_config["linear_embedding3"], self.encoder_config["embedding_dim3"]),
            )
        ])

        self.down_sample_blocks = nn.ModuleList([
            DownSampleBlock(
                input_format=(self.encoder_config["sparse_embedding1"], self.encoder_config["embedding_dim1"]),
                output_format=(self.encoder_config["linear_embedding1"], self.encoder_config["embedding_dim1"]),
                momentum=encoder_config["bn_momentum"]
            ),
            DownSampleBlock(
                input_format=(self.encoder_config["sparse_embedding2"], self.encoder_config["embedding_dim2"]),
                output_format=(self.encoder_config["linear_embedding2"], self.encoder_config["embedding_dim2"]),
                momentum=encoder_config["bn_momentum"]
            ),
            DownSampleBlock(
                input_format=(self.encoder_config["sparse_embedding3"], self.encoder_config["embedding_dim3"]),
                output_format=(self.encoder_config["linear_embedding3"], self.encoder_config["embedding_dim3"]),
                momentum=encoder_config["bn_momentum"]
            ),
        ])

        if self.kernel_mask_list is None:
            mask_list = []
            for layer in self.encoder_blocks.children():
                mask_list.append(layer.kernel_mask)

            self.kernel_mask_list = mask_list

        self.condition_x = nn.Sequential(
            Swish(),
            nn.Linear(self.encoder_config["linear_embedding3"] * self.encoder_config["embedding_dim3"], latent_dim * 2)
        )

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std)
        if torch.cuda.is_available():
            eps = Variable(eps.to(mu.device))
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def sample(self, params):
        mu = params.select(-1, 0)
        log = params.select(-1, 1)
        std_z = Variable(torch.randn(mu.size()).type_as(mu.data))
        sample = std_z * torch.exp(log) + mu
        return sample

    def forward(self, x):
        assert (x.shape[1] == self.num_class * self.num_each_class)
        B = x.shape[0]
        
        x = x.reshape(-1, self.num_class, self.input_dim * self.num_each_class)

        # (B, 256, 32)
        x = self.encoder_blocks[0](x)
        # (B, 32, 32)
        x = self.down_sample_blocks[0](x)
        x = self.encoder_residual_blocks[0](x)

        # (B, 128, 48)
        x = self.encoder_blocks[1](x)
        # (B, 16, 48)
        x = self.down_sample_blocks[1](x)
        x = self.encoder_residual_blocks[1](x)

        # (B, 64, 64)
        x = self.encoder_blocks[2](x)
        # (B, 4, 64)
        x = self.down_sample_blocks[2](x)
        x = self.encoder_residual_blocks[2](x)

        x = x.reshape(B, -1)

        if self.is_variational:
            z = self.condition_x(x).view(x.size(0), self.latent_dim * 2)

            return z
        else:
            raise NotImplementedError


class DecoderBlock(nn.Module):
    def __init__(self, input_format, output_format, kernel_size=4, kernel_mask=None, momentum=0.01):
        super().__init__()
        self.output_format = output_format
        self.kernel_mask = kernel_mask
        self.sparse_linear = AutoSparseLinear(
            input_format=input_format,
            output_format=output_format,
            kernel_size=kernel_size,
            kernel_mask=kernel_mask,
            channel_wise=True,
            channel_nums=1
        )

        if self.kernel_mask is None:
            self.kernel_mask = self.sparse_linear.kernel_mask

        self.bn = nn.BatchNorm1d(output_format[0] * output_format[1], momentum=momentum)

        self.swish = Swish()

    def forward(self, x):
        B = x.shape[0]
        y = self.sparse_linear(x)
        y = self.swish(self.bn(y.reshape(B, self.output_format[0] * self.output_format[1])))
        return y.reshape(B, self.output_format[0], self.output_format[1])


class UpSampleLinearBlock(nn.Module):
    def __init__(self, input_format=None, output_format=None, momentum=0.01, hidden_dim=None):
        super().__init__()
        assert input_format is not None or hidden_dim is not None
        self.input_format = input_format
        self.output_format = output_format
        self.hidden_dim = hidden_dim
        self.seq = nn.Sequential(
            nn.Linear(
                (input_format[0] * input_format[1]) if hidden_dim is None else hidden_dim,
                output_format[0] * output_format[1]
            ),
            nn.BatchNorm1d(output_format[0] * output_format[1], momentum=momentum),
            Swish()
        )

    def forward(self, x):
        B = x.shape[0]
        return self.seq(
            x.reshape(B, self.input_format[0] * self.input_format[1]) if self.hidden_dim is None else x
        ).reshape(B, self.output_format[0], self.output_format[1])


class Decoder(nn.Module):
    def __init__(self, data_config, decoder_config, kernel_mask_list, latent_dim=32):
        super().__init__()

        self.num_class = data_config["num_class"]
        self.num_each_class = data_config["num_each_class"]
        self.output_dim = decoder_config["input_dim"]
        self.decoder_config = decoder_config
        self.kernel_mask_list = kernel_mask_list

        self.up_sample_blocks = nn.ModuleList([
            UpSampleLinearBlock(
                hidden_dim=latent_dim,
                output_format=(decoder_config["sparse_embedding3"], decoder_config["embedding_dim3"]),
                momentum=decoder_config["bn_momentum"]
            ),
            UpSampleLinearBlock(
                input_format=(decoder_config["linear_embedding2"], decoder_config["embedding_dim2"]),
                output_format=(decoder_config["sparse_embedding2"], decoder_config["embedding_dim2"]),
                momentum=decoder_config["bn_momentum"]
            ),
            UpSampleLinearBlock(
                input_format=(decoder_config["linear_embedding1"], decoder_config["embedding_dim1"]),
                output_format=(decoder_config["sparse_embedding1"], decoder_config["embedding_dim1"]),
                momentum=decoder_config["bn_momentum"]
            ),

        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                input_format=(decoder_config["sparse_embedding3"], decoder_config["embedding_dim3"]),
                output_format=(decoder_config["linear_embedding2"], decoder_config["embedding_dim2"]),
                kernel_size=self.num_each_class,
                kernel_mask=kernel_mask_list[0] if kernel_mask_list is not None else None,
                momentum=decoder_config["bn_momentum"],
            ),
            DecoderBlock(
                input_format=(decoder_config["sparse_embedding2"], decoder_config["embedding_dim2"]),
                output_format=(decoder_config["linear_embedding1"], decoder_config["embedding_dim1"]),
                kernel_size=self.num_each_class,
                kernel_mask=kernel_mask_list[1] if kernel_mask_list is not None else None,
                momentum=decoder_config["bn_momentum"],
            ),
            AutoSparseLinear(
                input_format=(decoder_config["sparse_embedding1"], decoder_config["embedding_dim1"]),
                output_format=(self.num_class * self.num_each_class, self.output_dim),
                kernel_size=self.num_each_class,
                kernel_mask=kernel_mask_list[2] if kernel_mask_list is not None else None,
                channel_wise=True,
                channel_nums=self.num_each_class
            )
        ])

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_format=(decoder_config["linear_embedding2"], decoder_config["embedding_dim2"]),
                bn_momentum=decoder_config["bn_momentum"]),
            ResidualBlock(
                hidden_format=(decoder_config["linear_embedding1"], decoder_config["embedding_dim1"]),
                bn_momentum=decoder_config["bn_momentum"]),
        ])

        if self.kernel_mask_list is None:
            mask_list = []
            for layer in self.decoder_blocks.children():
                mask_list.append(layer.kernel_mask)
            self.kernel_mask_list = mask_list

    def forward(self, x):
        """
        :param x: (B, 256)
        :return : (B, N, output_dim),  (qi, ti, si, ci)
        """
        # (B, 32, 64)
        x = self.up_sample_blocks[0](x)

        # (B, 16, 48)
        x = self.decoder_blocks[0](x)
        x = self.residual_blocks[0](x)

        # (B, 128, 48)
        x = self.up_sample_blocks[1](x)

        # (B, 32, 32)
        x = self.decoder_blocks[1](x)
        x = self.residual_blocks[1](x)

        # (B, 256, 32)
        x = self.up_sample_blocks[2](x)

        # (B, num_class * num_each_class, output_dim)
        x = self.decoder_blocks[2](x)

        return x
    
    def sample(self, params):
        mu = params.select(-1, 0)
        log = params.select(-1, 1)
        std_z = Variable(torch.randn(mu.size()).type_as(mu.data))
        sample = std_z * torch.exp(log) + mu
        return sample



class ImprovedBetaTCVAE(BaseVAE):
    def __init__(self, data_config, kernel_mask_dict, model_config):
        super().__init__()
        self.num_class = data_config["num_class"]
        self.num_each_class = data_config["num_each_class"]
        self.latent_dim = model_config['latent_dimension']
        self.weight_kld = model_config['kld_weight']
        self.kld_interval = model_config['kld_interval']
        self.kernel_mask_dict = kernel_mask_dict
        self.is_mss = model_config['mss']

        self.alpha = 0.001
        self.beta = 64.
        self.gamma = 1.
        
        self.normalization = Variable(torch.Tensor([np.log(2 * np.pi)]))
        
        # hyperparamters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.latent_dim, 2))
        

        self.encoder = Encoder(
            data_config,
            model_config,
            kernel_mask_dict.get('encoder') if kernel_mask_dict is not None else None,
            self.latent_dim
        )
        self.decoder = Decoder(
            data_config,
            model_config,
            kernel_mask_dict.get('decoder') if kernel_mask_dict is not None else None,
            self.latent_dim
        )

        if self.kernel_mask_dict is None:
            self.kernel_mask_dict = {
                'encoder': self.encoder.kernel_mask_list,
                'decoder': self.decoder.kernel_mask_list
            }

        # loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss(beta=0.5, reduction='mean')
        
    # return prior paramters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params
    
    # samples from the model p(x|z)p(z)
    def sample(self, batch_size=1):
        prior_params = self._get_prior_params(batch_size)
        zs = self.encoder.sample(prior_params)
        x_params = self.decoder(zs)
        return x_params
        
    
    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        start_weight = (N - M) / (M * N)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M + 1] = 1 / N
        W.view(-1)[1::M + 1] = start_weight
        W[M - 1, 0] = start_weight
        return W.log()
        
    def encode(self, x):
        x = x.view(x.size(0), 92, 16)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder(x).view(x.size(0), self.latent_dim, 2)
        # sample the latent code z
        zs = z = self.encoder.sample(z_params)
        return zs, z_params
    
    def decode(self, z):
        x_params = self.decoder(z).view(z.size(0), 92, 16)
        xs = self.decoder.sample(x_params)
        return xs, x_params

    def forward(self, x) -> List[Tensor]:
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params
    
    def _log_density(self, sample, params=None):
        mu = params.select(-1, 0)
        log = params.select(-1, 1)
        
        c = self.normalization.type_as(sample.data)
        inv_sigma = torch.exp(-log)
        tmp = (sample - mu) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * log + c)

    def loss_function(self, **kwargs: Any):

        ground_truth = kwargs['ground_truth']
        x, x_params, zs, z_params = kwargs['package']
        dataset_size = kwargs['dataset_size']
        total_loss = kwargs['total_loss']

        loss_dict = dict()

        batch_size, latent_dim = zs.shape
        prior_params = self._get_prior_params(batch_size)
        
        # calculate log q(z|x)
        log_qz_condx = self._log_density(zs, params=z_params).view(batch_size, -1).sum(dim=1)

        # calculate log p(z)
        log_p_z = self._log_density(zs, params=prior_params).view(batch_size, -1).sum(dim=1)

        # calculate log density of a Gaussian for all combination of batch pairs of x and mu
        mat_log_q_z = self._log_density(
            zs.view(batch_size, 1, latent_dim),
            z_params.view(1, batch_size, latent_dim, 2)
        )

        # mss
        if self.is_mss:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(mat_log_q_z.data))
            log_qz = torch.logsumexp(logiw_matrix + mat_log_q_z.sum(2), dim=1, keepdim=False)
            log_prod_q_z = torch.logsumexp(logiw_matrix.view(batch_size, batch_size, 1) + mat_log_q_z, dim=1, keepdim=False).sum(1)
        else:
            log_prod_q_z = (torch.logsumexp(mat_log_q_z, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            log_qz = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size)

        mi_loss = (log_qz_condx - log_qz).mean()
        tc_loss = (log_qz - log_prod_q_z).mean()
        dw_kld_loss = (log_prod_q_z - log_p_z).mean()

        loss_dict['mi_loss'] = mi_loss
        loss_dict['tc_loss'] = tc_loss
        loss_dict['dw_kld_loss'] = dw_kld_loss
        
        # total_loss += mi_loss * self.alpha + (tc_loss * self.beta + dw_kld_loss * self.gamma) * self.weight_kld
        total_loss += mi_loss * self.alpha + (tc_loss * self.beta + dw_kld_loss * self.gamma) * self.weight_kld

        batch_size, max_num_parts, x_dim = ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2]
        num_class = self.num_class
        num_each_class = self.num_each_class
        assert (max_num_parts == num_class * num_each_class)

        x_prob = x_params[:, :, 9:]
        # Compute distance matrix
        distance_matrix = torch.zeros((batch_size, num_class, num_each_class, num_each_class))
        for b in range(batch_size):
            for c in range(num_class):
                offset = c * num_each_class
                distance_matrix[b, c] = torch.norm(
                    ground_truth[b, offset: offset + num_each_class, 9:].unsqueeze(1) -
                    x_prob[b, offset: offset + num_each_class, :].unsqueeze(0),
                    dim=2
                )

        batch_idx, ground_truth_idx, reconstruct_idx = ImprovedBetaTCVAE.linear_assignment_class(distance_matrix)

        # (batch_size * max_num_parts, feature_dim)
        ground_truth_match = ground_truth[batch_idx, ground_truth_idx]
        reconstruct_match = x_params[batch_idx, reconstruct_idx]

        # Overlap

        # Furniture Class reconstruct loss(Huber Loss)
        # (batch_size * max_num_parts, feature_dim)
        ground_truth_class = ground_truth_match[:, 9:]
        reconstruct_class = reconstruct_match[:, 9:]
        mask = ground_truth_class[:, -1:]
        reconstruct_class_new = torch.cat([reconstruct_class[:, :-1] * mask, reconstruct_class[:, -1:]], dim=1)
        # reduction = 'mean' or 'sum'

        class_reconstruct_loss = self.huber_loss(reconstruct_class_new, ground_truth_class) * (x_dim - 9)
        total_loss += class_reconstruct_loss * 10
        loss_dict['class_reconstruct_loss'] = class_reconstruct_loss

        # Furniture Angle reconstruct loss(Cross Entropy Loss + Huber Loss)
        reconstruct_angle = reconstruct_match[:, :9]
        ground_truth_angle = ground_truth_match[:, :9]
        angle_index = torch.nonzero(ground_truth_angle[:, :8])
        angle_label = angle_index[:, 1]
        angle_class_loss = self.cross_entropy_loss(reconstruct_angle[angle_index[:, 0], :8], angle_label)
        angle_residual_loss = self.huber_loss(reconstruct_angle[:, -1], ground_truth_angle[:, -1])
        total_loss += angle_class_loss * 0.1
        loss_dict['angle_class_loss'] = angle_class_loss
        total_loss += angle_residual_loss
        loss_dict['angle_residual_loss'] = angle_residual_loss

        return total_loss, loss_dict, batch_idx, ground_truth_idx, reconstruct_idx

    @torch.no_grad()
    def sample(self, latent_code: Tensor) -> Tuple[Any, Any]:
        sample = self.decoder(latent_code)
        return sample
