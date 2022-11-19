import math
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
            mu, log_var = self.condition_x(x).chunk(2, dim=1)

            return mu, log_var
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


class EnhancedBetaTCVAE(BaseVAE):
    def __init__(self, data_config, kernel_mask_dict, model_config):
        super().__init__()
        self.num_class = data_config["num_class"]
        self.num_each_class = data_config["num_each_class"]
        self.latent_dim = model_config['latent_dimension']
        self.weight_kld = model_config['kld_weight']
        self.kld_interval = model_config['kld_interval']
        self.kernel_mask_dict = kernel_mask_dict

        self.alpha = 1.0
        self.beta = 6.
        self.gamma = 1.
        self.anneal_steps = 6600
        self.num_iter = 0

        self.is_mss = True

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

    def forward(self, x) -> List[Tensor]:
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var)
        sample = self.decoder(z)
        return [mu, log_var, z, sample]

    def adjust_weight_kld(self, epoch):
        if epoch < self.kld_interval:
            self.weight_kld = (epoch / self.kld_interval) * self.weight_kld

    def log_density_gaussion(self, x, mu, logvar):
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_function(self, **kwargs: Any):
        if self.training:
            self.num_iter += 1
            self.anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            self.anneal_rate = 1.

        ground_truth = kwargs['ground_truth']
        mu, log_var, z, x = kwargs['package']
        dataset_size = kwargs['dataset_size']

        loss_dict = dict()

        batch_size, latent_dim = z.shape

        # calculate log q(z|x)
        log_q_zx = self.log_density_gaussion(z, mu, log_var).sum(dim=1)

        # calculate log p(z)
        # mean and log var is 0
        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussion(z, zeros, zeros).sum(dim=1)

        # calculate log density of a Gaussian for all combination of batch pairs of x and mu
        mat_log_q_z = self.log_density_gaussion(
            z.view(batch_size, 1, latent_dim),
            mu.view(1, batch_size, latent_dim),
            log_var.view(1, batch_size, latent_dim)
        )

        # mss
        if self.is_mss:
            start_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))

            importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(x.device)
            importance_weights.view(-1)[::batch_size] = 1 / dataset_size
            importance_weights.view(-1)[1::batch_size] = start_weight
            importance_weights[batch_size - 2, 0] = start_weight
            log_importance_weights = importance_weights.log()

            mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size)
        log_prod_q_z = (torch.logsumexp(mat_log_q_z, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
        # kld_loss = (log_prod_q_z - log_p_z).mean()

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1) - log_p_z).mean()

        loss_dict['mi_loss'] = mi_loss * self.alpha
        loss_dict['tc_loss'] = tc_loss * self.beta * self.weight_kld
        loss_dict['kld_loss'] = kld_loss * self.gamma * self.anneal_rate * self.weight_kld

        batch_size, max_num_parts, x_dim = ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2]
        num_class = self.num_class
        num_each_class = self.num_each_class
        assert (max_num_parts == num_class * num_each_class)

        x_prob = x[:, :, 9:]
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

        batch_idx, ground_truth_idx, reconstruct_idx = EnhancedBetaTCVAE.linear_assignment_class(distance_matrix)

        # (batch_size * max_num_parts, feature_dim)
        ground_truth_match = ground_truth[batch_idx, ground_truth_idx]
        reconstruct_match = x[batch_idx, reconstruct_idx]

        # Overlap

        # Furniture Class reconstruct loss(Huber Loss)
        # (batch_size * max_num_parts, feature_dim)
        ground_truth_class = ground_truth_match[:, 9:]
        reconstruct_class = reconstruct_match[:, 9:]
        mask = ground_truth_class[:, -1:]
        reconstruct_class_new = torch.cat([reconstruct_class[:, :-1] * mask, reconstruct_class[:, -1:]], dim=1)
        # reduction = 'mean' or 'sum'

        class_reconstruct_loss = self.huber_loss(reconstruct_class_new, ground_truth_class) * (x_dim - 9)
        loss_dict['class_reconstruct_loss'] = class_reconstruct_loss * 10

        # Furniture Angle reconstruct loss(Cross Entropy Loss + Huber Loss)
        reconstruct_angle = reconstruct_match[:, :9]
        ground_truth_angle = ground_truth_match[:, :9]
        angle_index = torch.nonzero(ground_truth_angle[:, :8])
        angle_label = angle_index[:, 1]
        angle_class_loss = self.cross_entropy_loss(reconstruct_angle[angle_index[:, 0], :8], angle_label)
        angle_residual_loss = self.huber_loss(reconstruct_angle[:, -1], ground_truth_angle[:, -1])
        loss_dict['angle_class_loss'] = angle_class_loss * 0.1
        loss_dict['angle_residual_loss'] = angle_residual_loss

        return loss_dict, batch_idx, ground_truth_idx, reconstruct_idx

    @torch.no_grad()
    def sample(self, latent_code: Tensor) -> Tuple[Any, Any]:
        sample = self.decoder(latent_code)
        return sample
