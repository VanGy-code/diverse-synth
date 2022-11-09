import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor, index_select
from torch.nn.parameter import Parameter


class SparseLinear(nn.Module):

    def __init__(self, input_format: tuple, output_format: tuple, sparse_num: int, pair_params: list, bias: bool = True,
                 is_decoder: bool = False):
        super(SparseLinear, self).__init__()
        self.input_format = input_format
        self.output_format = output_format
        self.sparse_num = sparse_num
        self.bias = bias
        self.is_decoder = is_decoder

        self.pair_params = pair_params

        self.weight_list = []
        self.bias_list = []
        for i in range(len(self.pair_params)):
            if is_decoder:
                self.weight_list.append(
                    Parameter(
                        data=Tensor(output_format[1], input_format[1] * len(self.pair_params[i])),
                        requires_grad=True)
                )
            else:

                self.weight_list.append(
                    Parameter(
                        data=Tensor(output_format[1], input_format[1] * sparse_num),
                        requires_grad=True)
                )
            # Turn the untrainable tensor to trianable parameters
            self.register_parameter("weight" + str(i), self.weight_list[-1])

            if self.bias:
                self.bias_list.append(Parameter(data=Tensor(output_format[1]), requires_grad=True))
            else:
                self.bias_list.append(None)
            self.register_parameter("bias" + str(i), self.bias_list[-1])

        try:
            self.reset_parameters()
        except RuntimeError:
            raise RuntimeError

    def reset_parameters(self):
        # Same as the reset_parameters of _ConvNd
        for i in range(len(self.weight_list)):
            init.kaiming_uniform_(self.weight_list[i], a=math.sqrt(5))

        if self.bias:
            for i in range(len(self.weight_list)):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_list[i])
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_list[i], -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (height_in, width_in)
        :return outputs: (height_out, width_out)
        """
        B = x.shape[0]  # batch size
        assert (len(self.pair_params) == self.output_format[0])

        class_stack = []
        for i in range(len(self.pair_params)):
            # Weight and bias don't need .to(device) since we have registered parameter
            ndex = self.pair_params[i].long().to(x.device)
            # Select sparse_num tuple from height_in and get (B, sparse_num, width_in),
            # Reshape to (B, sparse_num * width_in)
            # Multiply weight(sparse_num * width_in, width_out) and get (B, width_out)

            # temp = index_select(x, 1, ndex).reshape(B, -1)
            layer_data = F.linear(
                # (B, self.input_dim * self.num_each_class * sparse_num)
                index_select(x, 1, ndex).reshape(B, -1),
                # (output_format[1], self.input_dim * self.num_each_class * sparse_num))
                self.weight_list[i],
                self.bias_list[i]
            )
            # Stack all together and get (B, height_out, width_out), height_out = num_pairs
            class_stack.append(layer_data)

        return torch.stack(class_stack, 1)

    @property
    def representation(self):
        return 'input_format={}, output_format={}, bias={}'.format(
            self.input_format, self.output_format, self.bias
        )


class AutoSparseLinear(nn.Module):
    def __init__(self, input_format: tuple, output_format: tuple, kernel_size: int, kernel_mask: list = None,
                 bias: bool = True, channel_wise: bool = False, channel_nums: int = 1):
        super().__init__()
        """
            :param input_format: (height_in, width_in)
                                height_in can not be too large (e.g.200)
                                since connect num is C_{height_in}^{4}.
            :param output_format: (height_out, width_out)
                                height_out = len(self.pair_params) need to be computed manually
            :param kernel_size: same as conv, but in RandomSparseLinear 
            :param pair_params:
            :param bias:
         """
        super(AutoSparseLinear, self).__init__()
        self.input_format = input_format
        self.output_format = output_format
        self.kernel_size = kernel_size
        self.bias = bias

        if kernel_mask is not None:
            self.kernel_mask = kernel_mask
        else:
            if channel_wise:
                self._generate_mask(output_format[0], kernel_size, input_format[0])
                self._generate_channel_wise_mask(int(output_format[0] / channel_nums), channel_nums)
            else:
                self._generate_mask(input_format[0], kernel_size, output_format[0])

        self.weight_list = []
        self.bias_list = []
        for i in range(len(self.kernel_mask)):
            if channel_wise:
                self.weight_list.append(
                    Parameter(
                        data=Tensor(output_format[1], input_format[1] * len(self.kernel_mask[i])),
                        requires_grad=True
                    )
                )
            else:
                self.weight_list.append(
                    Parameter(
                        data=Tensor(output_format[1], input_format[1] * kernel_size),
                        requires_grad=True)
                )
            self.register_parameter("weight" + str(i), self.weight_list[-1])

            if self.bias:
                self.bias_list.append(Parameter(data=Tensor(output_format[1]), requires_grad=True))
            else:
                self.bias_list.append(None)
            self.register_parameter("bias" + str(i), self.bias_list[-1])

        try:
            self.reset_parameters()
        except RuntimeError:
            raise RuntimeError

    def reset_parameters(self):
        for i in range(len(self.weight_list)):
            init.kaiming_uniform_(self.weight_list[i], a=math.sqrt(5))

        if self.bias:
            for i in range(len(self.weight_list)):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_list[i])
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_list[i], -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (height_in, width_in)
        :return outputs: (height_out, width_out)
        """
        B = x.shape[0]  # batch size
        assert (len(self.kernel_mask) == self.output_format[0])

        out_stack = []
        for i in range(len(self.kernel_mask)):
            # Weight and bias don't need .to(device) since we have registered parameter
            ndex = self.kernel_mask[i].long().to(x.device)
            # Select kernel_size tuples from height_in and get (B, kernel_size, width_in),
            # Reshape to (B, kernel_size * width_in)
            # Multiply weight(kernel_size * width_in, width_out) and get (B, width_out)

            layer_data = F.linear(
                index_select(x, 1, ndex).reshape(B, -1),
                self.weight_list[i],
                self.bias_list[i]
            )
            # Stack all together and get (B, height_out, width_out), height_out = num_pairs
            out_stack.append(layer_data)

        return torch.stack(out_stack, 1)

    def _count_hit(self, select, pairs2) -> int:
        pairs4 = list(itertools.combinations(select, 2))
        for pair in pairs4:
            if pair in pairs2:
                return 1
        return 0

    def _generate_mask(self, num_layer, subset_length, next_layer):
        assert (num_layer > (subset_length + 2))

        result = []
        # combination of each subset_length classes
        allpairs = list(itertools.combinations(range(num_layer), subset_length))
        # combination of each 2 classes
        allpairs2 = list(itertools.combinations(range(num_layer), 2))
        # randomly generate an ordered list
        order = list(np.random.permutation(len(allpairs)))

        for i in range(next_layer):
            if len(allpairs2) == 0:
                for j in order[0: next_layer - i]:
                    result.append(list(allpairs[j]))
                break

            idx = np.random.randint(0, len(order) - 1)
            # select one subset, if subset_length = 4, select should be like (4, 5, 1, 6)
            select = allpairs[order[idx]]
            # if there is no relation between selected class, then redo select progress.
            while self._count_hit(select, allpairs2) < 1:
                idx = np.random.randint(0, len(order) - 1)
                select = allpairs[order[idx]]

            result.append(list(select))

            pairs4 = list(itertools.combinations(select, 2))
            for pair in pairs4:
                if pair in allpairs2:
                    allpairs2.remove(pair)

            order.remove(order[idx])

        self.kernel_mask = torch.tensor(result)

    def _generate_channel_wise_mask(self, num_class, subset_length):

        mask = []
        for i in range(num_class):
            type_contain = []
            for j in range(len(self.kernel_mask)):
                if i in self.kernel_mask[j]:
                    type_contain.append(j)

            for _ in range(subset_length):
                mask.append(torch.tensor(type_contain))

        self.kernel_mask = mask

    @property
    def representation(self):
        return 'input_format={}, output_format={}, bias={}'.format(
            self.input_format, self.output_format, self.bias
        )
