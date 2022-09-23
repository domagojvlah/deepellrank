#! /usr/bin/env python

# Models

import torch
from torch import nn
import math  # for math.floor()
import logging


# raised by model constructor in case of illegal combination of model hyperparameters
class IllegalArgument(Exception):
    pass

# Deep convolutional encoder


def conv_nonlin(in_ch, out_ch, kernel_size, stride=None, nonlin=None):
    return [nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nonlin,
            nn.BatchNorm1d(out_ch),  # , eps=1e-3)
            ]


class ConvEncoder(nn.Module):
    def __init__(self,
                 channels=1,
                 expected_input_length=669,
                 dim=None,
                 input_layers=None,
                 reducing_layers=None,
                 output_layers=None,
                 reduction_exponent=None,
                 kernel_size=None,
                 stride=None,
                 encode_position=None,
                 dropout=None,
                 nonlin=None,
                 maxpool_layer=False,
                 ):
        super().__init__()

        assert (dim is not None) and (input_layers is not None) and (
            reducing_layers is not None) and (output_layers is not None)
        assert (reduction_exponent is not None) and (
            kernel_size is not None) and (stride is not None)
        assert (encode_position is not None)
        assert (dropout is not None)
        assert (nonlin is not None)

        self.expected_input_length = expected_input_length
        self.dim = dim
        self.reducing_layers = reducing_layers
        self.reduction_exponent = reduction_exponent
        self.kernel_size = kernel_size
        self.stride = stride
        self.encode_position = encode_position
        self.maxpool_layer = maxpool_layer

        self.reduced_kernel_size_flag = False

        # total number of input channels
        in_channels = channels
        if self.encode_position:
            self.register_buffer("pos_encoding", torch.linspace(-1, 1,
                                 expected_input_length).reshape(1, 1, expected_input_length))
            in_channels += 1

        # check to see if number of reducing layers is not to big compared to stride
        if self.reducing_layers >= 1:
            if self.output_length(self.expected_input_length, red_lay=self.reducing_layers - 1) == 1 and \
                    self.output_length(self.expected_input_length, red_lay=self.reducing_layers) == 1:
                raise IllegalArgument(
                    "The number of reducing layers compared to stride to big.")

        layers = [nn.Dropout(dropout)]  # Initial dropout
        layers += conv_nonlin(in_channels, dim, kernel_size,
                              stride=1, nonlin=nonlin)

        for _ in range(input_layers):
            layers += conv_nonlin(dim, dim, kernel_size,
                                  stride=1, nonlin=nonlin)

        #layers += [nn.Dropout(dropout)]

        for i in range(reducing_layers):
            proper_kernel_size = self.calculate_proper_kernel_size(i)
            layers += conv_nonlin(self.get_layer_channels(i),
                                  self.get_layer_channels(i+1),
                                  proper_kernel_size, stride=stride, nonlin=nonlin)  # Reduction using kernel_size and stride

        #layers += [nn.Dropout(dropout)]

        proper_kernel_size = self.calculate_proper_kernel_size()
        if proper_kernel_size < self.kernel_size:
            self.reduced_kernel_size_flag = True
        for _ in range(output_layers):
            layers += conv_nonlin(self.get_layer_channels(reducing_layers),
                                  self.get_layer_channels(reducing_layers),
                                  proper_kernel_size, stride=1, nonlin=nonlin)

        if self.maxpool_layer:  # Maxpool layer is the last layer. Exit length is 1.
            layers += [nn.MaxPool1d(
                kernel_size=self.output_length(self.expected_input_length))]
            logging.debug(
                "Model is using maxpool layer as the last encoder layer")

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.prepare(x)
        return self.layers(x)

    def prepare(self, x):
        if x.dim() == 2:  # make it a batch of one
            x = x.unsqueeze(0)

        input_list = [x]

        if self.encode_position:
            input_list += [self.pos_encoding.expand(x.shape[0], -1, -1)]

        return torch.cat(input_list, dim=1)

    def output_length(self, input_length, red_lay=None):
        l = input_length
        if red_lay is None:
            red_lay = self.reducing_layers
        for _ in range(red_lay):
            l = math.floor((l + 2 * (self.kernel_size // 2) -
                           (self.kernel_size - 1) - 1) / self.stride + 1)
        return l

    def get_latent_shape(self, input_length=None):
        if input_length is None:
            input_length = self.expected_input_length
        if self.maxpool_layer:
            return [self.get_layer_channels(self.reducing_layers),
                    1]
        else:
            return [self.get_layer_channels(self.reducing_layers),
                    self.output_length(input_length)]

    def get_layer_channels(self, l):
        return self.dim * math.floor(self.reduction_exponent ** l)

    def calculate_proper_kernel_size(self, red_lay=None):
        if red_lay is None:
            red_lay = self.reducing_layers
        out_len = self.output_length(self.expected_input_length, red_lay)
        # Reduce kernel size if too big for output length. Kernel size is odd.
        return min(self.kernel_size, 2 * (out_len // 2) + 1)


class SimpleConvolutionalClassificationModel(nn.Module):
    def __init__(self,
                 number_of_classes=2,
                 dropout_after_encoder=None,  # default is no dropout after encoder
                 nonlin=nn.ReLU(inplace=True),
                 # string encoding nonlin func - not used directly, but passed as a hyperparameter
                 nonlin_str=None,
                 hidden_lin_ftrs=[],  # default is no hidden layers in FC classification head
                 force_nonincreasing_lin_ftrs=False,
                 MIN_LATENT_SIZE=2,  # latent space should not be smaller than number of classes
                 **kwargs):
        super().__init__()

        assert nonlin_str is not None

        self.encoder = ConvEncoder(nonlin=nonlin, **kwargs)

        self.dropout_after_encoder = dropout_after_encoder
        if self.dropout_after_encoder is not None:
            self.dropout_after_encoder_layer = nn.Dropout(
                self.dropout_after_encoder)

        self.latent_channels, self.latent_length = self.encoder.get_latent_shape(
            self.encoder.expected_input_length)
        self.latent_size = self.latent_channels * self.latent_length
        assert self.latent_size >= MIN_LATENT_SIZE

        lin_ftrs = [self.latent_size] + hidden_lin_ftrs + [number_of_classes]

        # Forces FC layers sizes to be nonincreasing
        if force_nonincreasing_lin_ftrs:
            for l1, l2 in zip(lin_ftrs, lin_ftrs[1:]):
                if l1 < l2:
                    raise IllegalArgument(
                        "FC layer sizes should be nonincreasing.")

        it_lin_ftrs = iter(zip(lin_ftrs, lin_ftrs[1:]))

        in_dim, out_dim = next(it_lin_ftrs)
        classification_head_layers = [nn.Linear(in_dim, out_dim)]
        for in_dim, out_dim in it_lin_ftrs:
            classification_head_layers += [nonlin,
                                           nn.BatchNorm1d(in_dim),
                                           nn.Linear(in_dim, out_dim)]

        self.classification_head = nn.Sequential(*classification_head_layers)

    def forward(self, *args, **kwargs):
        latent = self.encoder(*args, **kwargs)
        if self.dropout_after_encoder is not None:
            latent = self.dropout_after_encoder_layer(latent)

        bs = latent.shape[0]
        # torch.max(latent, dim=1).values).squeeze(1)
        return self.classification_head(latent.view(bs, -1)).squeeze(1)

    def count_num_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleFCClassificationModel(nn.Module):
    def __init__(self,
                 channels=1,
                 number_of_classes=2,
                 nonlin=nn.ReLU(inplace=True),
                 # string encoding nonlin func - not used directly, but passed as a hyperparameter
                 nonlin_str=None,
                 hidden_lin_ftrs=[50],
                 hidden_lin_ftrs_no=None,  # if not None, then assume int for hidden_lin_ftrs
                 dropout=0.0,
                 ):
        super().__init__()

        assert nonlin_str is not None
        
        # use all hidden layers of same given size
        if hidden_lin_ftrs_no is not None:
            assert isinstance(hidden_lin_ftrs, int)
            hidden_lin_ftrs = [hidden_lin_ftrs] * hidden_lin_ftrs_no

        lin_ftrs = [channels] + hidden_lin_ftrs + [number_of_classes]

        it_lin_ftrs = iter(zip(lin_ftrs, lin_ftrs[1:]))

        in_dim, out_dim = next(it_lin_ftrs)
        classification_head_layers = [nn.Linear(in_dim, out_dim)]
        for in_dim, out_dim in it_lin_ftrs:
            classification_head_layers += [nonlin,
                                           # nn.BatchNorm1d(in_dim),
                                           nn.Dropout(dropout),
                                           nn.Linear(in_dim, out_dim)]

        self.classification_head = nn.Sequential(*classification_head_layers)

    def forward(self, x):
        return self.classification_head(x)

    def count_num_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    pass
