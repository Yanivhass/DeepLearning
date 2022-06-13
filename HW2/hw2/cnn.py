import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        self.num_of_params = self.in_size
        layers = []
        dims = [self.in_size[0]] + self.channels
        for i in range(len(self.channels)):
            # Define all parameters
            p = self.conv_params['padding']
            s = self.conv_params['stride']
            d = self.conv_params.get('dialation', 1)
            f = self.conv_params['kernel_size']
            c = dims[i + 1]
            h = int(torch.floor(torch.tensor((self.num_of_params[1] + 2 * p - d * (f - 1) - 1) / s)) + 1)
            w = int(torch.floor(torch.tensor((self.num_of_params[2] + 2 * p - d * (f - 1) - 1) / s)) + 1)

            # Stack layers
            layers += [torch.nn.Conv2d(dims[i], dims[i + 1], **self.conv_params)]
            layers += [ACTIVATIONS[self.activation_type](**self.activation_params)]

            self.num_of_params = (c, h, w)
            if torch.remainder(torch.tensor(i + 1),torch.tensor(self.pool_every)) == 0:
                p = self.pooling_params['kernel_size']
                c = self.num_of_params[0]
                h = int(torch.floor(torch.tensor(self.num_of_params[1] / p)))
                w = int(torch.floor(torch.tensor(self.num_of_params[2] / p)))
                layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
                self.num_of_params = (c, h, w)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            return self.num_of_params[0] * self.num_of_params[1] * self.num_of_params[2]
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        if self.activation_type in ACTIVATION_DEFAULT_KWARGS:
            activation = self.activation_type
        else:
            activation = ACTIVATIONS[self.activation_type](**self.activation_params)

        mlp = MLP(in_dim=int(self._n_features()),
                  dims=self.hidden_dims + [self.out_classes],
                  nonlins=[activation] * len(self.hidden_dims) + ['none'])

        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        encoding = self.feature_extractor(x)
        out = self.mlp(encoding.reshape(encoding.shape[0], -1))
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        self.main_path = []
        dims = [in_channels] + channels
        for i,kernel_size in enumerate(kernel_sizes):
            pad = int(torch.floor(torch.tensor(kernel_size) / torch.tensor(2)))
            kernel_params = dict(kernel_size=kernel_size, padding=pad)
            self.main_path += [torch.nn.Conv2d(dims[i], dims[i + 1], **kernel_params)]
            if i != len(channels) - 1:
                if dropout:
                    self.main_path += [nn.Dropout2d(dropout)]
                if batchnorm:
                    self.main_path += [nn.BatchNorm2d(dims[i + 1])]
                self.main_path += [ACTIVATIONS[activation_type](**activation_params)]
        self.main_path = torch.nn.Sequential(*self.main_path)

        self.shortcut_path = []
        if channels[-1] != in_channels:
            self.shortcut_path += [torch.nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False)]
        self.shortcut_path = torch.nn.Sequential(*self.shortcut_path)
        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.main_path(x) + self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        in_kernel_sizes = [1] + inner_kernel_sizes + [1]
        super().__init__(channels=[inner_channels[0]] + inner_channels + [in_out_channels],
                         in_channels=in_out_channels,
                         kernel_sizes=in_kernel_sizes, **kwargs)
        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======

        self.num_of_params = self.in_size
        layers = []
        dims = [self.in_size[0]] + self.channels
        num_of_blocks = torch.floor(torch.tensor(len(self.channels) / self.pool_every))

        for i in range(num_of_blocks.int()):
            p = self.conv_params.get('padding', 1)
            f = self.conv_params.get('kernel_size', 3)
            d = self.conv_params.get('dialation', 1)
            s = self.conv_params.get('stride', 1)

            res_channels = self.channels[i * self.pool_every:(i + 1) * self.pool_every]

            if self.bottleneck and self.num_of_params[0] == res_channels[-1]:
                resblock = ResidualBottleneckBlock(
                    in_out_channels=self.num_of_params[0], inner_channels=res_channels[1:-1],
                    inner_kernel_sizes=[3] * len(res_channels[1:-1]),
                    batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                    activation_params=self.activation_params
                )
                c = self.num_of_params[0]

            else:
                resblock = ResidualBlock(
                    in_channels=self.num_of_params[0], channels=res_channels, kernel_sizes=[3] * len(res_channels),
                    batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                    activation_params=self.activation_params
                )
                c = res_channels[-1]

            layers += [resblock]

            h = self.num_of_params[1]
            w = self.num_of_params[2]
            self.num_of_params = (c, h, w)

            # Pooling
            p = self.pooling_params['kernel_size']
            c = self.num_of_params[0]
            h = int(torch.floor(torch.tensor(self.num_of_params[1] / p)))
            w = int(torch.floor(torch.tensor(self.num_of_params[2] / p)))
            layers += [POOLINGS[self.pooling_type](**self.pooling_params)]

            self.num_of_params = (c, h, w)

        last_block_size = torch.remainder(torch.tensor(len(self.channels)), torch.tensor(self.pool_every))
        last_res_channels = self.channels[-last_block_size:]
        if last_block_size > 0:
            d = self.conv_params.get('dialation', 1)
            f = self.conv_params.get('kernel_size', 3)
            s = self.conv_params.get('stride', 1)
            p = self.conv_params.get('padding', 1)

            if self.bottleneck and self.num_of_params[0] == res_channels[-1]:
                resblock = ResidualBottleneckBlock(
                    in_out_channels=self.num_of_params[0], inner_channels=last_res_channels[1:-1],
                    inner_kernel_sizes=[3] * len(last_res_channels[1:-1]),
                    batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                    activation_params=self.activation_params
                )
                c = self.num_of_params[0]
            else:
                resblock = ResidualBlock(
                    in_channels=self.num_of_params[0], channels=last_res_channels,
                    kernel_sizes=[3] * len(last_res_channels),
                    batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                    activation_params=self.activation_params
                )
                c = last_res_channels[-1]

            layers += [resblock]
            h = self.num_of_params[1]
            w = self.num_of_params[2]
            self.num_of_params = (c, h, w)
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCNN_original(CNN):

    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            bottleneck: bool = False,
            **kwargs,
    ):


        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )
        self.mlp = torch.nn.Sequential(
            # torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            self.mlp
        )
        # ========================

    # TODO: Change whatever you want about the CNN to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======
        self.num_of_params = self.in_size
        layers = []
        dims = [self.in_size[0]] + self.channels
        blocks = torch.tensor(self.channels).split(2)

        for block_indx, block in enumerate(blocks, 1):
            block_channels = block.tolist()
            resblock = ResidualBlock(
                in_channels=self.num_of_params[0], channels=block_channels, kernel_sizes=[3] * len(block_channels),
                batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                activation_params=self.activation_params
            )
            C = block_channels[-1]
            layers += [resblock]

            H = self.num_of_params[1]
            W = self.num_of_params[2]
            self.num_of_params = (C, H, W)

            # Pooling
            if block_indx % self.pool_every == 0 and block_indx != len(blocks):
                P = self.pooling_params['kernel_size']
                layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
                C = self.num_of_params[0]
                H = int(torch.floor(torch.tensor(self.num_of_params[1] / P)))
                W = int(torch.floor(torch.tensor(self.num_of_params[2] / P)))
                self.num_of_params = (C, H, W)

        # Max pool as final stage before FC layer
        layers += [POOLINGS['max'](kernel_size=self.num_of_params[1])]

        P = self.num_of_params[1]
        C = self.num_of_params[0]
        H = int(torch.floor(torch.tensor(self.num_of_params[1] / P)))
        W = int(torch.floor(torch.tensor(self.num_of_params[2] / P)))
        self.num_of_params = (C, H, W)

        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================

    def forward(self, x: Tensor):
        out: Tensor = None
        encoding = self.feature_extractor(x)
        out = self.mlp(encoding)
        return out


class YourCNN(CNN):

    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            bottleneck: bool = False,
            **kwargs,
    ):


        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )
        self.mlp = torch.nn.Sequential(
            # torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            self.mlp
        )
        # ========================

    # TODO: Change whatever you want about the CNN to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======
        self.num_of_params = self.in_size

        dims = [self.in_size[0]] + self.channels
        blocks = torch.tensor(self.channels).split(2)

        for block_indx, block in enumerate(blocks, 1):
            block_channels = block.tolist()
            resblock = ResidualBlock(
                in_channels=self.num_of_params[0], channels=block_channels, kernel_sizes=[3] * len(block_channels),
                batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                activation_params=self.activation_params
            )
            C = block_channels[-1]
            layers += [resblock]

            H = self.num_of_params[1]
            W = self.num_of_params[2]
            self.num_of_params = (C, H, W)

            # Pooling
            if block_indx % self.pool_every == 0 and block_indx != len(blocks):
                P = self.pooling_params['kernel_size']
                layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
                C = self.num_of_params[0]
                H = int(torch.floor(torch.tensor(self.num_of_params[1] / P)))
                W = int(torch.floor(torch.tensor(self.num_of_params[2] / P)))
                self.num_of_params = (C, H, W)

        # Max pool as final stage before FC layer
        layers += [POOLINGS['max'](kernel_size=self.num_of_params[1])]

        P = self.num_of_params[1]
        C = self.num_of_params[0]
        H = int(torch.floor(torch.tensor(self.num_of_params[1] / P)))
        W = int(torch.floor(torch.tensor(self.num_of_params[2] / P)))
        self.num_of_params = (C, H, W)

        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================

    def forward(self, x: Tensor):
        out: Tensor = None
        encoding = self.feature_extractor(x)
        out = self.mlp(encoding)
        return out


