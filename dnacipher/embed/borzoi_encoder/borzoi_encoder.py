""" Adapted from borzoi-pytorch:

    https://github.com/johahi/borzoi-pytorch/tree/main

But made minimalist, so that the goal is just to extract sequence embeddings of arbitrary lengths, for training sequence
models for other tasks.
"""

import torch
import torch.nn.init as init

import numpy as np

from pathlib import Path

from safetensors import safe_open

from dnacipher.embed.borzoi_encoder.borzoi_transformer import Attention

def load_safetensor_weights(path, device, keys=None):
    # Initialize an empty dictionary to store the loaded tensors
    tensors = {}

    # Open the file in read mode ('pt' for PyTorch framework)
    # and iterate through the keys to load each tensor
    with safe_open(path, framework="pt", device=device) as f:
        keys_ = list(f.keys())
        if type(keys)!=type(None):
            keys = [key for key in keys_ if np.any([key_ in key for key_ in keys])]
        else:
            keys = keys_

        for key in keys:
            tensors[key] = f.get_tensor(key)

    return tensors

class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ConvDna(torch.nn.Module):
    def __init__(self):
        super(ConvDna, self).__init__()
        self.conv_layer = torch.nn.Conv1d(in_channels = 4,out_channels = 512, kernel_size = 15, padding="same")
        self.max_pool = torch.nn.MaxPool1d(kernel_size = 2, padding = 0)

    def forward(self, x):
        return self.max_pool(self.conv_layer(x))

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1,
                 conv_type="standard"):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = torch.nn.Identity()
            depthwise_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=in_channels,
                                       padding='same', bias=False)
            pointwise_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = torch.nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = torch.nn.Identity()
        else:
            self.norm = torch.nn.BatchNorm1d(in_channels, eps=0.001)
            self.activation = torch.nn.GELU(approximate='tanh')
            self.conv_layer = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding='same')

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x

class ResidMLP(torch.nn.Module):
    def __init__(self, n_layers, latent_dims, n_in, n_out, init_bias, drop):

        super(ResidMLP, self).__init__()

        self.activation = torch.nn.GELU()
        self.drop = drop

        self.dense_layers = torch.nn.ModuleList([])
        if n_layers > 0:

            layeri = torch.nn.Linear(n_in, latent_dims)
            layeri = torch.nn.Sequential(layeri, self.activation, torch.nn.LayerNorm(latent_dims))
            self.dense_layers.append(layeri)

            for i in range(n_layers - 1):
                layeri = torch.nn.Linear(latent_dims, latent_dims)
                layeri = torch.nn.Sequential(layeri, self.activation, torch.nn.LayerNorm(latent_dims))
                self.dense_layers.append(layeri)

            self.linear = torch.nn.Linear(latent_dims, n_out, bias=True)
            self.dense_layers.append(self.linear)

        else:
            # Literally just flatten the CNN output and regress.
            self.linear = torch.nn.Linear(n_in, n_out, bias=True)
            self.dense_layers.append(self.linear)

        if type(init_bias) == type(None):
            init.zeros_(self.linear.bias)  # Setting this to 0, which should be the main PIP value.
        else:
            self.linear.bias.data = torch.tensor(init_bias)  # .log().softmax(-1)

    def forward(self, X):
        """ MLP forward.
        """

        x_out_flat = X

        for i, layer in enumerate(self.dense_layers):

            out_ = layer(x_out_flat)
            if out_.shape == x_out_flat.shape:
                x_out_flat = torch.add(out_, x_out_flat)  # Added residual connections to make deeper.
            else:
                x_out_flat = out_

            if i < (len(self.dense_layers) - 1):
                x_out_flat = self.drop( x_out_flat )

        if len(x_out_flat.shape)==3:
            x_out_flat = x_out_flat.squeeze(2)

        return x_out_flat


class BorzoiEncoder( torch.nn.Module ):
    """ Linear model baseline.
    """

    def __init__(self, input_seqlen=2048, load_weights=True, deep=True, device='cpu'):
        super(BorzoiEncoder, self).__init__()

        self.name = 'borzoi_encoder'

        self.input_seqlen = input_seqlen

        # Controls what inputs the model will take, this actually get set during training, depending on the input the
        # model receives.
        self.deep = deep
        self.device = device

        # This is the layer that we take from Borzoi !!!! NOTE they normally do a stride 2 max pooling, but I won't.
        self.conv_dna = ConvDna()
        self.conv = self.conv_dna.conv_layer

        ### Also utilizing the resnet tower!
        self._max_pool = torch.nn.MaxPool1d(kernel_size = 2, padding = 0)
        self.res_tower = torch.nn.Sequential(
            ConvBlock(in_channels = 512,out_channels = 608, kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 608,out_channels = 736, kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 736,out_channels = 896, kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 896,out_channels = 1056, kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 1056,out_channels = 1280, kernel_size = 5),
        )

        if not self.deep:
            n_filters = 1280 # This is now hardcoded because of the Borzoi model weights.
        else:
            # Adding additional Borzoi components, essentially so we are forward passing
            # Until before the final convulational layer that converts the values to the tracks.
            # These are the constants from the Borzoi config:
            config_dim = 1536
            config_depth = 8
            config_heads = 8
            config_attn_dim_key = 64
            config_attn_dim_value = 192
            config_attn_dropout = 0.05
            config_pos_dropout = 0.01
            config_dropout_rate = 0.2

            self.unet1 = torch.nn.Sequential(
                self._max_pool,
                ConvBlock(in_channels=1280, out_channels=config_dim, kernel_size=5),
            )
            transformer = []
            for _ in range( config_depth ): # transformer depth.
                transformer.append(torch.nn.Sequential(
                    Residual(torch.nn.Sequential(
                        torch.nn.LayerNorm(config_dim, eps=0.001),
                        Attention(
                            config_dim,
                            heads=config_heads,
                            dim_key=config_attn_dim_key,
                            dim_value=config_attn_dim_value,
                            dropout=config_attn_dropout,
                            pos_dropout=config_pos_dropout,
                            num_rel_pos_features=32
                        ),
                        torch.nn.Dropout(0.2))
                    ),
                    Residual(torch.nn.Sequential(
                        torch.nn.LayerNorm(config_dim, eps=0.001),
                        torch.nn.Linear(config_dim, config_dim * 2),
                        torch.nn.Dropout(config_dropout_rate),
                        torch.nn.ReLU(),
                        torch.nn.Linear(config_dim * 2, config_dim),
                        torch.nn.Dropout(config_dropout_rate)
                    )))
                )
            self.horizontal_conv0, self.horizontal_conv1 = ConvBlock(in_channels=1280, out_channels=config_dim,
                                                                     kernel_size=1), ConvBlock(in_channels=config_dim,
                                                                                               out_channels=config_dim,
                                                                                               kernel_size=1)
            self.upsample = torch.nn.Upsample(scale_factor=2)
            self.transformer = torch.nn.Sequential(*transformer)
            self.upsampling_unet1 = torch.nn.Sequential(
                ConvBlock(in_channels=config_dim, out_channels=config_dim, kernel_size=1),
                self.upsample,
            )
            self.separable1 = ConvBlock(in_channels=config_dim, out_channels=config_dim, kernel_size=3,
                                        conv_type='separable')
            self.upsampling_unet0 = torch.nn.Sequential(
                ConvBlock(in_channels=config_dim, out_channels=config_dim, kernel_size=1),
                self.upsample,
            )
            self.separable0 = ConvBlock(in_channels=config_dim, out_channels=config_dim, kernel_size=3,
                                        conv_type='separable')

            # This is the final expected output now, with the deeper representation.
            n_filters = config_dim

        # Downloaded the Borzoi model weights, trained by Johannes, from here:
        # https://huggingface.co/johahi/borzoi-replicate-0/tree/main
        self.embed_dim = n_filters

        if load_weights:
            path = Path(__file__).parent
            # Just extracting the weights which we will re-use here.
            keys_ = ['conv_dna', 'res_tower']
            if self.deep:
                keys_.extend( ['unet1', 'horizontal_conv0', 'horizontal_conv1', 'upsample', 'transformer',
                               'upsampling_unet1', 'separable1', 'upsampling_unet0', 'separable0']
                            )
                # Just added this so I can get the parameter count.
                self.bz_deep_layers = torch.nn.Sequential(self.unet1, self._max_pool, self.horizontal_conv1,
                                                          self.horizontal_conv0, self.transformer,self.upsampling_unet1,
                                                          self.separable1, self.upsampling_unet0, self.separable0
                                                          )

            borzoi_conv_weights = load_safetensor_weights(f'{path}/model.safetensors', str(self.device),
                                                         keys=keys_,
                                                         )
            ###### Loading the weights!!!
            for component_name in keys_:
                component = getattr(self, component_name)
                current_weights = component.state_dict()
                new_weights = {key_: borzoi_conv_weights[f"{component_name}.{key_}"] for key_ in current_weights}
                component.load_state_dict( new_weights ) # Adding these weights to the component.

                # Freezing the weights!
                for param in component.parameters():
                    param.requires_grad = False

    def _forward_deep(self, x_unet0):
        """
        Performs the forward pass of the model until right before the final conv layers, and includes a cropping layer.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).

        Returns:
             torch.Tensor: Output of the model up to the cropping layer with shape (N, dim, crop_length)
        """
        # in: (seqs, 1280, 32), for seqlen input 1024 (2**10)
        x_unet1 = self.unet1(x_unet0) # x_out.shape = (seqs, 1536, 16), so channels up 1.2x, and the seqlen now 64x smaller, so 64bp bins...
        x = self._max_pool(x_unet1) # x_out.shape = (seqs, 1536, 8), so 128x reduction from original seqlen, max_pool kept number of channels.
        x_unet1 = self.horizontal_conv1(x_unet1) # x_out.shape = (seqs, 1536, 16), so blew it back up again.. this is the U-net architecture in action.
        x_unet0 = self.horizontal_conv0(x_unet0) # x_out.shape = (seqs, 1536, 32), again, below it back up to the output of the resnet, so this is U-net again.
        x = self.transformer(x.permute(0,2,1)) # x_out.shape = (seqs, 1536, 8), switched places, but now a 128bp reduction in channels.
        x = x.permute(0,2,1) # x_out.shape = (seqs, 16, 1536), so they just flip it back, not sure why.
        x = self.upsampling_unet1(x) # x_out.shape = (seqs, 1536, 16), so this is now same shape as x_unet1, on both side of the unet.
        x += x_unet1 # x_out.shape = (seqs, 1536, 16), so this is a residual connection - adding the output of the x_unet1 output to the upsampled transformer output..
        x = self.separable1(x) # x_out.shape = (seqs, 1536, 16), not sure what this did, output same size.
        x = self.upsampling_unet0(x) # x_out.shape = (seqs, 1536, 32), now same size as the x_unet0 either side of the unet.
        x += x_unet0 # x_out.shape = (seqs, 1536, 32), so this is now the same as the last and first parts of the unet, so another residual connection from the first to this last layer.
        x = self.separable0(x) # x_out.shape = (seqs, 1536, 32), not sure what this did...
        #x = x.permute(0,2,1) # x_out.shape = (1, 16384, 1536), flipped the channels and seq positions again... at this point, we are at 32x smaller than input sequence, correspond to 32bp, so the output resolution.
        #x = self.crop(x) # x_out.shape = (1, 6144, 1536), cropped the sequences, so guessing this just cuts of the represented seq positions either side. So at the 32bp resolution, this would mean the target output is actually 196,608bp.
        #x = x.permute(0,2,1) # x_out.shape = (1, 1536, 6144), flipping positions and channels again!
        return x

    def forward(self, X):
        """ Forward pass through the fixed borzoi components to extract sequence features.
        """
        x_out = self.conv_dna(X)
        x_out = self.res_tower(x_out)

        if self.deep:
            x_out = self._forward_deep( x_out )

        return x_out