import numpy as np
import torch.nn as nn
from model.melgan.modules import WNConv1d, WNConvTranspose1d, ResnetBlock
from model.modules import weights_init

class Generator(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        # In original paper, input_size == n_mel_channels
        # ngf is a model hyperparameter, meaning the final number of feature maps in Generator, 32 in paper.
        super().__init__()
        # ratios = [8, 8, 2, 2] # 4 stages of upsampling: 8x, 8x, 2x, 2x --> 256x (hop_length when calculating Mel Spectrogram)
        ratios = [8, 8, 4, 2, 2] # Note(houwx): Modify to 8x, 8x, 4x, 2x, 2x here
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios)) # 16

        model = [
            nn.ReflectionPad1d(3), # Padding on left & right: (N, n_mel, T_mel) --> (N, n_mel, T_mel + 3 left + 3 right)
            WNConv1d(in_channels=input_size, out_channels=mult * ngf, kernel_size=7, padding=0), # (N, n_mel=80, T_mel + 6) --> (N, ngf * mult=16 * 32, T_mel)
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2, # [16, 16, 4, 4]
                    stride=r, # [8, 8, 2, 2]
                    padding=r // 2 + r % 2, # [4, 4，1, 1]
                    output_padding=r % 2, # All 0s
                ), # First upsample as example: (N, ngf * mult=16*32, T_mel) --> (N, ngf * mult // 2=16*16, (T_mel - 1)* r - r + r * 2 - 1 + 1 = T_mel * r)
            ] 

            for j in range(n_residual_layers): # [0 , 1, 2]
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)] 
                # No Change in shape. First ResBlock: (N, ngf * mult / 2, T_mel * r) --> (N, ngf * mult / 2, T_mel * r)

            mult //= 2
        # After 4 stages, get: (N, ngf, T_mel * 256)

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3), # (N, ngf, T_mel * 256) -- > (N, ngf, T_mel * 256 + 3 + 3)
            WNConv1d(ngf, 1, kernel_size=7, padding=0), # (N, ngf, T_mel * 256 + 6) --> (N, 1, T_mel * 256)
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        ) # (N, 1, T_in) --> (N, ndf, T_in)

        nf = ndf # # ndf is a model hyperparameter, meaning the final number of feature maps in Discriminator, 16 in paper.
        stride = downsampling_factor # 4 in paper.
        for n in range(1, n_layers + 1): # [1, 2, 3, 4]
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            ) # (N, nf_prev, T_in) --> (N, nf_prev * 4, (T_in - 1) / stride + 1) TODO Problem??? Maybe should be (N, nf_prev * 4, T_in / 4)
        # After 4 iterations: (N, nf_prev * 256, T_in / 256)
        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        ) # Shape no change

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        ) # Shape no change TODO Why no ReLU here?

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor): # numD = 3, n_layers=4 in paper.
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            ) # (N, 1, T_mel * 256) --> (N, 1 * 256, T_mel)

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x) # (N, 1, T) --> (N, 1, T / 2)
        return results