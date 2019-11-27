import torch.nn as nn
from model.vqvae.modules import ResBlock

# Mel Decoder Inspired by: https://github.com/swasun/VQ-VAE-Speech
# Code Adapted from: https://github.com/rosinality/vq-vae-2-pytorch
class MelDecoder(nn.Module):
    def __init__(
        self, in_channel, channel, out_channel, stride
    ):
        super(MelDecoder, self).__init__()

        blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]

        if stride == 2:
            blocks.extend([
                ResBlock(channel, channel),
                ResBlock(channel, channel),
                nn.LeakyReLU(inplace=True),

                nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
                nn.BatchNorm1d(channel),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channel),

                ResBlock(out_channel, out_channel),
                nn.LeakyReLU(inplace=True),
            ])
        elif stride == 4:
             blocks.extend([
                ResBlock(channel, channel),
                ResBlock(channel, channel),
                nn.LeakyReLU(inplace=True),

                nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
                nn.BatchNorm1d(channel),
                nn.ReLU(inplace=True),

                nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
                nn.BatchNorm1d(channel),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channel),

                ResBlock(out_channel, out_channel),
                nn.LeakyReLU(inplace=True),
            ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super(Decoder, self).__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)