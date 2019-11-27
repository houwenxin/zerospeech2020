import torch.nn as nn
from model.vqvae.modules import ResBlock

# Mel Encoder Inspired by: https://github.com/swasun/VQ-VAE-Speech
# Code Adapted from: https://github.com/rosinality/vq-vae-2-pytorch
class MelEncoder(nn.Module):
    def __init__(self, in_channel, channel, stride):
        super(MelEncoder, self).__init__()
        # Input: (B, n_mel, T)
        if stride == 2:
            blocks = [
                nn.Conv1d(in_channels=in_channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(channel),
                # After: (B, channel, T)

                ResBlock(channel, channel),
                nn.LeakyReLU(inplace=True),
                # After: (B, channel, T)

                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(channel),
                # After: (B, channel, T/2)

                ResBlock(channel, channel),
                ResBlock(channel, channel),
                
                nn.LeakyReLU(inplace=True),
                # After: (B, channel, T/2)
            ]
        elif stride == 4:
            blocks = [
                nn.Conv1d(in_channels=in_channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(channel),
                # After: (B, channel, T)

                ResBlock(channel, channel),
                nn.LeakyReLU(inplace=True),
                # After: (B, channel, T)

                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(channel),
                nn.LeakyReLU(inplace=True),
                # After: (B, channel, T/2)
                
                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(channel),
                # After: (B, channel, T/4)
                
                ResBlock(channel, channel),
                ResBlock(channel, channel),

                nn.LeakyReLU(inplace=True),
                # After: (B, channel, T/4)
            ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

# Origin Encoder in VQVAE-2: https://github.com/rosinality/vq-vae-2-pytorch
class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super(Encoder, self).__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


if __name__ == "__main__":
    import torch
    X = torch.randn((1, 128, 80)) # B, T, n_mel
    encoder = MelEncoder(in_channel=80, channel=1024, stride=2)
    lin = encoder.forward(X.permute(0, 2, 1))
    print(lin.shape)
