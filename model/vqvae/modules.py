import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out