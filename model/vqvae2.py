import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()
        self.VQLayer = VQLaer() # TODO
        self.decoder = Decoder()