import torch
from torch import nn
from torch.nn import functional as F

from model.vqvae.encoder import MelEncoder as Encoder
from model.vqvae.decoder import MelDecoder as Decoder

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

# Adapted by houwx to the ZeroSpeech 2020

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super(Quantize, self).__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed # a @ b --> matmul(a, b)
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        
        if self.training: # In nn.Module: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super(VQVAE, self).__init__()
        # enc_b: Bottom Encoder & enc_t: Top Encoder
        self.enc_b = Encoder(in_channel, channel, stride=4) 
        self.enc_t = Encoder(channel, channel, stride=2)
        self.quantize_conv_t = nn.Conv1d(in_channels=channel, out_channels=embed_dim, kernel_size=1)

        self.quantize_t = Quantize(embed_dim, n_embed)
        
        self.dec_t = Decoder(in_channel=embed_dim, channel=channel, out_channel=embed_dim, stride=2)

        self.quantize_conv_b = nn.Conv1d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose1d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(in_channel=embed_dim + embed_dim, channel=channel, out_channel=in_channel, stride=4)
        

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input) # Return: (B, channel, T/4)
        enc_t = self.enc_t(enc_b) # Return: (B, channel, T/8)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 1) # Return: (B, T/8, embed_dim)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)

        quant_t = quant_t.permute(0, 2, 1) # Return: (B, embed_dim, T/8)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t) # Return: (B, embed_dim, T/8)
        
        enc_b = torch.cat([dec_t, enc_b], 1) # Return: (B, embed_dim + channel, T/8)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 1) # Return: (B, T/8, embed_dim)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 2, 1) # Return: (B,  embed_dim, T/8)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t) # Return: (B, embed_dim, T/4)
        quant = torch.cat([upsample_t, quant_b], 1) # Return: (B, 2 * embed_dim, T/4)
        dec = self.dec(quant) # Return: (B, in_channel, T/4)
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 2, 1)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 2, 1)
        #print(quant_t)
        dec = self.decode(quant_t, quant_b)

        return dec

if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn((1, 128, 80)) # B, T, n_mel
    vqvae = VQVAE(in_channel=80, channel=1024)
    vqvae.eval()

    quant_t, quant_b, diff, id_t, id_b = vqvae.encode(X.permute(0, 2, 1))
    print(X.shape)
    dec_1 = vqvae.decode(quant_t, quant_b)
    #vqvae.forward(X.permute(0, 2, 1))
    dec_2 = vqvae.decode_code(id_t, id_b)
    print(dec_1.shape)
    # dec_1 and dec_2 have some trival differences under evaluation mode.