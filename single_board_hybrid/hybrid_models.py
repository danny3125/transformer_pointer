import math
import numpy as np
import torch.nn.functional as F
import random
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import torch
import torch.nn as nn
import time
import argparse
import os
import datetime

from torch.distributions.categorical import Categorical

# visualization
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings


class TransEncoderNet(nn.Module):
    """
    Encoder network based on self-attention transformer
    Inputs :
      h of size      (bsz, nb_nodes, dim_emb)    batch of input cities
    Outputs :
      h of size      (bsz, nb_nodes, dim_emb)    batch of encoded cities
      score of size  (bsz, nb_nodes, nb_nodes+1) batch of attention scores
    """

    def __init__(self, nb_layers, dim_emb, nb_heads, dim_ff, batchnorm):
        super(TransEncoderNet, self).__init__()
        assert dim_emb == nb_heads * (dim_emb // nb_heads)  # check if dim_emb is divisible by nb_heads
        self.MHA_layers = nn.ModuleList([nn.MultiheadAttention(dim_emb, nb_heads) for _ in range(nb_layers)])
        self.linear1_layers = nn.ModuleList([nn.Linear(dim_emb, dim_ff) for _ in range(nb_layers)])
        self.linear2_layers = nn.ModuleList([nn.Linear(dim_ff, dim_emb) for _ in range(nb_layers)])
        if batchnorm:
            self.norm1_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)])
            self.norm2_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)])
        else:
            self.norm1_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(nb_layers)])
            self.norm2_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(nb_layers)])
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.batchnorm = batchnorm

    def forward(self, h):
        # PyTorch nn.MultiheadAttention requires input size (seq_len, bsz, dim_emb)
        h = h.transpose(0, 1)  # size(h)=(nb_nodes, bsz, dim_emb)
        # L layers
        for i in range(self.nb_layers):
            h_rc = h  # residual connection, size(h_rc)=(nb_nodes, bsz, dim_emb)
            h, score = self.MHA_layers[i](h, h,
                                          h)  # size(h)=(nb_nodes, bsz, dim_emb), size(score)=(bsz, nb_nodes, nb_nodes)
            # add residual connection

            h = h_rc + h  # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                # Pytorch nn.BatchNorm1d requires input size (bsz, dim, seq_len)
                h = h.permute(1, 2, 0).contiguous()  # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm1_layers[i](h)  # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2, 0, 1).contiguous()  # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm1_layers[i](h)  # size(h)=(nb_nodes, bsz, dim_emb)
            # feedforward
            h_rc = h  # residual connection
            h = self.linear2_layers[i](torch.relu(self.linear1_layers[i](h)))
            h = h_rc + h  # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                h = h.permute(1, 2, 0).contiguous()  # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm2_layers[i](h)  # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2, 0, 1).contiguous()  # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm2_layers[i](h)  # size(h)=(nb_nodes, bsz, dim_emb)
        # Transpose h
        h = h.transpose(0, 1)  # size(h)=(bsz, nb_nodes, dim_emb)
        return h, score


class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden

        v = torch.FloatTensor(n_hidden)
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-1 / math.sqrt(n_hidden), 1 / math.sqrt(n_hidden))

        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)

    def forward(self, q, ref):  # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)  # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)

        q_ex = q.unsqueeze(1).repeat(1, self.size, 1)  # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)

        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)

        return u, ref


class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()

        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)  # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)  # w(ct)

        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)  # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)  # w(ct)

        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)  # W(ht)

        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)  # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)  # w(ct)

    def forward(self, x, h, c):  # query and reference

        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))

        h = o * torch.tanh(c)

        return h, c


class HPN(nn.Module):
    def __init__(self, n_feature, n_hidden):

        super(HPN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden

        # lstm for first turn
        # self.lstm0 = nn.LSTM(n_hidden, n_hidden)

        # pointer layer
        self.pointer = Attention(n_hidden)
        self.TransPointer = Attention(n_hidden)

        # lstm encoder
        self.encoder = LSTM(n_hidden)

        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden)
        c0 = torch.FloatTensor(n_hidden)

        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()

        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)

        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1 / math.sqrt(n_hidden), 1 / math.sqrt(n_hidden))
        self.c0.data.uniform_(-1 / math.sqrt(n_hidden), 1 / math.sqrt(n_hidden))

        r1 = torch.ones(1)
        r2 = torch.ones(1)
        r3 = torch.ones(1)
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)

        # embedding
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)
        self.Transembedding_all = TransEncoderNet(6, 128, 8, 512, batchnorm=True)

        # vector to start decoding
        self.start_placeholder = nn.Parameter(torch.randn(n_hidden))

        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)

        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
        self.outagg = nn.Linear(2, 1)

    def forward(self, context, Transcontext, x, X_all, mask, h=None, c=None, latent=None):
        '''
        Inputs (B: batch size, size: city size, dim: hidden dimension)

        x: current city coordinate (B, 2)
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)

        Outputs

        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        '''

        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)

        # Check if this the first iteration loop
        if h is None or c is None:
            x = self.start_placeholder
            context = self.embedding_all(X_all)
            Transcontext, _ = self.Transembedding_all(context)

            # =============================
            # graph neural network encoder
            # =============================

            # (B, size, dim)
            context = context.reshape(-1, self.dim)
            Transcontext = Transcontext.reshape(-1, self.dim)

            context = self.r1 * self.W1(context) \
                      + (1 - self.r1) * F.relu(self.agg_1(context / (self.city_size - 1)))

            context = self.r2 * self.W2(context) \
                      + (1 - self.r2) * F.relu(self.agg_2(context / (self.city_size - 1)))

            context = self.r3 * self.W3(context) \
                      + (1 - self.r3) * F.relu(self.agg_3(context / (self.city_size - 1)))
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()

            # let h0, c0 be the hidden variable of first turn
            h = h0.squeeze(0)
            c = c0.squeeze(0)
        else:
            x = self.embedding_x(x)
        # LSTM encoder
        h, c = self.encoder(x, h, c)
        # query vector
        q = h
        # pointer
        u1, _ = self.pointer(q, context)
        u2, _ = self.TransPointer(q, Transcontext)
        # Avg Agg between the two attention vectors
        u = (u1 + u2) / 2
        latent_u = u.clone()
        u = 10 * torch.tanh(u) + mask
        return context, Transcontext, F.softmax(u, dim=1), h, c, latent_u

