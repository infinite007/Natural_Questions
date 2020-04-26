import torch.nn as nn
import torch.nn.functional as F


class HIGHWAYNetworks(nn.Module):
    def __init__(self, embedding_dimension):
        super(HIGHWAYNetworks, self).__init__()
        self.w_h = nn.Linear(embedding_dimension, embedding_dimension)
        self.w_t = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x):
        h_x_w = self.w_h(x)
        h_x_w = F.tanh(h_x_w)
        t_x_w = self.w_t(x)
        t_x_w = F.sigmoid(t_x_w)

        transform = h_x_w * t_x_w
        carry = x * (1 - t_x_w)

        return transform + carry

