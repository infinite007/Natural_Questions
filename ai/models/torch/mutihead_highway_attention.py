import torch
import torch.nn as nn
import torch.nn.functional as F
from .highway_networks import HIGHWAYNetworks


class MULTI_HEAD_HIGHWAY_ATTENTION_Networks(nn.Module):
    def __init__(self, embedding_dimension, num_att_heads=16):
        super().__init__()
        out_dimension = embedding_dimension
        if num_att_heads == 0:
            out_dimension = embedding_dimension
        elif num_att_heads % 2 == 0:
            out_dimension = embedding_dimension // num_att_heads
        else:
            assert num_att_heads % 2 == 0, "Number of attention heads should be a multiple of 2."
        self.highway = [HIGHWAYNetworks(embedding_dimension) for _ in range(num_att_heads)]
        self.output_transformation_layer = [nn.Linear(embedding_dimension, out_dimension) for _ in range(num_att_heads)]

    def forward(self, inputs):
        concatenated_outputs = torch.cat([o(h(inputs)) for h, o in zip(self.highway, self.output_transformation_layer)], dim=-1)
        return F.leaky_relu(concatenated_outputs)
