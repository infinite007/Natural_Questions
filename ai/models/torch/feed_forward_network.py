import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers, activation_fn):
        super().__init__()
        self.layers = [nn.Linear(input_size, output_size) for _ in range(num_layers)]
        self.activation_fn = activation_fn

    def forward(self, inputs):
        outputs = self.layers[0](inputs)
        for layer in self.layers[1:]:
            outputs = layer(outputs)
            outputs = inputs + self.activation_fn(outputs)
        return outputs
