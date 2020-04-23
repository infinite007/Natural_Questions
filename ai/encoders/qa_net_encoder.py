import torch.nn as nn


class QANetEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.cnn = [nn.Conv1d(self.params["in_channels"],
                              self.params["in_channels"],
                              kernel_size=3,
                              groups=self.params["in_channels"])
                    for _ in range(self.params["n_encoders"])]
        # self.ffn = nn.Linear(self.params["in_channels"], )

    def forward(self, inputs):
        for i in range(self.params["n_encoders"]):
            ln = nn.LayerNorm(inputs.size())
            inputs = ln(inputs)
            inputs = self.cnn[i](inputs)

        # inputs = nn.LayerNorm(inputs)
        return inputs
