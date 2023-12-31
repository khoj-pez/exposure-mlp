import torch
import torch.nn as nn

from models.pe import PE

class CorrectionMLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=3, width=256, num_layers=2):
        super(CorrectionMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, exposure):
        exposure = exposure.unsqueeze(-1)
        return self.layers(exposure)

class MLPExposure(nn.Module):
    def __init__(self, input_dim, output_dim, width, num_layers):
        super(MLPExposure, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, exposure):
        x = torch.cat((x, exposure), dim=-1)  # concatenate along the last dimension
        out = self.layers(x)
        out = torch.sigmoid(out)
        return out

class FCNetExposure(nn.Module):
    def __init__(self):
        super(FCNetExposure, self).__init__()
        self.pe = PE(num_res=10)
        self.mlp = MLPExposure(43, 3, 256, 9)
        self.correction = CorrectionMLP()

    def forward(self, x, exposure=None):
        out = self.pe(x)
        out = self.mlp(out, exposure)
        correction = self.correction(exposure)
        correction = correction.squeeze(1)
        # print(out.shape)
        # print(correction.shape)
        out = out + correction
        return out