from turtle import forward
from torch import nn
import torch

#将图片作为模型权重进行训练
class synthesizedImage(nn.Module):
    def __init__(self, img_shape, device, img=None):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(*img_shape))
        if img is not None:
            self.weight.data.copy_(img.data)
        self.to(device)
    
    def forward(self):
        return self.weight
        