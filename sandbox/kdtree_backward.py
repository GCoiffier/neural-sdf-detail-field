import torch
from scipy.spatial import KDTree
from torch import nn
import numpy as np
import implicitlab as IL

class ClampedDistance(nn.Module):

    def __init__(self, seeds):
        super().__init__()
        self.seeds = seeds
        self.tree = KDTree(seeds.detach().cpu().numpy())

    def forward(self, x):
        x_num = x.detach().numpy()
        inds = self.tree.query_ball_point(x_num, 0.3)
        rbf_values = torch.zeros(x.shape[0])
        for i,near_i in enumerate(inds):
            rbf_values[i] = torch.sum(torch.norm(x[i,:] - self.seeds[near_i], dim=1),dim=0)
        return rbf_values
    

points = torch.rand((100,3))

D = ClampedDistance(points)

x = torch.rand((1,3))
x.requires_grad = True
y = D(x)

print(y)
grad = IL.utils.gradient(x, y)
print(grad)