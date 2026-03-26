import os, sys
import mouette as M
import numpy as np
from scipy import sparse as sp
from scipy.spatial import KDTree
import torch
from tqdm import tqdm

class ImplicitRepresentation:

    def __init__(self, neural_model, detail_model):
        self.neural_model = neural_model
        assert self.neural_model is not None
        self.detail_model = detail_model

    def __call__(self, x):
        with torch.no_grad():
            if self.detail_model is not None:
                neur = self.neural_model(x)
                det = self.detail_model(x.cpu().numpy())
                return neur + torch.Tensor(det).to(neur.device)
            return self.neural_model(x)