import torch
import numpy as np
from torch.nn import functional as F
import implicitlab as IL
from implicitlab.training import callbacks

def chi(x, nu):
    return 1/(1+torch.pow(x/nu, 4.))

class SirenDisplacementField(torch.nn.Module):

    def __init__(self, layer_size, n_layers):
        super().__init__()
        self.base_siren = IL.nn.SirenNet(3, layer_size, n_layers, w0=15)
        self.detail_siren = IL.nn.SirenNet(3, layer_size, n_layers, w0=60)
        self.kappa = 1.
        self.alpha = 0.05
        self.nu = 0.02

    def forward(self, x):
        x.requires_grad = True
        fx = self.base_siren(x)
        if self.kappa == 1 : return fx
        chix = chi(fx, self.nu)
        nx = self.alpha * F.tanh(self.detail_siren(x))
        gd_bx = IL.utils.gradient(x, fx)
        return self.kappa* fx + (1-self.kappa) * self.base_siren(x + chix*nx*F.normalize(gd_bx))

class KappaUpdateCallback(callbacks.Callback):
    def __init__(self, tm, n_epochs):
        self.tm = tm
        self.n = n_epochs

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        t = epoch/self.n
        if t<self.tm:
            model.kappa = 0.
        else:
            model.kappa = 0.5*(1 + np.cos(np.pi * (t - self.tm)/(1 - self.tm)))
