import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer

import torch
from torch.nn import functional as F

import mouette as M

from .utils import *


class MaxValueTrainer(Trainer):
    
    def __init__(self, 
        input_geometry,
        ref_model,
        config : TrainingConfig, 
        **kwargs
    ):
        super().__init__(config)
        self.geom = input_geometry
        self.ref_model = ref_model
        self.ref_model.trainable = False

        self.margin = kwargs.get("margin", 1e-2)
        self.lossfun = IL.training.losses.HKRLoss(margin = self.margin, lmbd=kwargs.get("lmbd", 100.))

    def forward_train_batch(self, data, model):
        X_on, = data        
        X_near = X_on + torch.randn_like(X_on)*1e-1
        with torch.no_grad():
            ref = torch.sign(self.ref_model(X_near))
        Y = model(X_near)
        return torch.sum(self.lossfun(ref*Y))