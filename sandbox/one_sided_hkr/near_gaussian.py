import os, sys
import mouette as M
import numpy as np
import tqdm

import torch
from torch import nn
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer
from implicitlab.training import callbacks, losses

"""
Sample the outside distribution as a gaussian near the surface distribution and optimize for an unsigned distance field
"""

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
file_name = M.utils.get_filename(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()


####### Dataset Sampling
n_points = 50_000 if geometry.geom_type is not IL.data.GeometryType.POINT_CLOUD_3D else len(geometry.vertices)

sampler = IL.data.PointSampler(geometry,
    IL.data.sampling_strategy.UniformBox(geometry),
)
points_on = sampler.sample(n_points, on_ratio=1.)
pc = M.mesh.from_arrays(points_on)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts_0.geogram_ascii"))
train_data = IL.data.make_tensor_dataset((points_on,), DEVICE)


###### Training 

model = IL.nn.DenseLipSDP(geometry.dim, 128, 12).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

class CustomTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.attach_loss_weight = 100.
        self.gaussian_spread = 0.02
        self.grad_loss_weight = 0.01

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        Xon, = data
        Xnear = Xon +  torch.randn_like(Xon)*self.gaussian_spread
        Xfar = Xon + torch.randn_like(Xon)
        # Xnear.requires_grad = True
        # Xfar.requires_grad = True
        Yon = model(Xon)
        
        Ynear = model(Xnear)
        Yfar = model(Xfar)

        loss_attach = torch.sum(Yon**2)
        loss_hkr_out = torch.sum(-torch.abs(Ynear))/2 + torch.sum(-torch.abs(Yfar))/2
        # gd = IL.utils.gradient(Xfar, Yfar)
        # loss_gdnorm = -gd.norm(dim=1).sum()

        return loss_hkr_out + self.attach_loss_weight*loss_attach  #+ self.grad_loss_weight*loss_gdnorm
        # return loss_hkr_out + loss_hkr_in + self.grad_loss_weight*loss_gdnorm 
    
    def initialize_sphere(self, model, dim):
        self.optimizer = self.get_optimizer(model)
        for _ in tqdm.trange(1_000):
            self.optimizer.zero_grad() # zero the parameter gradients
            X_rdm = 3*torch.rand((self.config.BATCH_SIZE, dim)) - 1.5
            X_rdm = X_rdm.to(self.config.DEVICE)
            Y_rdm = model(X_rdm)
            loss = F.mse_loss(Y_rdm, torch.linalg.norm(X_rdm, dim=1)[:,None]-1)
            loss.backward()
            self.optimizer.step()

    
trainer = CustomTrainer(TrainingConfig(
    BATCH_SIZE=100,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=200,
    LEARNING_RATE=1e-2,
    DEVICE=DEVICE,
    OPTIMIZER="muon"
))

if geometry.dim == 3:
    trainer.add_callbacks( callbacks.MarchingCubeCB(OUTPUT_DIR, 10, res=200, iso=[-0.01, 0., 0.01]) )
else:
    trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, 10))

trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
)

trainer.set_training_data(train_data)
trainer.initialize_sphere(model, geometry.dim)
trainer.train(model)