import os, sys
import mouette as M
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.data import PointSampler
from implicitlab.training import TrainingConfig, Trainer, callbacks, hKRTrainer
from implicitlab.training import losses
from implicitlab.queries.visualize import render_sdf_2d

try:
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    geometry = IL.load_geometry(sys.argv[1])
    print(geometry.geom_type)
    DEVICE = IL.utils.get_device()
    print("DEVICE:", DEVICE)
    NDF = IL.nn.load_model(sys.argv[2]).to(DEVICE)
except:
    print("Usage: pyton detail_field.py <geometry_file> <model_file>")
    exit()


####### Dataset Sampling
N_pts = 50_000

class NeuralSDFValues(IL.fields.FieldGenerator):

    def __init__(self, ndf, device, batch_size=10_000):
        self.ndf = ndf
        self.device = device
        self.batch_size = batch_size

    def compute(self, query):
        return IL.utils.forward_in_batches(self.ndf, query, self.device, batch_size=self.batch_size)

# training data
train_sampler = PointSampler(
    geometry, 
    IL.sampling_strategy.UniformBox(geometry),
    NeuralSDFValues(NDF, DEVICE)
)
points, val = train_sampler.sample(N_pts,on_ratio=1.)
train_data = IL.data.make_tensor_dataset((points, val), DEVICE) 

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))
M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_geometry.obj"))


class DetailFieldModel(torch.nn.Module):

    def __init__(self, ndf, geom):
        super().__init__()
        self.base_ndf = ndf
        # self.detail_field = nn.Sequential(IL.nn.SirenNet(geom.dim, 64, 8), nn.Tanh())
        self.detail_field = IL.nn.SirenNet(geom.dim, 64, 8)

    def forward(self, x):
        return self.base_ndf(x) + self.detail_field(x)

def hessian(x, y):
    g = IL.utils.gradient(x, y)
    h0 = IL.utils.gradient(x, g[:,0])[:,None,:]
    h1 = IL.utils.gradient(x, g[:,1])[:,None,:]
    h2 = IL.utils.gradient(x, g[:,2])[:,None,:]
    h = torch.cat((h0,h1,h2), dim=1)
    return h

def hessian(x, y):
    g = IL.utils.gradient(x, y)
    h = torch.cat([IL.utils.gradient(x, g[:,i])[:,None,:] for i in range(g.shape[1])], dim=1)
    return h


# Setup trainer
class DetailFieldTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.acc_loss_weight = 100.

    def forward_train_batch(self, data, model):
        Xon, Yon = data
    
        Y_detail = model.detail_field(Xon)
        loss_acc =  F.mse_loss(Y_detail, Yon)

        Xrdm = 2.*torch.rand_like(Xon).to(Xon.device)-1.
        Xrdm.requires_grad = True
        Yrdm = model.detail_field(Xrdm)

        hess = hessian(Xrdm, Yrdm)
        lap = torch.diagonal(hess, dim1=-2, dim2=-1).sum(dim=-1)
        loss_lap = 1e-2*torch.mean(lap**2)

        loss_zero = 1e-2*torch.mean(Yrdm**2)
        loss = self.acc_loss_weight * loss_acc + loss_lap + loss_zero
        return loss

model = DetailFieldModel(NDF, geometry).to(DEVICE)

IL.visualize.render_sdf_2d(None, os.path.join(OUTPUT_DIR, "contour_0.png"), os.path.join(OUTPUT_DIR, "grad_0.png"), NDF,  M.geometry.AABB([-1.5,-1.5],[1.5,1.5]), device=DEVICE, batch_size=5000)
# for (_,iso),mesh in IL.visualize.reconstruct_surface_marching_cubes(NDF, M.geometry.AABB.unit_cube(3,True).pad(0.6), DEVICE, res=300).items():
#         M.mesh.save(mesh, f"output/e00_iso{iso}.obj")

trainer = DetailFieldTrainer(TrainingConfig(
    BATCH_SIZE=100,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE,
    OPTIMIZER="adam"
))
trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "log.txt")),
    callbacks.MarchingCubeCB(OUTPUT_DIR, 5, res=200) if geometry.dim==3 else callbacks.Render2DCB(OUTPUT_DIR, 5)
)
trainer.set_training_data(train_data)
trainer.train(model)