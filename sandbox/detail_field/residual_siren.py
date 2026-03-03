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
N_pts = 100_000

train_sampler = PointSampler(
    geometry, 
    IL.sampling_strategy.CombinedStrategy([
        IL.sampling_strategy.UniformBox(geometry),
        IL.sampling_strategy.NearGeometryGaussian(geometry)
    ], [2.,1.]),
    IL.fields.Distance(geometry)
)

points, val_GT = train_sampler.sample(N_pts,on_ratio=0.1)
val_NDF = IL.utils.forward_in_batches(NDF, points, DEVICE, batch_size=10_000).squeeze()

train_data = IL.data.make_tensor_dataset((points, val_GT-val_NDF), DEVICE) 

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val_GT-val_NDF)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))
M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_geometry.obj"))


class DetailFieldModel(torch.nn.Module):

    def __init__(self, ndf, geom):
        super().__init__()
        self.base_ndf = ndf
        # self.detail_field = nn.Sequential(IL.nn.SirenNet(geom.dim, 64, 8), nn.Tanh())
        self.detail_field = IL.nn.SirenNet(geom.dim, 64, 6)

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
        X, val = data

        Y_lip = model.base_ndf(X)
        abs_Y_lip = torch.abs(Y_lip).squeeze()

        near_filter = abs_Y_lip<0.05
        far_filter = ~near_filter

        Y_detail_near = model.detail_field(X[near_filter,:])
        Y_detail_far = model.detail_field(X[far_filter,:])

        return F.mse_loss(Y_detail_near, val[near_filter]) + torch.mean(Y_detail_far**2)
    

model = DetailFieldModel(NDF, geometry).to(DEVICE)

IL.visualize.render_sdf_2d(None, os.path.join(OUTPUT_DIR, "contour_0.png"), os.path.join(OUTPUT_DIR, "grad_0.png"), NDF,  M.geometry.AABB([-1.5,-1.5],[1.5,1.5]), device=DEVICE, batch_size=5000)
# for (_,iso),mesh in IL.visualize.reconstruct_surface_marching_cubes(NDF, M.geometry.AABB.unit_cube(3,True).pad(0.6), DEVICE, res=300).items():
#         M.mesh.save(mesh, f"output/e00_iso{iso}.obj")

class DetailFieldRenderer(IL.training.Callback):

    def __init__(self, save_folder: str, freq: int, plot_domain: M.geometry.AABB = None, resolution: int = 800, output_contours: bool = True, output_gradient_norm: bool = True):
        super().__init__()
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        if plot_domain is None:
            self.domain = M.geometry.AABB([-1.5,-1.5],[1.5,1.5])
        else:
            self.domain = plot_domain
        self.freq = freq
        self.res = resolution
        self.output_contours = output_contours
        self.output_gradient_norm = output_gradient_norm

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            render_path = os.path.join(self.save_folder, f"render_details_{epoch}.png")
            contour_path = os.path.join(self.save_folder, f"contour_details_{epoch}.png") if self.output_contours else None
            gradient_path = os.path.join(self.save_folder, f"grad_details_{epoch}.png") if self.output_gradient_norm else None
            render_sdf_2d(
                render_path,
                contour_path,
                gradient_path,
                model.detail_field,
                self.domain, 
                trainer.config.DEVICE, 
                res=self.res, 
                batch_size=trainer.config.TEST_BATCH_SIZE,
            )

trainer = DetailFieldTrainer(TrainingConfig(
    BATCH_SIZE=200,
    N_EPOCHS=100,
    LEARNING_RATE=1e-4,
    DEVICE=DEVICE,
    OPTIMIZER="adam"
))
trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "log.txt")),
    callbacks.MarchingCubeCB(OUTPUT_DIR, 5, res=200) if geometry.dim==3 else callbacks.Render2DCB(OUTPUT_DIR, 5),
    DetailFieldRenderer(OUTPUT_DIR, 5)
)
trainer.set_training_data(train_data)
trainer.train(model)