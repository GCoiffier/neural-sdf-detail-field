import os, sys
import mouette as M
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer
from implicitlab.training import callbacks, losses


"""
Resample an outside (+eps level set) and an inside (-eps level set) distribution every n training epochs.
"""

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
file_name = M.utils.get_filename(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

def sample_circle(center, radius, n_points):
    theta = 2*np.pi*np.random.random(size=n_points)
    points = np.asarray(center) + radius*np.array([[np.cos(t), np.sin(t)] for t in theta])
    return points


####### Dataset Sampling
n_points = 5_000 if geometry.geom_type is not IL.data.GeometryType.POINT_CLOUD_3D else len(geometry.vertices)

sampler = IL.data.PointSampler(geometry,
    IL.data.sampling_strategy.UniformBox(geometry),
)
points_on = sampler.sample(n_points, on_ratio=1.)



domain = M.geometry.AABB.of_points(points_on, padding=0.3)
if geometry.dim == 3:
    points_out = M.sampling.sample_sphere(M.Vec.zeros(geometry.dim), 1.5, n_points)
    points_in = M.sampling.sample_sphere(M.Vec.zeros(geometry.dim), 1e-2, n_points)
elif geometry.dim == 2:
    points_out = sample_circle(M.Vec.zeros(2), 1.5, n_points)
    points_in = sample_circle(M.Vec(0., 0.1), 1e-2, n_points)
    # points_in = sample_circle(M.Vec.zeros(2), 1e-2, n_points)

train_data = IL.data.make_tensor_dataset((points_on, points_out, points_in), DEVICE)


points = np.concatenate((points_on, points_out, points_in))
pc = M.mesh.from_arrays(points)
val = np.concatenate((np.zeros(n_points), np.ones(n_points), -np.ones(n_points)))
    
pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts_0.geogram_ascii"))


###### Training 

model = IL.nn.DenseLipSDP(geometry.dim, 256, 15).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

class CustomTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.attach_loss_weight = 200.
        self.grad_loss_weight = 1e-3
        self.grad_loss_spread = 100.
        self.out_spread = 1000.

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        Xon, Xout, Xin = data
        
        Xout = Xout + torch.randn_like(Xout)/self.out_spread
        Xin = Xin + torch.randn_like(Xin)/self.out_spread

        Yon = model(Xon)
        Yout = model(Xout)
        Yin = model(Xin)
        
        loss_attach = torch.sum(Yon**2)
        loss_hkr = torch.sum(Yin) - torch.sum(Yout)

        # Xrdm = 3.*torch.rand_like(Xin).to(Xin.device)-1.5
        # Xrdm = Xin + torch.randn_like(Xin)/self.grad_loss_spread
        # Xrdm.requires_grad = True
        # Yrdm = model(Xrdm)
        # gd = IL.utils.gradient(Xrdm, Yrdm)
        # loss_gdnorm = -gd.norm(dim=1).sum()

        return loss_hkr + self.attach_loss_weight*loss_attach  #+ self.grad_loss_weight*loss_gdnorm  
        # return loss_hkr_out + loss_hkr_in + self.grad_loss_weight*loss_gdnorm  # + 10*loss_eq
    

class DistributionUpdateCallback(IL.training.Callback):
    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.margin = 0.01

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:

            # resample points
            points_on = sampler.sample(n_points, on_ratio=1.)


            n_points_out = 0
            points_out = []
            while n_points_out<n_points:
                batch = IL.queries.sample_iso_raytraced(model, n_points, device=DEVICE, iso=self.margin, threshold=1e-4)
                points_out.append(batch)
                n_points_out += batch.shape[0]
            points_out = np.concatenate(points_out)[:n_points, :]


            n_points_in = 0
            points_in = []
            while n_points_in<n_points:
                batch = IL.queries.sample_iso_raytraced(model, n_points, device=DEVICE, iso=-self.margin, threshold=1e-4)
                points_in.append(batch)
                n_points_in += batch.shape[0]
            points_in = np.concatenate(points_in)[:n_points, :]

            # set new training data
            train_data = IL.data.make_tensor_dataset((points_on, points_out, points_in), DEVICE)
            trainer.set_training_data(train_data)
            trainer.optimizer = trainer.get_optimizer(model)
            
            # output visualization
            points = np.concatenate((points_on, points_out, points_in))
            pc = M.mesh.from_arrays(points)
            pc.vertices.register_array_as_attribute("val", val)
            M.mesh.save(pc, os.path.join(OUTPUT_DIR, f"train_pts_{epoch}.geogram_ascii"))


trainer = CustomTrainer(TrainingConfig(
        BATCH_SIZE=100,
        TEST_BATCH_SIZE = 5000,
        N_EPOCHS=200,
        LEARNING_RATE=1e-2,
        DEVICE=DEVICE,
        OPTIMIZER="muon"
    ))


UPDATE_FREQ = 10

if geometry.dim == 3:
    trainer.add_callbacks( callbacks.MarchingCubeCB(OUTPUT_DIR, UPDATE_FREQ, res=200, iso=[-0.01, 0., 0.01]) )
else:
    trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, UPDATE_FREQ))

trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
    DistributionUpdateCallback(UPDATE_FREQ)
)

trainer.set_training_data(train_data)
trainer.train(model)