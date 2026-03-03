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
Resample the outside distribution after a number of epochs. The new distribution consist of uniformly sampled points from the m-level set (where m is the margin).

Special case of point cloud following 3D curves for experiments.
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
elif geometry.dim == 2:
    points_out = sample_circle(M.Vec.zeros(2), 1.4, n_points)
train_data = IL.data.make_tensor_dataset((points_on, points_out), DEVICE)

points = np.concatenate((points_on, points_out))
pc = M.mesh.from_arrays(points)
val = np.concatenate((-np.ones(n_points), np.ones(n_points)))

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts_0.geogram_ascii"))


###### Training

model = IL.nn.DenseLipSDP(geometry.dim, 128, 8).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

class CustomTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.attach_loss_weight = 100.
        self.grad_loss_weight = 1e-2
        self.grad_loss_spread = 10.

        self.hkrloss = losses.HKRLoss()

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        Xin,Xout = data
        Yin = model(Xin)
        Yout = model(Xout)

        # loss_attach = torch.sum(Yin**2)
        loss_attach = F.relu(Yin).sum()
        # loss_hkr_in = self.hkrloss(-Yin).sum()
        loss_hkr_out = torch.sum(-Yout)
        # loss_hkr_out = self.hkrloss(Yout).sum()
        # loss_hkr_in = self.hkrloss(-Yin).sum()
        # loss_eq = torch.sqrt(torch.var(Yin)).sum()

        # Xrdm = 3.*torch.rand_like(Xin).to(Xin.device)-1.5
        # # Xrdm = Xin + (torch.rand_like(Xin) - 0.5)/self.grad_loss_spread
        # Xrdm.requires_grad = True
        # Yrdm = model(Xrdm)
        # gd = IL.utils.gradient(Xrdm, Yrdm)
        # loss_gdnorm = -gd.norm(dim=1).sum()

        #return loss_hkr_out + self.attach_loss_weight*loss_attach  # + self.grad_loss_weight*loss_gdnorm  # + 10*loss_eq
        # return loss_hkr_out + loss_hkr_in  + self.grad_loss_weight*loss_gdnorm  # + 10*loss_eq
        return loss_hkr_out + 10.*loss_attach #+ loss_eq
        # return loss_hkr_out + loss_hkr_in + self.grad_loss_weight*loss_gdnorm  # + 10*loss_eq

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)


class DistributionUpdateCallback(IL.training.Callback):
    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.margin = 0.02

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:

            # resample points
            points_on = sampler.sample(n_points, on_ratio=1.)
            n_points_out = 0
            points_out = []
            while n_points_out<n_points:
                batch = IL.queries.sample_iso_raytraced(model, n_points, device=DEVICE, iso=self.margin, threshold=1e-2)
                points_out.append(batch)
                n_points_out += batch.shape[0]
                print(n_points_out)
            points_out = np.concatenate(points_out)[:n_points, :]

            # set new training data
            train_data = IL.data.make_tensor_dataset((points_on, points_out), DEVICE)
            trainer.set_training_data(train_data)
            trainer.optimizer = trainer.get_optimizer(model)

            # output visualization
            points = np.concatenate((points_on, points_out))
            pc = M.mesh.from_arrays(points)
            pc.vertices.register_array_as_attribute("val", val)
            M.mesh.save(pc, os.path.join(OUTPUT_DIR, f"train_pts_{epoch}.geogram_ascii"))


## Step 1
trainer = CustomTrainer(TrainingConfig(
        BATCH_SIZE=100,
        TEST_BATCH_SIZE = 5000,
        N_EPOCHS=100,
        LEARNING_RATE=1e-2,
        DEVICE=DEVICE
    ))

if geometry.dim == 3:
    trainer.add_callbacks( callbacks.MarchingCubeCB(OUTPUT_DIR, 100, res=100, iso=[0., 0.01], prefix="step1") )
else:
    trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, 100))

trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
    DistributionUpdateCallback(10)
)

trainer.set_training_data(train_data)
trainer.train(model)


# Step 2
trainer = IL.training.hKRTrainer(TrainingConfig(
    BATCH_SIZE=100,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE,
    OPTIMIZER="adam"),
    margin=0.01,
    lmbd=10.
)
if geometry.dim == 3:
    trainer.add_callbacks( callbacks.MarchingCubeCB(OUTPUT_DIR, 20, res=200, iso=[0., 0.01], prefix="step2") )
else:
    trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, 20))

trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
)
trainer.set_training_data(train_data)
trainer.train(model)