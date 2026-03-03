import os, sys
import mouette as M
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer
from implicitlab.training import callbacks, losses

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
n_points = 1000
sampler = IL.data.PointSampler(geometry,
    IL.data.sampling_strategy.UniformBox(geometry),
)
points_on = sampler.sample(n_points, on_ratio=1.)

domain = M.geometry.AABB.of_points(points_on, padding=0.3)
if geometry.dim == 3:
    points_out = M.sampling.sample_sphere(M.Vec.zeros(geometry.dim), 1.4, n_points)
elif geometry.dim == 2:
    points_out = sample_circle(M.Vec.zeros(2), 1.4, n_points)

# points_out = sampler.sample(n_points, on_ratio=0.)
train_data = IL.data.make_tensor_dataset((points_on, points_out), DEVICE)


points = np.concatenate((points_on, points_out))
pc = M.mesh.from_arrays(points)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts_0.geogram_ascii"))
# M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_data.xyz"))


###### Training 

# setup model
model = IL.nn.DenseLipSDP(geometry.dim, 128, 8).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")


def update_training_points(points, model):
    pts_values, pts_grads = IL.utils.forward_in_batches(model, points_on, DEVICE, compute_grad=True, batch_size=10_000)
    pts_grads /= np.linalg.norm(pts_grads, axis=1)[:,None]
    points -= 0.3*pts_values*pts_grads
    return points

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=200,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=1000,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE
)


class CustomTrainer(Trainer):

    def __init__(self, config, margin, lmbd=100., test_mode="sdf"):
        super().__init__(config)
        self.lossfun = losses.HKRLoss(margin, lmbd)
        if test_mode.lower()=="sdf":
            self.testlossfun = torch.nn.MSELoss()
        elif test_mode.lower()=="hkr":
            self.testlossfun = losses.HKRLoss(margin, lmbd)

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        Xin, Xout = data
        Yin, Yout = model(Xin), model(Xout)
        
        loss_hkr_in = self.lossfun(-Yin).sum()
        loss_hkr_out = self.lossfun(Yout).sum()
        loss_eq = torch.sqrt(torch.var(Yin))
        return loss_hkr_in + loss_hkr_out + 10*loss_eq
    
    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE) 
    

trainer = CustomTrainer(config, 0.1, 100.)
trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
)
if geometry.dim == 3:
    trainer.add_callbacks( callbacks.MarchingCubeCB(OUTPUT_DIR, 100, res=100, iso=[-0.3, -0.2, -0.1, 0.]) )
else:
    trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, 100, output_gradient_norm=False))

# for i_iter in range(10):
#     trainer.set_training_data(train_data)
#     trainer.train(model, starting_epoch=10*i_iter)

#     points_out = update_training_points(points_out, model)
#     points = np.concatenate((points_on, points_out))
#     val = np.concatenate((-np.ones(n_points), np.ones(n_points)))
    
#     train_data = IL.data.make_tensor_dataset((points, val), DEVICE)
    
#     pc = M.mesh.from_arrays(points)
#     pc.vertices.register_array_as_attribute("val", val)
#     M.mesh.save(pc, os.path.join(OUTPUT_DIR, f"train_pts_{i_iter+1}.geogram_ascii"))

trainer.set_training_data(train_data)
trainer.train(model)

pc_on = M.mesh.from_arrays(points_on)
val_on, pts_grads = IL.utils.forward_in_batches(model, points_on, DEVICE, compute_grad=True, batch_size=10_000)
normals = pts_grads/np.linalg.norm(pts_grads, axis=1)[:,None]
print(np.amin(val_on), np.amax(val_on))
pc_on.vertices.register_array_as_attribute("val",val_on)
M.mesh.save(pc_on, os.path.join(OUTPUT_DIR, "final.geogram_ascii"))

M.mesh.save(M.procedural.vector_field(points_on, normals, 0.1), os.path.join(OUTPUT_DIR,"normals.mesh"))