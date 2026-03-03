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
Take the outside points as a small offset along the outward normal of the inside points.
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
assert geometry.dim == 2

n_points = 1000
points_on, normals = IL.data.sample_utils.sample_points_and_normals2D(geometry, n_points)

points_out = points_on + 0.05*normals

points_on = np.concatenate((points_on, IL.data.sample_utils.sample_points_and_normals2D(geometry, n_points//2)[0]))
points_out = np.concatenate((points_out, sample_circle(M.Vec.zeros(2), 1.8, n_points//2)))
                            
train_data = IL.data.make_tensor_dataset((points_on, points_out), DEVICE)
points = np.concatenate((points_on, points_out))
pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("label", np.concatenate((-np.ones(n_points+n_points//2), np.ones(n_points+n_points//2))))
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))

###### Training 

# setup model
model = IL.nn.DenseLipSDP(geometry.dim, 128, 8).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=200,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=5000,
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
        # loss_eq = torch.sqrt(torch.var(Yin))
        return loss_hkr_in + loss_hkr_out #+ 10*loss_eq
    
    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE) 
    

trainer = CustomTrainer(config, 0.01, 100.)
trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
    callbacks.Render2DCB(OUTPUT_DIR, 500, output_gradient_norm=True)
)
trainer.set_training_data(train_data)
trainer.train(model)

pc_on = M.mesh.from_arrays(points_on)
val_on, pts_grads = IL.utils.forward_in_batches(model, points_on, DEVICE, compute_grad=True, batch_size=10_000)
final_normals = pts_grads/np.linalg.norm(pts_grads, axis=1)[:,None]
print(np.amin(val_on), np.amax(val_on))
pc_on.vertices.register_array_as_attribute("val",val_on)
M.mesh.save(pc_on, os.path.join(OUTPUT_DIR, "final.geogram_ascii"))

M.mesh.save(M.procedural.vector_field(points_on, final_normals, 0.1), os.path.join(OUTPUT_DIR,"normals_final.mesh"))
M.mesh.save(M.procedural.vector_field(points_on, normals, 0.1), os.path.join(OUTPUT_DIR,"normals_init.mesh"))