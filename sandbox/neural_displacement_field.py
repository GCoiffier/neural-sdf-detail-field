import os, sys
import mouette as M
import numpy as np
import torch
from torch import nn

import implicitlab as IL
from implicitlab.data import PointSampler
from implicitlab.training import TrainingConfig, Trainer, callbacks, hKRTrainer
from implicitlab.training import losses
from implicitlab.queries.visualize import render_sdf_2d

OUTPUT_DIR = "_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()
print("DEVICE:", DEVICE)


####### Dataset Sampling

N_pts = 30_000

# training data
train_sampler = PointSampler(
    geometry, 
    IL.sampling_strategy.UniformBox(geometry),
    IL.fields.Occupancy(geometry, v_in=-1, v_out=1, v_on=-1)
)
points, val = train_sampler.sample(N_pts)


train_field2 = PointSampler(
    geometry, 
    IL.sampling_strategy.NearGeometryGaussian(geometry),
    IL.fields.Distance(geometry)
)
points_close, val_close = train_sampler.sample(N_pts)
train_data = IL.data.make_tensor_dataset((points, val, points_close, val_close), DEVICE) 

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))

# testing data
test_field = IL.fields.Distance(geometry, signed=True)
test_sampling_strat = IL.sampling_strategy.UniformBox(geometry)
test_sampler = PointSampler(geometry, test_sampling_strat, test_field)
test_pts, test_val = test_sampler.sample(10_000)
test_data = IL.data.make_tensor_dataset((test_pts, test_val), DEVICE)


###### Training 

class LipDisplacement(nn.Module):

    def __init__(self, geometry):
        super().__init__()
        self.lip_model = IL.nn.DenseLipSDP(geometry.dim,128,20)
        # self.detail_model = nn.Sequential(IL.nn.SirenNet(geometry.dim,64,6), nn.Tanh())
        self.detail_model = IL.nn.SirenNet(geometry.dim,64,6)

    def forward(self, x):
        d_lip = self.lip_model(x)
        d_detail = self.detail_model(x)
        return d_lip + d_detail/(1 + d_lip**2)

model = LipDisplacement(geometry).to(DEVICE)

# Setup trainer
class CustomTrainer(Trainer):

    def __init__(self, config, margin, lmbd=0.1):
        super().__init__(config)
        self.lmbd = lmbd
        self.margin = margin
        self.testlossfun = torch.nn.MSELoss()

    def get_optimizer(self, model):
        # return torch.optim.SGD(model.parameters(), lr=self.config.LEARNING_RATE, momentum=0.9)
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        X,occ, Xnear, dist = data
        Y_lip = model.lip_model(X)
        loss_hkr = nn.functional.relu(self.margin - occ*Y_lip) - (1./self.lmbd) * occ*Y_lip
        loss = loss_hkr.sum()

        Y_detail = model.detail_model(X)
        loss += 1e-4*torch.sum(Y_detail**2)

        Y_near = model(Xnear)
        loss += 1e-4*torch.nn.functional.mse_loss(Y_near, dist)
        return loss

class CustomRender2DCB(IL.training.Callback):

    def __init__(self, save_folder: str, freq: int, plot_domain: M.geometry.AABB = None, resolution: int = 800):
        super().__init__()
        self.save_folder = save_folder
        self.freq = freq
        if plot_domain is None:
            self.domain = M.geometry.AABB([-1.5,-1.5],[1.5,1.5])
        else:
            self.domain = plot_domain
        self.freq = freq
        self.res = resolution

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            contour_path = os.path.join(self.save_folder, f"contour_lip_{epoch}.png")
            gradient_path = os.path.join(self.save_folder, f"grad_lip_{epoch}.png")
            render_sdf_2d(
                None,
                contour_path,
                gradient_path,
                model.lip_model, 
                self.domain, 
                trainer.config.DEVICE, 
                res=self.res, 
                batch_size=trainer.config.TEST_BATCH_SIZE,
            )

            contour_path = os.path.join(self.save_folder, f"contour_full_{epoch}.png")
            gradient_path = os.path.join(self.save_folder, f"grad_full__{epoch}.png")
            render_sdf_2d(
                None,
                contour_path,
                gradient_path,
                model, 
                self.domain, 
                trainer.config.DEVICE, 
                res=self.res, 
                batch_size=trainer.config.TEST_BATCH_SIZE,
            )

class UpdateHkrRegulCB(callbacks.Callback):

    def __init__(self, when : dict):
        super().__init__()
        self.when = when

    def callOnBeginTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch in self.when:
            trainer.lmbd = self.when[epoch]
            print("Updated loss regul weight to", self.when[epoch])



# config1 = TrainingConfig(
#     BATCH_SIZE=100,
#     N_EPOCHS=100,
#     LEARNING_RATE=5e-4,
#     DEVICE=DEVICE
# )
# trainer1 = hKRTrainer(config1, 0.01, 100.)
# trainer1.add_callbacks(
#     callbacks.LoggerCB("output/training_log.txt"),
#     callbacks.Render2DCB("output/lip", 10),
#     UpdateHkrRegulCB({1 : 1., 5 : 10., 10: 100.})
# )
# trainer1.set_training_data(train_data)
# trainer1.train(model.lip_model)

config2 = TrainingConfig(
    BATCH_SIZE=200,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE
)
trainer2 = CustomTrainer(config2, 0.01, 100.)
trainer2.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "log.txt")),
    CustomRender2DCB(OUTPUT_DIR, 10),
    UpdateHkrRegulCB({1: 1., 5: 10., 10: 100.})
)

trainer2.set_training_data(train_data)
trainer2.set_test_data(test_data)
trainer2.train(model)
# IL.nn.save_model(model, "output/model.pt")