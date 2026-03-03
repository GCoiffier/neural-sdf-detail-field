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

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
file_name = M.utils.get_filename(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling
n_points = 2000 if geometry.dim == 2 else 10_000
sampler = IL.data.PointSampler(geometry,
    IL.data.sampling_strategy.UniformBox(geometry),
)
points_on = sampler.sample(n_points, on_ratio=1.)

train_data = IL.data.make_tensor_dataset((points_on,), DEVICE)

M.mesh.save(M.mesh.from_arrays(points_on), os.path.join(OUTPUT_DIR, "sampled_points.xyz"))

###### Training 

# setup model
# model = IL.nn.DenseLipSDP(geometry.dim, 256, 20, activation=nn.Softplus(10)).to(DEVICE)
model = IL.nn.DenseLipSDP(geometry.dim, 256, 20, activation=nn.ReLU()).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
class CustomTrainer2D(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.grad_loss_weight = 1e-1
        self.grad_loss_spread = 10.

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        Xin, = data
        R_out = 1.2+torch.rand((Xin.shape[0],1)).to(Xin.device)/10
        T_out = torch.rand(Xin.shape[0]).to(Xin.device)*2*torch.pi
        Xout = R_out*torch.vstack((torch.cos(T_out), torch.sin(T_out))).transpose(0,1)
        
        Yin = model(Xin)
        Yout = model(Xout)
        
        # loss_hkr_in = self.lossfun(-Yin).sum()
        loss_attach = torch.sum(Yin**2)
        loss_hkr_out = torch.sum(-Yout) # self.lossfun(Yout).sum()

        # loss_eq = torch.sqrt(torch.var(Yin))

        Xrdm = 2*torch.rand_like(Xin).to(Xin.device)-1
        Xrdm = Xin + (torch.rand_like(Xin) - 0.5)/self.grad_loss_spread
        Xrdm.requires_grad = True
        Yrdm = model(Xrdm)
        gd = IL.utils.gradient(Xrdm, Yrdm)

        loss_gdnorm = -torch.linalg.norm(gd, dim=1).sum()

        return loss_hkr_out + 100*loss_attach  + self.grad_loss_weight*loss_gdnorm
    

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
    

    def initialize_sphere(self, model):
        self.optimizer = self.get_optimizer(model)
        for _ in tqdm.trange(1_000):
            self.optimizer.zero_grad() # zero the parameter gradients
            X_rdm = 2*torch.rand((self.config.BATCH_SIZE, 2))-1
            X_rdm = X_rdm.to(self.config.DEVICE)
            Y_rdm = model(X_rdm)
            loss = F.mse_loss(Y_rdm, torch.linalg.norm(X_rdm, dim=1)[:,None]-1)
            loss.backward()
            self.optimizer.step()

class CustomTrainer3D(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.attach_loss_weight = 100.
        self.grad_loss_weight = 1e-2
        self.grad_loss_spread = 100.

        self.hkrloss = losses.HKRLoss()

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        bs = self.config.BATCH_SIZE
        Xin, = data
        Xout = (1.5+0.1*torch.rand((bs,1)))*torch.nn.functional.normalize(torch.rand((bs,3)))
        Xout = Xout.to(Xin.device)
        Yin = model(Xin)
        Yout = model(Xout)
        
        # loss_hkr_in = self.hkrloss(-Yin).sum()
        loss_attach = torch.sum(Yin**2)
        loss_hkr_out = torch.sum(-Yout) 
        # loss_hkr_out = self.hkrloss(Yout).sum()

        # loss_eq = torch.sqrt(torch.var(Yin))

        Xrdm = 2*torch.rand_like(Xin).to(Xin.device)-1
        Xrdm = Xin + (torch.rand_like(Xin) - 0.5)/self.grad_loss_spread
        Xrdm.requires_grad = True
        Yrdm = model(Xrdm)
        gd = IL.utils.gradient(Xrdm, Yrdm)

        loss_gdnorm = -gd.norm(dim=1).sum()

        return loss_hkr_out + self.attach_loss_weight*loss_attach + self.grad_loss_weight*loss_gdnorm  # + 10*loss_eq
    

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
    

    def initialize_sphere(self, model, radius:float = 1.):
        self.optimizer = self.get_optimizer(model)
        for _ in tqdm.trange(1_000):
            self.optimizer.zero_grad() # zero the parameter gradients
            X_rdm = 2*torch.rand((self.config.BATCH_SIZE, 3))-1
            X_rdm = X_rdm.to(self.config.DEVICE)
            Y_rdm = model(X_rdm)
            loss = F.mse_loss(Y_rdm, torch.linalg.norm(X_rdm, dim=1)[:,None]- radius)
            loss.backward()
            self.optimizer.step()


class ParamUpdateCallback(callbacks.Callback):
    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch==200:
            trainer.config.LEARNING_RATE = 1e-3
            trainer.optimizer = trainer.get_optimizer(model)
        if epoch==1000:
            # trainer.grad_loss_weight = 1.
            trainer.grad_loss_spread = 10.


if geometry.dim == 3:
    config3D = TrainingConfig(
        BATCH_SIZE=500,
        TEST_BATCH_SIZE = 5000,
        N_EPOCHS=2000,
        LEARNING_RATE=1e-2,
        DEVICE=DEVICE
    )
    trainer = CustomTrainer3D(config3D)
    trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, 100, res=100, iso=[-0.01, 0., 0.01]))

elif geometry.dim == 2:
    config2D = TrainingConfig(
        BATCH_SIZE=500,
        TEST_BATCH_SIZE = 5000,
        N_EPOCHS=2000,
        LEARNING_RATE=1e-1,
        DEVICE=DEVICE
    )
    trainer = CustomTrainer2D(config2D)
    trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, 200))


trainer.add_callbacks(
    callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
    # ParamUpdateCallback()
)

trainer.set_training_data(train_data)
trainer.initialize_sphere(model)
if geometry.dim == 2 :
    IL.visualize.render_sdf_2d(None, os.path.join(OUTPUT_DIR, "contour_init.png"), os.path.join(OUTPUT_DIR, "gradient_init.png"), model, domain=M.geometry.AABB([-1.5,-1.5],[1.5,1.5]), device=DEVICE, res=800, batch_size=config.TEST_BATCH_SIZE)
elif geometry.dim == 3:
    iso_surfaces = IL.visualize.reconstruct_surface_marching_cubes(model, domain=M.geometry.AABB([-1.2]*3,[1.2]*3), device=DEVICE)
    for (n,off),mesh in iso_surfaces.items():
        M.mesh.save(mesh, os.path.join(OUTPUT_DIR, f"level_set_init.obj"))

trainer.train(model)


pc_on = M.mesh.from_arrays(points_on)
val_on, pts_grads = IL.utils.forward_in_batches(model, points_on, DEVICE, compute_grad=True, batch_size=10_000)
normals = pts_grads/np.linalg.norm(pts_grads, axis=1)[:,None]
print(np.amin(val_on), np.amax(val_on))
pc_on.vertices.register_array_as_attribute("val",val_on)
M.mesh.save(pc_on, os.path.join(OUTPUT_DIR, "final.geogram_ascii"))

M.mesh.save(M.procedural.vector_field(points_on, normals, 0.1), os.path.join(OUTPUT_DIR,"normals.mesh"))