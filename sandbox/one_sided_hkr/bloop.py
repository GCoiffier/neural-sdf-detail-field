import os, sys
import mouette as M
import numpy as np
import tqdm
import time
import csv
from collections import deque

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
n_points = 10_000 if geometry.geom_type is not IL.data.GeometryType.POINT_CLOUD_3D else len(geometry.vertices)

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
        self.gaussian_spread = 0.02

        self.rho = 0.01 # weight of the exponential moving average for the gradient
        self.lmbd = 1e-3 # grad_main + lmbd*pi(grad_aux, grad_main)

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

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


    def train(self, model, starting_epoch=0):
        if self.train_data_loader is None:
            raise Exception("No training data was provided. Call the `set_training_data` before training.")
        self.optimizer = self.get_optimizer(model)
        
        grad_EMA = deque()
        grad_attach = []
        # Initialize the EMA accumulated gradient to 0
        for data in self.train_data_loader:
            self.optimizer.zero_grad()
            Xon, = data
            Yon = model(Xon)
            loss_attach = torch.sum(Yon**2)
            loss_attach.backward()
            for ip,p in enumerate(model.parameters()):
                grad_EMA.append(p.grad)
                grad_attach.append(torch.zeros_like(p.grad))
            break

        for epoch in range(self.config.N_EPOCHS):
            epoch += starting_epoch
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            total_attach_loss = 0.
            total_hkr_loss = 0.

            # train
            for data in tqdm.tqdm(self.train_data_loader, total=len(self.train_data_loader)):
                self.optimizer.zero_grad() # zero the parameter gradients

                Xon, = data
                Xnear = Xon +  torch.randn_like(Xon)*self.gaussian_spread
                Xnear.requires_grad = True
                Xfar = Xon + torch.randn_like(Xon)
                Xfar.requires_grad = True
                Yon = model(Xon)
                Ynear = model(Xnear)
                Yfar = model(Xfar)

                # Attach = main, Hkr = aux
                loss_attach = torch.sum(Yon**2)
                loss_attach.backward()
                total_attach_loss += float(loss_attach.detach())

                for ip,p in enumerate(model.parameters()):
                    grad_attach[ip] = p.grad
                    grad_EMA[ip] = (1-self.rho)*grad_EMA[ip] + self.rho*p.grad

                self.optimizer.zero_grad()
                loss_hkr_out = -torch.sum(torch.abs(Ynear)) - 0.01*IL.utils.gradient(Xfar,Yfar).norm(dim=1).sum()
                # loss_hkr_out = -torch.sum(torch.abs(Ynear)) - IL.utils.gradient(Xnear,Ynear).norm(dim=1).sum()
                loss_hkr_out.backward()
                total_hkr_loss += float(loss_hkr_out.detach())

                for ip,p in enumerate(model.parameters()):
                    g_main, g_aux, g_mean = grad_attach[ip], p.grad, grad_EMA[ip]
                    p.grad = g_main + self.lmbd*( g_aux - torch.sum(g_aux * g_mean) / torch.sum(g_mean * g_mean) * g_mean )

                self.optimizer.step()
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["loss_attach"] = total_attach_loss
            self.metrics["loss_hkr"] = total_hkr_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)

    
trainer = CustomTrainer(TrainingConfig(
    BATCH_SIZE=100,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=200,
    LEARNING_RATE=1e-2,
    DEVICE=DEVICE,
    OPTIMIZER="adam"
))

if geometry.dim == 3:
    trainer.add_callbacks( callbacks.MarchingCubeCB(OUTPUT_DIR, 10, res=200, iso=[-0.01, 0., 0.01]) )
else:
    trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, 10))



class CustomLoggerCB(IL.training.Callback):

    def __init__(self, file_path: str):
        super().__init__()
        self.path = file_path
        self.logged = {"epoch" : -1, "time" : 0, "train_loss" : -1, "test_loss" : -1}
        with open(self.path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writeheader()

    def callOnEndTrain(self, trainer, model):
        self.logged.update({"epoch" :  trainer.metrics["epoch"]})
        self.logged.update({"time" :  trainer.metrics["epoch_time"]})
        self.logged.update({"loss_attach" : trainer.metrics["loss_attach"]})
        self.logged.update({"loss_hkr" : trainer.metrics["loss_hkr"]})
        print(f"Train loss after epoch {trainer.metrics['epoch']} : attach = {trainer.metrics['loss_attach']} | hkr = {trainer.metrics["loss_hkr"]}")
        if trainer.test_data_loader is None:
            # no test_data_loader means that callOnEndTest will not be called.
            self._write_log()

    def callOnEndTest(self, trainer, model):
        self.logged.update({"test_loss" : trainer.metrics["test_loss"]})
        print(f"Test loss after epoch {trainer.metrics['epoch']} : {trainer.metrics['test_loss']}\n")
        self._write_log()

    def _write_log(self):
        with open(self.path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writerow(self.logged)
trainer.add_callbacks(
    CustomLoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
)

trainer.set_training_data(train_data)
trainer.initialize_sphere(model, geometry.dim)
trainer.train(model)