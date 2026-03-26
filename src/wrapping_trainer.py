import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer
from implicitlab.training import callbacks, losses

import torch
from torch.nn import functional as F

import mouette as M

from .utils import *

class NeuralWrappingTrainer(Trainer):
    def __init__(self, 
        input_geometry, 
        config : TrainingConfig,
        weight_attach = 110., 
        resample_freq:int = 10, 
        resample_surface:bool = True, 
        **kwargs
    ):
        super().__init__(config)
        
        self.geom = input_geometry
        self.attach_loss_weight: float = weight_attach
        self.attach_value: float = kwargs.get("attach_value", -1e-2)
        self.resample_margin: float = kwargs.get("margin", 1e-2)
        self.Xout_spread: float = kwargs.get("out_noise", 1e-3)
        self.Xnear_spread: float = kwargs.get("near_noise", 1e-3)

        self.resample_surface : bool = resample_surface
        self.add_callbacks(ResampleOutsidePointsCallback(freq=resample_freq, margin=self.resample_margin))

        self.n_points : int = None
        self.points : dict = None


    def initialize_training_data(self, n_points):
        if self.geom.geom_type in (IL.data.GeometryType.POINT_CLOUD_3D, IL.data.GeometryType.POINT_CLOUD_2D):
            self.n_points = len(self.geom.vertices)
        else:
            self.n_points = n_points

        self.sampler = IL.data.PointSampler(self.geom, None) # we need a sampler only to sample the geometry : no other sampling strategy is needed here
        points_on = self.sampler.sample_geometry(self.n_points)
        if self.geom.dim == 3:
            points_out = M.sampling.sample_sphere(M.Vec.zeros(self.geom.dim), 1.5, self.n_points)
        elif self.geom.dim == 2:
            points_out = sample_circle(M.Vec.zeros(2), 1.5, self.n_points)
        train_data = IL.data.make_tensor_dataset((points_on, points_out), self.config.DEVICE)
        self.set_training_data(train_data)

        # setup training point cloud for output
        self.points = {"on" : points_on, "out" : points_out}

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))
    

    def forward_train_batch(self, data, model):
        Xin,Xout = data
        Xout = Xout + torch.randn_like(Xout)*self.Xout_spread

        Yin = model(Xin)
        Yout = model(Xout)
        
        loss_attach = torch.sum((Yin-self.attach_value)**2)
        loss_hkr_out = torch.sum(-Yout)

        # Xnear = Xin + torch.randn_like(Xin)*self.Xnear_spread
        # Xnear.requires_grad = True
        # Ynear = model(Xnear)
        # gd_near = IL.utils.gradient(Xnear, Ynear)
        # loss_gdnorm = -gd_near.norm(dim=1).sum()

        Xcircle = 1.5*F.normalize(torch.randn_like(Xin))
        Ycircle = model(Xcircle)
        loss_max_circle = torch.sum(-Ycircle)

        return loss_hkr_out + 2*self.attach_loss_weight*loss_attach + 0.1*loss_max_circle # 0.1*loss_gdnorm
    



class ResampleOutsidePointsCallback(IL.training.Callback):
    def __init__(self, freq, margin, max_iter = 1000):
        super().__init__()
        self.freq = freq
        self.margin = margin
        self.max_iter = max_iter

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            DEVICE = trainer.config.DEVICE
            n_points = trainer.n_points

            if trainer.resample_surface:
                trainer.points["on"] = trainer.sampler.sample_geometry(n_points)

            n_points_out = 0
            points_out = []
            while n_points_out<n_points:
                batch = IL.queries.sample_iso_raytraced(model, n_points, device=DEVICE, iso=0., threshold=1e-3, max_iter=self.max_iter)
                points_out.append(batch)
                n_points_out += batch.shape[0]
            points_out = np.concatenate(points_out)[:n_points, :]
            trainer.points["out"] = points_out

            train_data = IL.data.make_tensor_dataset((trainer.points["on"] , points_out), DEVICE)
            trainer.set_training_data(train_data)
            trainer.optimizer = trainer.get_optimizer(model) # reset optimizer
