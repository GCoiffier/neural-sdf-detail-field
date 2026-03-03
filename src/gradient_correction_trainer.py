import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer

import torch
from torch.nn import functional as F

import mouette as M

from .utils import *


class GradientCorrectionTrainer(Trainer):
    
    def __init__(self, 
        input_geometry,
        model,
        n_points : int,
        spacing : float,
        config : TrainingConfig, 
        **kwargs
    ):
        super().__init__(config)
        self.geom = input_geometry
        self.n_points = n_points
        self.spacing = spacing
        self.margin = kwargs.get("margin", 1e-2)
        self.attach_weight = kwargs.get("attach_weight", 10.)
        self.lossfun = IL.training.losses.HKRLoss(margin = self.margin, lmbd=kwargs.get("lmbd", 100.))

        self.points_on = None
        self.points_in = None
        self.points_out = None
        self.sample_points(model)
        dataset = IL.data.make_tensor_dataset((self.points_on, self.points_in, self.points_out), self.config.DEVICE)
        self.set_training_data(dataset)


    def forward_train_batch(self, data, model):
        X_on, X_in, X_out = data
        Y_on, Y_in, Y_out = model(X_on), model(X_in), model(X_out)
        # X_noise = X_on + torch.randn_like(X_on)/10.
        # X_noise.requires_grad = True
        # Y_noise = model(X_noise)
        # loss_gdmax = torch.sum(-Y_noise)
        return torch.sum(self.lossfun(Y_out)) + torch.sum(self.lossfun(-Y_in)) + self.attach_weight*torch.sum(Y_on**2) #+ 0.1*loss_gdmax
    
    # def forward_train_batch(self, data, model):
    #     X_on, _,_ = data
    #     Y_on = model(X_on)
    #     X_noise = X_on + torch.randn_like(X_on)/10.
    #     X_noise.requires_grad = True
    #     Y_noise = model(X_noise)
    #     loss_gdmax = torch.sum(-Y_noise)
    #     return  0.1*loss_gdmax + self.attach_weight*torch.sum(Y_on**2)

    def sample_points(self, model):
        sampler = IL.PointSampler(self.geom, IL.sampling_strategy.UniformBox(self.geom), NeuralSDFValues(model, self.config.DEVICE))
        points, vals = sampler.sample(5*self.n_points, on_ratio=0)
        vals = np.squeeze(vals)
        # pc = M.mesh.from_arrays(points)
        # pc.vertices.register_array_as_attribute("val", vals)
        # M.mesh.save(pc, "points_test.geogram_ascii")
        # exit()
        self.points_out = points[vals>self.spacing, :]
        self.points_in = points[vals<-self.spacing, :]
        # balance dataset
        min_n = min(self.points_out.shape[0], self.points_in.shape[0])
        print(f"Sampled {min_n} points inside and outside")
        self.points_out, self.points_in = self.points_out[:min_n, :], self.points_in[:min_n, :]
        self.points_on, _ = sampler.sample(min_n, on_ratio=1.)
        # print(self.points_in.shape, self.points_on.shape, self.points_out.shape)

    @property
    def point_cloud(self):
        pc = M.mesh.from_arrays(np.concatenate((self.points_on, self.points_in, self.points_out)))
        classif = np.concatenate([np.zeros(self.points_on.shape[0]), -np.ones(self.points_in.shape[0]), np.ones(self.points_out.shape[0])])
        pc.vertices.register_array_as_attribute("classif", classif)
        return pc