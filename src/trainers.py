import torch
from implicitlab.training.losses import EikonalLoss
from implicitlab.training import TrainingConfig,Trainer

class ImplicitSurfaceTrainer(Trainer):
        def __init__(self, 
            config : TrainingConfig
        ):
            super().__init__(config)
            self.rho = 100.
            self.weights = {
                "eikonal" : 50.,
                "on" : 7000.,
                "out" : 600.,
                "normals": 100.,
            }
        
        def get_optimizer(self, model):
            return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        def forward_test_batch(self, data, model): pass
        
        def forward_train_batch(self, data, model):
            pts, normals = data
            pts.requires_grad = True
            Y_on = model(pts)
            batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

            pts_out = 3*torch.rand_like(pts)-1.5
            pts_out.requires_grad = True
            Y_out = model(pts_out)
            batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(Y_out)))

            grad = torch.autograd.grad(Y_on, pts, grad_outputs=torch.ones_like(Y_on), create_graph=True)[0]
            batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(grad, normals)
            
            batch_loss += self.weights["eikonal"] * EikonalLoss()(pts_out, Y_out)        
            return batch_loss
        

class HotspotTrainer(Trainer):

    def __init__(self, 
        config : TrainingConfig,
        lmbd = 10.
    ):
        super().__init__(config)
        self.lmbd = lmbd
        self.weights = {
            "on" : 50.,
            "eikonal": 0.1,
            "normals": 1.,
            "heat": 1.
        }

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts,normals = data
        pts.requires_grad = True
        Y_on = model(pts)
        batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

        pts_out = 3.*torch.rand_like(pts)-1.5
        pts_out.requires_grad = True
        Y_out = model(pts_out)

        batch_grad_on = torch.autograd.grad(Y_on, pts, grad_outputs=torch.ones_like(Y_out), create_graph=True)[0]
        batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(batch_grad_on, normals)

        batch_grad = torch.autograd.grad(Y_out, pts_out, grad_outputs=torch.ones_like(Y_out), create_graph=True)[0]
        batch_grad_norm = batch_grad.norm(dim=-1)
        batch_loss += self.weights["heat"] * torch.mean(torch.exp(-2*self.lmbd*torch.abs(Y_out))*(1 + batch_grad_norm**2))
        batch_loss += self.weights["eikonal"] * torch.nn.functional.mse_loss(batch_grad_norm, torch.ones_like(batch_grad_norm))
        
        return batch_loss