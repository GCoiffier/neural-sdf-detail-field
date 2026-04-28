import os
import implicitlab as IL
import torch
from .metadata import MetaData
from .rbf import CompactSupportRBFInterpolant
from .idf import SirenDisplacementField

class NeuralImplicitWithDetails:
    def __init__(self, neural_model, detail_model, implicit=False):
        self.neural_model = neural_model
        assert self.neural_model is not None
        self.detail_model = detail_model
        self.implicit = implicit

    def __call__(self, x):
        if self.detail_model is None:
            return self.neural_model(x)
        
        if not self.implicit:
            neural_v = self.neural_model(x)
            detail_v = self.detail_model(x.cpu()).to(neural_v.device).reshape((neural_v.shape[0],1))
            return neural_v + detail_v
        
        else:
            x.requires_grad = True
            neural_v = self.neural_model(x)
            grad_v = IL.utils.gradient(x, neural_v)
            detail_v = self.detail_model(x.detach().cpu()).to(neural_v.device).reshape((neural_v.shape[0],1))
            pts_displaced = x + detail_v*torch.nn.functional.normalize(grad_v)
            return self.neural_model(pts_displaced)



def load_model(metadata: MetaData, folder_path:str, device: str, ignore_detail_field:bool = False):
    if metadata.architecture_type == "SLL":
        neural_model = IL.nn.DenseLipSDP(metadata.geometry_dim, metadata.layer_size, metadata.n_layers).to(device)

    elif metadata.architecture_type == "SIREN":
        neural_model = IL.nn.SirenNet(metadata.geometry_dim, metadata.layer_size, metadata.n_layers).to(device)

    elif metadata.architecture_type == "MLP":
        neural_model = IL.nn.MultiLayerPerceptron(metadata.geometry_dim, metadata.layer_size, metadata.n_layers).to(device)

    elif metadata.architecture_type == "RFF": # Random Fourier Features
        neural_model = torch.nn.Sequential(
            IL.nn.encodings.RandomFourierEncoding(metadata.geometry_dim, 1000),
            IL.nn.MultiLayerPerceptron(1000, metadata.layer_size, metadata.n_layers)
        ).to(device)

    elif metadata.architecture_type == "IDF": # Implicit Displacement Field
        neural_model = SirenDisplacementField(metadata.layer_size, metadata.n_layers).to(device)
        neural_model.kappa = 0.
    
    neural_model.load_state_dict(torch.load(os.path.join(folder_path, "weights_final.pt"), map_location=device))


    if not metadata.has_detail_field or ignore_detail_field:
        return neural_model
    else:
        detail_field = CompactSupportRBFInterpolant.load_from_file(os.path.join(folder_path, "rbf.pt"))
        return NeuralImplicitWithDetails(neural_model, detail_field, metadata.implicit)

