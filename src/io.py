import os
import implicitlab as IL
import torch
from .metadata import MetaData
from .detail_model import ImplicitRepresentation
from .rbf import CompactSupportRBFInterpolant

def initialize_model(md : MetaData):
    if md.architecture_type == "bjorck" :
        model = IL.nn.DenseLipBjorck(md.geometry_dim, md.layer_size, md.n_layers)
    elif md.architecture_type == "sdp":
        model = IL.nn.DenseLipSDP(md.geometry_dim, md.layer_size, md.n_layers)
    return model


def load_model(folder_path:str, device: str):
    """
    Reads a metadata file inside the given folder, and loads
    """
    meta : MetaData = MetaData.load(os.path.join(folder_path, "metadata.toml"))

    neural_model = initialize_model(meta).to(device)
    if meta.gradient_corrected:
        neural_model.load_state_dict(torch.load(os.path.join(folder_path, "weights_gradient_corrected.pt")))
    else:
        neural_model.load_state_dict(torch.load(os.path.join(folder_path, "weights_final.pt")))

    detail_field = None
    if meta.detail_field_computed:
        if meta.adaptative_support:
            raise NotImplementedError
        else:
            detail_field = CompactSupportRBFInterpolant.load_from_file("rbf.pt")
    return ImplicitRepresentation(neural_model, detail_field)

