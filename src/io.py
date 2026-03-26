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


def load_model(folder_path:str, device: str, ignore_grad_correct: bool = False, ignore_detail_field: bool = False):
    """
    Reads a metadata file inside the given folder, and loads
    """
    print("FOLDER", folder_path)
    meta : MetaData = MetaData.load_from_file(os.path.join(folder_path, "metadata.toml"))

    neural_model = initialize_model(meta).to(device)
    if meta.gradient_corrected and not ignore_grad_correct:
        neural_model.load_state_dict(torch.load(os.path.join(folder_path, "weights_gradient_corrected.pt"), map_location=device))
    else:
        neural_model.load_state_dict(torch.load(os.path.join(folder_path, "weights_final.pt"), map_location=device))

    if not meta.detail_field_computed or ignore_detail_field:
        return neural_model
    
    detail_field = None
    if meta.detail_field_computed:
        if meta.adaptative_support:
            raise NotImplementedError
        else:
            detail_field = CompactSupportRBFInterpolant.load_from_file(os.path.join(folder_path, "rbf.pt"))
    return ImplicitRepresentation(neural_model, detail_field)

