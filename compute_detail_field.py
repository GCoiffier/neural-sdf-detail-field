import os
import torch
import argparse
import implicitlab as IL

from src import CompactSupportRBFInterpolant, MetaData, initialize_model

if __name__ == "__main__":

    ###### Parse commandline arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("folder", type=str, help="path to the folder. Should contain a metadata file and neural weights")
    argument_parser.add_argument("-np", "--n-points", type=int, default=3_000)
    argument_parser.add_argument("--no-gradient-correction", action="store_true")
    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()

    metadata = MetaData.load_from_file(os.path.join(args.folder, "metadata.toml"))
    metadata.n_primitives = args.n_points
    metadata.adaptative_support = False

    neural_model = initialize_model(metadata).to(DEVICE)
    if not metadata.gradient_corrected or args.no_gradient_correction:
        neural_model.load_state_dict(torch.load(os.path.join(args.folder, "weights_final.pt")))
    else:
        neural_model.load_state_dict(torch.load(os.path.join(args.folder, "weights_gradient_corrected.pt")))

    metadata.detail_field_computed = True
    metadata.save_to_file(os.path.join(args.folder, "metadata.toml"))

