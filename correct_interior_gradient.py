import os
import argparse
import mouette as M
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.training import TrainingConfig, callbacks

from src import GradientCorrectionTrainer, MetaData, io

if __name__ == "__main__":

    ###### Parse commandline arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("folder", type=str, help="path to the folder. Should contain a metadata file and neural weights")
    argument_parser.add_argument("-np", "--n-points", type=int, default=60_000)
    argument_parser.add_argument("-ne", "--n-epochs", type=int, default=100)
    argument_parser.add_argument("--optimizer", type=str, choices=["adam", "muon", "sgd"], default="adam")
    argument_parser.add_argument("--learning-rate", type=float, default=1e-3)
    argument_parser.add_argument("--batch-size", type=int, default=1000)
    argument_parser.add_argument("--test-batch-size", type=int, default=5000)
    argument_parser.add_argument("-a", "--spacing", type=float, default=1e-2)
    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()
    print("Training will run on the following device:", DEVICE)
   
    ###### Load data from files
    metadata = MetaData.load_from_file(os.path.join(args.folder, "metadata.toml"))
    if metadata.training_type == "hkr":
        print("Gradient correction is not needed for models trained with hkr loss.")
        answer = input("Proceed anyway? [Y/N]")
        if not answer.lower()=="y": exit()

    geometry = IL.data.load_geometry(os.path.join(args.folder, "input_geometry.obj"))
    model = io.initialize_model(metadata).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(args.folder, "weights_final.pt"), map_location=DEVICE))

    ###### Initialize training
    trainer = GradientCorrectionTrainer(
        geometry,
        model, 
        args.n_points,
        args.spacing,
        TrainingConfig(
            BATCH_SIZE = args.batch_size,
            TEST_BATCH_SIZE = 5000,
            N_EPOCHS = args.n_epochs,
            LEARNING_RATE = args.learning_rate,
            OPTIMIZER = args.optimizer,
            DEVICE = DEVICE,
        ))

    if metadata.geometry_dim == 3:
        trainer.add_callbacks( callbacks.MarchingCubeCB(args.folder, args.n_epochs, res=300, iso=[0.], prefix="grad_correct") )
    else:
        trainer.add_callbacks( callbacks.Render2DCB(args.folder, args.n_epochs//2, prefix="grad_correct"))
    trainer.add_callbacks(callbacks.LoggerCB(os.path.join(args.folder, "grad_correct_training_log.txt")))
    M.mesh.save(trainer.point_cloud, os.path.join(args.folder, "points_gradient_correction.geogram_ascii"))

    ###### Run training
    trainer.train(model)

    torch.save(model.state_dict(),os.path.join(args.folder, "weights_gradient_corrected.pt"))
    metadata.gradient_corrected = True
    metadata.save_to_file(os.path.join(args.folder, "metadata.toml"))
