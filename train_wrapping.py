import os
import argparse
import mouette as M
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.training import TrainingConfig, callbacks

from src import NeuralWrappingTrainer, SaveTrainingPointsCB, MetaData, io

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_geometry_file", type=str, help="path to the input geometry file. Supported file types are .obj, .mesh, .stl and .geogram_ascii")
    argument_parser.add_argument("-o", "--output-dir", type=str, default="", help="name of the output folder")

    # Training parameters
    argument_parser.add_argument("-np", "--n-points", type=int, default=10_000)
    argument_parser.add_argument("-nr", "--n-resample", type=int, default=20)
    argument_parser.add_argument("-rf", "--resampling-freq", type=int, default=10)
    argument_parser.add_argument("--no-resample-surface", action="store_true")

    # Optimizer parameters
    argument_parser.add_argument("--optimizer", type=str, choices=["adam", "muon", "sgd"], default="adam")
    argument_parser.add_argument("--learning-rate", type=float, default=1e-3)
    argument_parser.add_argument("--batch-size", type=int, default=100)
    argument_parser.add_argument("--test-batch-size", type=int, default=5000)

    # Model parameters
    argument_parser.add_argument("-nl", "--n-layers", type=int, default=10)
    argument_parser.add_argument("-ls", "--layer-size", type=int, default=128)
    argument_parser.add_argument("-a", "--architecture", default="sdp", choices=["sdp", "bjorck"])

    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()
    print("Training will run on the following device:", DEVICE)

    geometry = IL.load_geometry(args.input_geometry_file)
    print("Read geometry of type", geometry.geom_type, "and of dimension", geometry.dim)
    
    if len(args.output_dir)>0:
        OUTPUT_DIR = os.path.join("trained_models", args.output_dir)
    else:
        OUTPUT_DIR = os.path.join("trained_models",  M.utils.get_filename(args.input_geometry_file)+"_wrap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_geometry.obj")) # save input geometry (rescaled) in output folder
    
    ####### Save metadata file
    metadata = MetaData(
        geometry_dim = geometry.dim,
        training_type = "wrapping",
        architecture_type = args.architecture,
        n_layers = args.n_layers,
        layer_size = args.layer_size,
        n_epochs = args.resampling_freq * args.n_resample,
        optimizer = args.optimizer,
        learning_rate = args.learning_rate,
        n_sampled_points = args.n_points,
        resampling_freq = args.resampling_freq,
        n_resample = args.n_resample,
    )
    metadata.save_to_file(os.path.join(OUTPUT_DIR, "metadata.toml"))


    ####### Training
    model = io.initialize_model(metadata).to(DEVICE)
    print(f"Initialized neural network with {IL.nn.count_parameters(model)} parameters")

    trainer = NeuralWrappingTrainer(
        geometry,
        TrainingConfig(
            BATCH_SIZE = args.batch_size,
            TEST_BATCH_SIZE = args.test_batch_size,
            N_EPOCHS=args.n_resample * args.resampling_freq,
            LEARNING_RATE=args.learning_rate,
            DEVICE=DEVICE,
            OPTIMIZER=args.optimizer),
        weight_attach = 100.,
        resample_freq = args.resampling_freq,
        resample_surface= not args.no_resample_surface
    )

    if geometry.dim == 3:
        trainer.add_callbacks( callbacks.MarchingCubeCB(OUTPUT_DIR, args.resampling_freq, res=300, iso=[0., 0.01]) )
    else:
        trainer.add_callbacks( callbacks.Render2DCB(OUTPUT_DIR, args.resampling_freq))
    trainer.add_callbacks(
        callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
        callbacks.CheckpointCB(OUTPUT_DIR, args.resampling_freq),
        SaveTrainingPointsCB(OUTPUT_DIR, args.resampling_freq)
    )

    trainer.initialize_training_data(args.n_points)
    M.mesh.save(M.mesh.from_arrays(trainer.points["on"]), os.path.join(OUTPUT_DIR, "points_e0_on.xyz"))
    M.mesh.save(M.mesh.from_arrays(trainer.points["out"]), os.path.join(OUTPUT_DIR, "points_e0_out.xyz"))
    trainer.train(model)
    torch.save(model.state_dict(),os.path.join(OUTPUT_DIR, "model_final.pt"))

