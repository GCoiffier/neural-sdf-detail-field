import os
import mouette as M
import argparse
import numpy as np

import torch
import implicitlab as IL
from implicitlab.training import TrainingConfig
from implicitlab.training import callbacks

from src import ImplicitSurfaceTrainer, KappaUpdateCallback, SirenDisplacementField, MetaData

"""
Reimplementation of:

Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields, Wang Yifan, Lukas Rahmann, Olga Sorkine-Hornung, 2022
https://arxiv.org/abs/2106.05187
"""

if __name__ == "__main__":
    np.random.seed(42)
        
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_geometry", type=str, help="path to the input geometry file. Supported file types are .obj, .mesh, .stl and .geogram_ascii")
    argument_parser.add_argument("-o", "--output-dir", type=str, default="", help="name of the output folder")
    argument_parser.add_argument("-np", "--n-points", type=int, default=200_000, help="Number of sampled point in the training dataset")

    argument_parser.add_argument("--test-batch-size", type=int, default=10_000)
    argument_parser.add_argument("-ne", "--n-epochs", type=int, default=300, help="number of epochs")

    argument_parser.add_argument("-nl", "--n-layers", type=int, default=5, help="Number of layers in the neural network")
    argument_parser.add_argument("-ls", "--layer-size", type=int, default=128, help="size of each layer in the neural network")
    args = argument_parser.parse_args()


    ####### Prepare environment and load geometry
    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()
    print("Training will run on the following device:", DEVICE)

    geometry = IL.load_geometry(args.input_geometry)
    assert geometry.dim == 3
    print("Read input geometry of type", geometry.geom_type)
    
    if len(args.output_dir)>0:
        OUTPUT_DIR = os.path.join("output", args.output_dir)
    else:
        OUTPUT_DIR = os.path.join("output",  M.utils.get_filename(args.input_geometry), "idf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Will save model in {OUTPUT_DIR} folder")
    M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_geometry.obj"))

    metadata = MetaData(
        geometry_dim = geometry.dim,
        architecture_type = "IDF",
        n_layers = args.n_layers,
        layer_size = args.layer_size,
        n_epochs = args.n_epochs,
        optimizer = "adam",
        learning_rate = 5e-5,
        n_sampled_points = args.n_points,
        batch_size = 4096,
        test_batch_size = args.test_batch_size
    )
    metadata.save_to_file(os.path.join(OUTPUT_DIR, "metadata.toml"))


    ####### Dataset Sampling
    points, normals = M.sampling.sample_surface(geometry, args.n_points, return_normals=True)
    train_data = IL.data.make_tensor_dataset((points, normals), DEVICE)

    pc = M.mesh.from_arrays(points)
    M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))

    ###### Training 
    # Setup model
    model = SirenDisplacementField(args.layer_size, args.n_layers).to(DEVICE)
    print(f"{IL.nn.count_parameters(model)} parameters")

    trainer = ImplicitSurfaceTrainer(
        TrainingConfig(
            BATCH_SIZE = metadata.batch_size,
            TEST_BATCH_SIZE = metadata.test_batch_size,
            N_EPOCHS = metadata.n_epochs,
            LEARNING_RATE = metadata.learning_rate,
            DEVICE = DEVICE
    ))
    trainer.weights = {"eikonal" : 5., "on" : 400., "normals": 40., "out" : 50.,} # weights from the paper
    trainer.add_callbacks([
        callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
        callbacks.MarchingCubeCB(OUTPUT_DIR, args.n_epochs, res=400, iso=0.),
        KappaUpdateCallback(0.2, args.n_epochs) # update the kappa parameter according to the correct schedule
    ])
    trainer.set_training_data(train_data)
    trainer.train(model)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "weights_final.pt"))