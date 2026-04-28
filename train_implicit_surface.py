import os
import mouette as M
import torch
import argparse
import numpy as np

import implicitlab as IL
from implicitlab.training import TrainingConfig
from implicitlab.training import callbacks

from src import MetaData, ImplicitSurfaceTrainer

if __name__ == "__main__":
    np.random.seed(42)
        
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_geometry", type=str, help="path to the input geometry file. Supported file types are .obj, .mesh, .stl and .geogram_ascii")
    argument_parser.add_argument("-o", "--output-dir", type=str, default="", help="name of the output folder")
    argument_parser.add_argument("-np", "--n-points", type=int, default=300_000, help="Number of sampled point in the training dataset")

    argument_parser.add_argument("--optimizer", type=str, choices=["adam", "muon", "sgd"], default="adam")
    argument_parser.add_argument("--learning-rate", type=float, default=1e-4)
    argument_parser.add_argument("--batch-size", type=int, default=1000)
    argument_parser.add_argument("--test-batch-size", type=int, default=5000)
    argument_parser.add_argument("-ne", "--n-epochs", type=int, default=150, help="number of epochs")

    argument_parser.add_argument("--architecture", type=str, default="SIREN", choices=["SIREN", "RFF", "MLP"])
    argument_parser.add_argument("-nl", "--n-layers", type=int, default=6, help="Number of layers in the neural network")
    argument_parser.add_argument("-ls", "--layer-size", type=int, default=256, help="size of each layer in the neural network")
    args = argument_parser.parse_args()

    ####### Prepare environment and load geometry
    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()
    print("Training will run on the following device:", DEVICE)

    geometry = IL.load_geometry(args.input_geometry)
    print("Read input geometry of type", geometry.geom_type)
    
    if len(args.output_dir)>0:
        OUTPUT_DIR = os.path.join("output", args.output_dir)
    else:
        OUTPUT_DIR = os.path.join("output",  M.utils.get_filename(args.input_geometry), args.architecture.lower())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Will save model in {OUTPUT_DIR} folder")
    M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_geometry.obj"))

    ####### Save metadata file
    metadata = MetaData(
        geometry_dim = geometry.dim,
        architecture_type = args.architecture,
        n_layers = args.n_layers,
        layer_size = args.layer_size,
        n_epochs = args.n_epochs,
        optimizer = args.optimizer,
        learning_rate = args.learning_rate,
        n_sampled_points = args.n_points,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size
    )
    metadata.save_to_file(os.path.join(OUTPUT_DIR, "metadata.toml"))

    ####### Dataset Sampling
    if geometry.dim == 3:
        points, normals = M.sampling.sample_surface(geometry, args.n_points, return_normals=True)
    elif geometry.dim == 2:
        points, normals = IL.data.sample_points_and_normals2D(geometry, args.n_points)
    train_data = IL.data.make_tensor_dataset((points, normals), DEVICE)

    pc = M.mesh.from_arrays(points)
    M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))

    ###### Training 

    # Setup model
    if metadata.architecture_type == "SIREN":
        model = IL.nn.SirenNet(geometry.dim, args.layer_size, args.n_layers).to(DEVICE)
    elif metadata.architecture_type == "RFF":
        model = torch.nn.Sequential(
            IL.nn.encodings.RandomFourierEncoding(geometry.dim, 1000),
            IL.nn.MultiLayerPerceptron(1000, args.layer_size, args.n_layers)
        ).to(DEVICE)
    elif metadata.architecture_type == "MLP":
        model = IL.nn.MultiLayerPerceptron(geometry.dim, args.layer_size, args.n_layers).to(DEVICE)
    print(f"{IL.nn.count_parameters(model)} parameters")

    # Setup trainer
    trainer = ImplicitSurfaceTrainer(
        TrainingConfig(
            BATCH_SIZE      = metadata.batch_size,
            TEST_BATCH_SIZE = metadata.test_batch_size,
            N_EPOCHS        = metadata.n_epochs,
            LEARNING_RATE   = metadata.learning_rate,
            OPTIMIZER       = metadata.optimizer,
            DEVICE          = DEVICE
    ))

    trainer.add_callbacks(callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")))
    if geometry.dim == 2:
        trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, metadata.n_epochs))
    elif geometry.dim == 3:
        trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, metadata.n_epochs, res=400, iso=0.))
    trainer.set_training_data(train_data)
    trainer.train(model)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "weights_final.pt"))