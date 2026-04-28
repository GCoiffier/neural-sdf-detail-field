import os
import argparse
import mouette as M
import numpy as np
import torch

import implicitlab as IL
from implicitlab.data import PointSampler
from implicitlab.training import TrainingConfig, hKRTrainer, callbacks
from src import MetaData

if __name__ == "__main__":
    np.random.seed(42)
    
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_geometry_file", type=str, help="path to the input geometry file. Supported file types are .obj, .mesh, .stl and .geogram_ascii")
    argument_parser.add_argument("-o", "--output-dir", type=str, default="", help="name of the output folder")
    argument_parser.add_argument("-np", "--n-points", type=int, default=300_000)

    argument_parser.add_argument("--optimizer", type=str, choices=["adam", "muon", "sgd"], default="muon")
    argument_parser.add_argument("--learning-rate", type=float, default=1e-4)
    argument_parser.add_argument("--batch-size", type=int, default=1000)
    argument_parser.add_argument("--test-batch-size", type=int, default=5000)
    argument_parser.add_argument("-ne", "--n-epochs", type=int, default=300, help="number of epochs")
    argument_parser.add_argument("--checkpoint-freq", type=int, default=100, help="frequency at which the model is saved on the disk (in terms of number of epochs)")

    argument_parser.add_argument("-m", "--margin", type=float, default=0.01, help="hkr margin parameter")

    argument_parser.add_argument("-nl", "--n-layers", type=int, default=10)
    argument_parser.add_argument("-ls", "--layer-size", type=int, default=128)

    ####### Prepare environment and load geometry
    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()
    print("Training will run on the following device:", DEVICE)

    geometry = IL.load_geometry(args.input_geometry_file)
    print("Read input geometry of type", geometry.geom_type)
    
    if len(args.output_dir)>0:
        OUTPUT_DIR = os.path.join("output", args.output_dir)
    else:
        OUTPUT_DIR = os.path.join("output",  M.utils.get_filename(args.input_geometry_file), "hkr")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Will save model in {OUTPUT_DIR} folder")
    M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_geometry.obj"))

    ####### Save metadata file
    metadata = MetaData(
        geometry_dim = geometry.dim,
        architecture_type = "SLL",
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
    train_field = IL.fields.Occupancy(geometry, v_in=-1, v_out=1, v_on=-1)
    train_sampling_strat = IL.sampling_strategy.CombinedStrategy([
        IL.sampling_strategy.UniformBox(geometry),
        IL.sampling_strategy.NearGeometryGaussian(geometry, 0.02)
    ], [1., 4.]) # 80% of points near the surface, 20% uniformly in bounding box
    train_sampler = PointSampler(geometry, train_sampling_strat, train_field)
    points, val = train_sampler.sample(args.n_points)

    # Balance the dataset : as many inside points that there are outside points
    points_pos = points[val>0, :]
    points_neg = points[val<0, :]
    n_pos, n_neg = points_pos.shape[0], points_neg.shape[0]
    print(n_pos, "sampled outside points")
    print(n_neg, "sampled inside points")
    if n_pos<n_neg:
        points_neg = points_neg[:n_pos, :]
    elif n_pos>n_neg:
        points_pos = points_pos[:n_neg, :]
    points = np.concatenate((points_pos, points_neg))
    val = np.concatenate((np.ones(min(n_pos,n_neg)), -np.ones(min(n_pos,n_neg))))
    train_data = IL.data.make_tensor_dataset((points, val), DEVICE) 

    pc = M.mesh.from_arrays(points)
    pc.vertices.register_array_as_attribute("occ", val)
    M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))

    ####### Training
    model = IL.nn.DenseLipSDP(metadata.geometry_dim, metadata.layer_size, metadata.n_layers).to(DEVICE)
    print(f"Initialized neural network with {IL.nn.count_parameters(model)} parameters")

    # Setup trainer
    class UpdateHkrRegulCB(callbacks.Callback):
        def __init__(self, when : dict):
            super().__init__()
            self.when = when

        def callOnBeginTrain(self, trainer, model):
            epoch = trainer.metrics["epoch"]
            if epoch in self.when:
                trainer.lossfun.lmbd = self.when[epoch]
                print("Updated loss regul weight to", self.when[epoch])

    trainer = hKRTrainer(TrainingConfig(
        BATCH_SIZE=args.batch_size,
        TEST_BATCH_SIZE=args.test_batch_size,
        N_EPOCHS=args.n_epochs,
        LEARNING_RATE=args.learning_rate,
        DEVICE=DEVICE,
        OPTIMIZER=args.optimizer), 
        margin=args.margin, 
        lmbd=10.
    )

    trainer.add_callbacks(
        callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
        UpdateHkrRegulCB({args.n_epochs//3: 100., 2*(args.n_epochs//3) : 1000.}),
        callbacks.CheckpointCB(OUTPUT_DIR, args.checkpoint_freq, only_weights=True)
    )
    if geometry.dim == 2:
        trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, args.n_epochs))
    elif geometry.dim == 3:
        if isinstance(geometry, M.mesh.SurfaceMesh):
            domain = M.geometry.AABB.of_mesh(geometry, 0.1)
        else:
            domain = None
        trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, args.n_epochs, res=300, domain=domain, iso=0.))

    trainer.set_training_data(train_data)
    trainer.train(model)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "weights_final.pt"))
