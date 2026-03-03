import argparse

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_geometry_file", type=str, help="path to the input geometry file. Supported file types are .obj, .mesh, .stl and .geogram_ascii")
    argument_parser.add_argument("-o", "--output-dir", type=str, default="output", help="name of the output folder")

    # Training parameters
    argument_parser.add_argument("--n-points", type=int, default=10_000)
    argument_parser.add_argument("--n-resample", type=int, default=20)
    argument_parser.add_argument("--n-epochs", type=int, default=10)
    argument_parser.add_argument("--batch-size", type=int, default=100)
    argument_parser.add_argument("--test-batch-size", type=int, default=5000)
    argument_parser.add_argument("--checkpoint-freq", type=int, default=50, help="")
    argument_parser.add_argument("--render-freq", type=int, default=10, help="")
    argument_parser.add_argument("--sampling-freq", type=int, default=10)

    # Optimizer parameters
    argument_parser.add_argument("--optimizer", type=str, choices=["adam", "muon", "sgd"], default="adam")
    argument_parser.add_argument("--learning-rate", type=float, default=1e-3)

    # Model paramaters
    argument_parser.add_argument("-nl", "--n-layers", type=int, default=10)
    argument_parser.add_argument("-ls", "--layer-size", type=int, default=128)

    args = argument_parser.parse_args()
    
    for key,value in vars(args).items():
        print(key,"=", value)