# RBF Detail field for neural implicit surfaces

This repository contains the code accompanying our SGP 2026 publication:
_Compactly supported detail field for high quality neural implicit surfaces_, Guillaume Coiffier & Justine Basselin.

Here is a link to the project's page: _link_


## Installation / Environment setup

First clone the repository and create a python virtual environment:

```bash
git clone https://github.com/GCoiffier/neural-sdf-detail-field.git
cd neural-sdf-detail-field
python -m venv neuraldetail
source neuraldetail/bin/activate
```

Then install the dependencies:
```bash
pip install -r requirements.txt
```

This code has been tested under python 3.12 with pytorch 2.11 and cuda 13.0

## Finding input data

Input data used in this project can be downloaded from:

- The Stanford 3D Scanning Repository: [https://graphics.stanford.edu/data/3Dscanrep/](https://graphics.stanford.edu/data/3Dscanrep/)
- [https://threedscans.com/](https://threedscans.com/)
- Designs from user YahooJapan on thingiverse: [https://www.thingiverse.com/YahooJAPAN/designs](https://www.thingiverse.com/YahooJAPAN/designs)
- Thingi10k: [https://ten-thousand-models.appspot.com/](https://ten-thousand-models.appspot.com/)

Our code supports input in the form of `.stl`, `.obj`, `.ply`, `.mesh` and `.geogram_ascii` files.

## Running the code

Running our code consists of two main parts:
1) Train a neural implicit to approximate the signed distance field of the input geometry;
2) Use our RBF field to correct this neural field to better account for surface details.

#### Train the neural implicit

```bash
python train_hkr.py <path/to/input/mesh>
```

This creates a folder `output/<geometry_name>/hkr` inside which the output will be written. Output meshes are given as wavefront .obj files. Debug meshes are provided as `.geogram_ascii` files. They are meant to be read by the [GraphiteThree](https://github.com/BrunoLevy/GraphiteThree) software. Run with the `-h` flag to see all commandline parameters and arguments.

#### Run the detail field

Once a neural field has been trained onto the input geometry, a detail field can be computed using this command:

```bash
python compute_detail_field.py output/<geometry_name>/hkr -nc 200000 -res 400
```

where the first argument is the folder created by the previous script that contains all of the output files. The `-nc` argument specifies the number of basis functions to consider. Typical values range from 100k to 500k. The `-res` argument specifies the resolution of the final marching cube extraction (400 is usually enough). Run the `-h` flag for help about all commandline arguments.

## Comparisons and additional experiments

We provide re-implementation of various previous works on neural distance fields for comparison. These can be substituted to the 1-Lip training.
Technically, a RBF detail field can then be computed for all of these models, but it is usually not needed. Additional scripts used to produce results are available on the `dev` branch.

#### Hotspot [2]
```bash
python train_hotspot.py <path/to/input/mesh>
```

#### Siren Network [3]
```bash
python train_implicit_surface.py <path/to/input/mesh> --architecture SIREN
```

#### Random Fourier Features [4]
```bash
python train_implicit_surface.py <path/to/input/mesh> --architecture RFF --layer-size 256 --n-layers 10
```

#### Implicit Displacement field [5]
```bash
python train_idf.py <path/to/input/mesh>
```


## References

[1] _1-Lipschitz Neural Distance Fields_, G.Coiffier & L.Béthune (2024)

[2] _Hotspot_

[3] _SIREN_

[4] _RFF_

[5] _IDF_