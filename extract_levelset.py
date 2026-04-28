import implicitlab as IL
import mouette as M
import os,sys
from src import MetaData, load_model
import numpy as np

DEVICE = IL.utils.get_device()
folder_path = sys.argv[1]
metadata = MetaData.load_from_file(os.path.join(folder_path, "metadata.toml"))

model = load_model(metadata, sys.argv[1], DEVICE)
print(model.kappa)

if len(sys.argv)>2:
    geometry = IL.data.load_geometry(sys.argv[2])
    domain = M.geometry.AABB.of_mesh(geometry, 0.1)
else:
    domain = M.geometry.AABB([-1.5]*3, [1.5]*3)

# data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, iso=[float(x) for x in np.linspace(0, 0.3, 31)], res=400, use_tqdm=True)
data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, iso=[0., 0.1], res=400, use_tqdm=True)
for name, iso in data.items():
    M.mesh.save(iso, f"iso_{name}.obj")