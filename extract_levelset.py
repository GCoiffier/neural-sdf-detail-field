import implicitlab as IL
import mouette as M
import sys
from src.io import load_model
import numpy as np
import torch

DEVICE = IL.utils.get_device()
model = load_model(sys.argv[1], DEVICE, ignore_detail_field=False)
# model = IL.nn.SirenNet(3, 256, 6).to(DEVICE)
# model = IL.nn.DenseLipSDP(3, 128, 10).to(DEVICE)
# model.load_state_dict(torch.load(sys.argv[1]))

if len(sys.argv)>2:
    geometry = IL.data.load_geometry(sys.argv[2])
    domain = M.geometry.AABB.of_mesh(geometry, 0.1)
else:
    domain = M.geometry.AABB([-1.5]*3, [1.5]*3)

data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, iso=[float(x) for x in np.linspace(0, 0.3, 31)], res=400, use_tqdm=True)
# data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, iso=[0., 0.05], res=400, use_tqdm=True)
for name, iso in data.items():
    M.mesh.save(iso, f"iso_{name}.obj")