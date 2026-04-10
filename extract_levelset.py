import implicitlab as IL
import mouette as M
import sys
from src.io import load_model
import torch

DEVICE = IL.utils.get_device()
model = load_model(sys.argv[1], DEVICE, ignore_detail_field=False)
# model = IL.nn.SirenNet(3, 256, 6).to(DEVICE)
# model.load_state_dict(torch.load(sys.argv[1]))

if len(sys.argv)>2:
    geometry = IL.data.load_geometry(sys.argv[2])
    domain = M.geometry.AABB.of_mesh(geometry, 0.1)
else:
    domain = M.geometry.AABB([-1.2]*3, [1.2]*3)

# data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, iso=[0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], res=500, use_tqdm=True)
data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, iso=[0., 0.05], res=400, use_tqdm=True)
for name, iso in data.items():
    M.mesh.save(iso, f"iso_{name}.obj")