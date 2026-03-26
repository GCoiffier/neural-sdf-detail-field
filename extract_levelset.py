import implicitlab as IL
import mouette as M
import sys
from src.io import load_model

DEVICE = IL.utils.get_device()
model = load_model(sys.argv[1], DEVICE, ignore_detail_field=True)

if len(sys.argv)>2:
    geometry = IL.data.load_geometry(sys.argv[2])
    domain = M.geometry.AABB.of_mesh(geometry, 0.1)
else:
    domain = M.geometry.AABB([-1.2]*3, [1.2]*3)
data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, res=300)
for name, iso in data.items():
    M.mesh.save(iso, "iso.obj")

    