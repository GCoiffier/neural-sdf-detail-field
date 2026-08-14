import sys

import numpy as np
import mouette as M
from mouette.mesh.datatypes import *
import igl

orig_mesh = M.mesh.load(sys.argv[1])
target_mesh = M.mesh.load(sys.argv[2])

distances = igl.signed_distance(np.asarray(orig_mesh.vertices), np.asarray(target_mesh.vertices), np.asarray(target_mesh.faces))[0]
distances = np.abs(distances)
print(np.max(distances))

distances_attr = orig_mesh.vertices.register_array_as_attribute("dist", distances)
M.attributes.uv_export.generate_uv_colormap_vertices(orig_mesh, distances_attr, 0., 0.01)

if len(sys.argv)>3:
    M.mesh.save(orig_mesh, sys.argv[3])
else:
    M.mesh.save(orig_mesh, "mesh_with_distance.obj")