import numpy as np
import mouette as M
from mouette.mesh.datatypes import *
from scipy.spatial import KDTree
import igl
import os
import sys

folder = sys.argv[1]

orig_mesh = M.mesh.load(os.path.join(folder, "hkr", "surface_with_details.obj"))
target_mesh = M.mesh.load(os.path.join(folder, "hkr", "input_geometry.obj"))

distances = igl.signed_distance(np.asarray(orig_mesh.vertices), np.asarray(target_mesh.vertices), np.asarray(target_mesh.faces))[0]
distances = np.abs(distances)
print(np.max(distances))

distances_attr = orig_mesh.vertices.register_array_as_attribute("dist", distances)
M.attributes.uv_export.generate_uv_colormap_vertices(orig_mesh, distances_attr, 0., 0.005)

M.mesh.save(orig_mesh, os.path.join(folder, "surface_with_distances.obj"))