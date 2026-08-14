import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import mouette as M
from mouette.mesh.datatypes import *
from scipy.spatial import KDTree

class ChamferDistance(M.Worker):

    @allowed_mesh_types(M.mesh.SurfaceMesh)
    def __init__(self, mesh : M.mesh.SurfaceMesh, n_samples: int, verbose = False):
        super().__init__("Chamfer Dist.", verbose)

        self.mesh = mesh

        self.n_samples = n_samples
        self.samples = M.sampling.sample_surface(self.mesh, self.n_samples)
        self.tree = KDTree(self.samples)


    @allowed_mesh_types(M.mesh.SurfaceMesh)
    def run(self, target: M.mesh.SurfaceMesh, return_point_cloud:bool = False):
        target_samples = M.sampling.sample_surface(target, self.n_samples)
        dist_target, _ = self.tree.query(target_samples, 1)
        dist_target = np.squeeze(dist_target)

        target_tree = KDTree(target_samples)
        dist_orig, _ = target_tree.query(self.samples, 1)
        dist_orig = np.squeeze(dist_orig)

        distance = np.mean(dist_target) + np.mean(dist_orig)
        
        if return_point_cloud:
            pc = M.mesh.from_arrays(target_samples)
            pc.vertices.register_array_as_attribute("chamfer", dist_orig+dist_target)
            return distance, pc
        
        return distance
    

if __name__ == "__main__":
    orig_mesh = M.mesh.load(sys.argv[1])
    target_mesh = M.mesh.load(sys.argv[2])

    CD = ChamferDistance(orig_mesh, 1_000_000)
    d, pc = CD.run(target_mesh, True)
    print(d)
    M.mesh.save(pc, "chamfer.geogram_ascii")