import os, sys
import mouette as M
import numpy as np
from scipy import sparse as sp
from scipy.spatial import KDTree
import torch
from tqdm import tqdm


class CompactSupportRBFInterpolant(M.Worker):

    def __init__(self, points: np.ndarray, values: np.ndarray, verbose:bool = True, **kwargs):
        super().__init__("CompactSplines", verbose)

        self.points : np.ndarray = np.asarray(points)
        self.values : np.ndarray = np.asarray(values)
        self.alpha : float = kwargs.get("alpha", 3*np.max(np.abs(self.values)))
        self.weights = None
        self.tree = KDTree(self.points)

    @classmethod
    def load_from_file(cls, folder_path):
        return

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def rbf(self, r):
        r = r/self.alpha
        return np.where(r>1, 0., np.pow(1-r,4)*(4*r+1))
    
    def _evaluate_rbf(self, x):
        near_x = self.tree.query_ball_point(x, self.alpha)
        if not near_x: return 0.
        dist_values = np.linalg.norm(self.points[near_x]-x, axis=1)
        return np.dot(self.weights[near_x], self.rbf(dist_values))

    def run(self):
        N = self.n_points
        vals, rows, cols = [], [], []
        self.log("Build system")
        for i in range(N):
            pi = self.points[i]
            near_i = self.tree.query_ball_point(self.points[i], self.alpha)
            for j in near_i:
                pj = self.points[j]
                phi_ij = self.rbf(M.geometry.distance(pi, pj))
                vals.append(phi_ij)
                rows.append(i)
                cols.append(j)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        self.log("System's sparsity:", f"{A.nnz}/{N*N} ({100*A.nnz/N/N:.2f} %)")
        self.log("Solve system")
        self.weights = sp.linalg.spsolve(A, self.values)
        self.log("Done")
        
    def __call__(self, x):
        if self.weights is None: self.run()
        return self._evaluate_rbf(x)