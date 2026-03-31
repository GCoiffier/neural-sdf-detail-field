import os, sys
import mouette as M
import numpy as np
from scipy import sparse as sp
from scipy.spatial import KDTree
import torch
from tqdm import tqdm


class CompactSupportRBFInterpolant(M.Worker):

    def __init__(self, points: np.ndarray, values: np.ndarray, verbose:bool = True, **kwargs):
        super().__init__("CompactSupportRBF", verbose)

        self.points : np.ndarray = np.asarray(points)
        self.values : np.ndarray = np.asarray(values)
        self.alpha : float = kwargs.get("alpha", 2*np.max(np.abs(self.values)))
        self.weights = None
        self.tree = kwargs.get("tree", KDTree(self.points))

    @classmethod
    def load_from_file(cls, file_path: str):
        data = torch.load(file_path, weights_only=False)
        rbf = cls(data["centers"], data["values"], alpha=data["alpha"])
        rbf.weights = data["weights"]
        return rbf
    
    def save_to_file(self, file_path:str):
        if self.weights is None:
            print("Weights have not been computed. RBF won't be saved.")
            return
        to_save = {
            "centers" : self.points,
            "values" : self.values,
            "alpha" : self.alpha,
            "weights" : self.weights,
        }
        torch.save(to_save, file_path)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.where(r>1, 0., np.pow(1-r,4)*(4*r+1))
    
    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.where(r>1, 0., np.pow(1-r,3)*(3*r+1))

    def rbf(self, r):
        r = r/self.alpha
        return np.where(r>1, 0., np.pow(1-r,2))

    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.maximum(1. - np.sign(r)*r, 0.)

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
                if abs(phi_ij)>1e-16:
                    vals.append(phi_ij)
                    rows.append(i)
                    cols.append(j)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        self.log("System's sparsity:", f"{A.nnz}/{N*N} ({100*A.nnz/N/N:.2f} %)")
        self.log("Solve system")
        self.weights = sp.linalg.cg(A, self.values)[0]
        # self.weights = sp.linalg.spsolve(A, self.values)
        self.log("Done")
        
    def __call__(self, x):
        if self.weights is None: self.run()
        if len(x.shape)==1:
            return self._evaluate_rbf_single(x)
        elif len(x.shape)==2:
            return self._evaluate_rbf_bulk(x, 1_000_000)
        
    def _evaluate_rbf_single(self, x):
        near_x = self.tree.query_ball_point(x, self.alpha)
        if not near_x: return 0.
        dist_values = np.linalg.norm(self.points[near_x]-x, axis=1)
        return np.dot(self.weights[near_x], self.rbf(dist_values))

    def _evaluate_rbf_bulk(self, x, batch_size):
        n_queries = x.shape[0]
        n_batches = n_queries//batch_size+1
        rbf_values = np.zeros(n_queries)
        for i_batch in range(n_batches):
            print(f"Batch {i_batch+1}/{n_batches}")
            x_batch = x[i_batch*batch_size:(i_batch+1)*batch_size]
            query_tree = KDTree(x_batch)
            near = query_tree.query_ball_tree(self.tree, self.alpha)
            for i,near_i in tqdm(enumerate(near), total=batch_size):
                if len(near_i)==0: continue
                dist = np.linalg.norm(self.points[near_i]-x_batch[i], axis=1)
                rbf_values[i_batch*batch_size+i] = np.dot(self.weights[near_i], self.rbf(dist))
        return rbf_values
    


class AdaptativeSupportRBFInterpolant(M.Worker):

    def __init__(self, points: np.ndarray, values: np.ndarray, sizes : np.ndarray, verbose:bool = True, **kwargs):
        super().__init__("AdaptSupportRBF", verbose)

        self.points : np.ndarray = np.asarray(points)
        self.values : np.ndarray = np.asarray(values).squeeze()
        self.sizes : np.ndarray = np.asarray(sizes).squeeze()
        self.alpha = np.max(np.abs(self.sizes))
        self.weights = None
        self.tree = kwargs.get("tree", KDTree(self.points))

    @classmethod
    def load_from_file(cls, file_path: str):
        data = torch.load(file_path)
        rbf = cls(data["centers"], data["values"], sizes=data["sizes"])
        rbf.weights = data["weights"]
        return rbf
    
    def save_to_file(self, file_path:str):
        if self.weights is None:
            print("Weights have not been computed. RBF won't be saved.")
            return
        to_save = {
            "centers" : self.points,
            "values" : self.values,
            "sizes" : self.sizes,
            "weights" : self.weights,
        }
        torch.save(to_save, file_path)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def rbf(self, r, s):
        r = r/s
        return np.where(r>1, 0., np.pow(1-r,4)*(4*r+1))
    
    # def rbf(self, r, s):
    #     r = r/s
    #     return np.where(r>1, 0., np.pow(1-r,3)*(3*r+1))

    # def rbf(self, r, s):
    #     r = r/s
    #     return np.where(r>1, 0., np.pow(1-r,2))

    # def rbf(self, r, s):
    #     r = r/s
    #     return np.maximum(1.-r, 0.)

    def run(self):
        N = self.n_points
        vals, rows, cols = [], [], []
        self.log("Build system")
        for i in range(N):
            pi = self.points[i]
            near_i = self.tree.query_ball_point(self.points[i], self.alpha)
            for j in near_i:
                pj = self.points[j]
                phi_ij = self.rbf(M.geometry.distance(pi, pj), self.sizes[i])
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
        if len(x.shape)==1:
            return self._evaluate_rbf_single(x)
        elif len(x.shape)==2:
            print("Evaluate RBF in bulk")
            return self._evaluate_rbf_bulk(x, batch_size=1_000_000)


    def _evaluate_rbf_single(self, x):
        near_x = self.tree.query_ball_point(x, self.alpha)
        if not near_x: return 0.
        dist_values = np.linalg.norm(self.points[near_x]-x, axis=1)
        return np.dot(self.weights[near_x], self.rbf(dist_values, self.sizes[near_x]))

    def _evaluate_rbf_bulk(self, x, batch_size):
        n_queries = x.shape[0]
        n_batches = n_queries//batch_size+1
        rbf_values = np.zeros(n_queries)
        for i_batch in range(n_batches):
            print(f"Batch {i_batch+1}/{n_batches}")
            x_batch = x[i_batch*batch_size:(i_batch+1)*batch_size]
            query_tree = KDTree(x_batch)
            near = query_tree.query_ball_tree(self.tree, self.alpha)
            for i,near_i in tqdm(enumerate(near), total=batch_size):
                if len(near_i)==0: continue
                dist = np.linalg.norm(self.points[near_i]-x_batch[i], axis=1)
                rbf_values[i+i_batch*batch_size] = np.dot(self.weights[near_i], self.rbf(dist, self.sizes[near_i]))
        return rbf_values