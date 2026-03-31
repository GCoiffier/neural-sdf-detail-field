import os, sys
import mouette as M
import numpy as np
from scipy import sparse as sp
from scipy.spatial import KDTree
import torch
from tqdm import tqdm, trange
from time import time


def rbf0(r, a):
    r = r/a
    return torch.where(r>1, 0., 1-r)

def rbf1(r, a):
    r = r/a
    return torch.where(r>1., 0., torch.pow(1-r,2.))

def rbf2(r, a):
    r = r/a
    return torch.where(r>1., 0., torch.pow(1-r,3.))

def rbf3(r, a):
    r = r/a
    return np.where(r>1, 0., np.pow(1-r,3)*(3*r+1))

def rbf4(r, a):
    r = r/a
    return np.where(r>1, 0., np.pow(1-r,4)*(4*r+1))


class CompactSupportRBFInterpolantTorch(torch.nn.Module):

    def __init__(self, points: np.ndarray, values: np.ndarray, alpha: float, **kwargs):
        super().__init__()

        self.values = np.asarray(values)
        self.alpha : float = alpha
        self.tree : KDTree = kwargs.get("tree", KDTree(points))
        self.points = torch.Tensor(points)
        self.weights : torch.Tensor = None

        shape = kwargs.get("rbf_shape", 1)

        self.rbf = lambda x : [
            rbf0, rbf1, rbf2, rbf3, rbf4
        ][shape](x, self.alpha)


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
    

    def run(self):
        N = self.n_points
        vals, rows, cols = [], [], []
        print("[RBF] Build system...")
        t0 = time()
        near = self.tree.query_pairs(self.alpha, output_type="ndarray")
        diff = self.points[near[:,0],:] - self.points[near[:,1],:]
        with torch.no_grad():
            phi_ij = self.rbf(torch.norm(diff, dim=1)).numpy()
        non_zero = phi_ij>1e-16
        vals = phi_ij[non_zero]
        inds = near[non_zero,:]
        rows, cols = inds[:,0], inds[:,1]
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        A = sp.eye(N,format="csr") + A + A.transpose() # don't forget transpose for symmetric terms
        print(f"[RBF] Built the system in {time() - t0:.3f} seconds")
        t0 = time()
        print("[RBF] System's sparsity:", f"{A.nnz}/{N*N} ({100*A.nnz/N/N:.2f} %)")
        print("[RBF] Solve system...")
        self.weights = sp.linalg.cg(A, self.values)[0]
        # self.weights = sp.linalg.spsolve(A, self.values)
        print(f"[RBF] Solved in {time()-t0:.3f} seconds")
        self.points = torch.Tensor(self.points).to("cpu")
        self.weights = torch.Tensor(self.weights).to("cpu")

    def prune(self, threshold:float):
        to_keep = torch.abs(self.weights)>threshold
        print(f"[RBF] Pruning basis functions with |weight|<{threshold}")
        
        n_init = self.points.shape[0]
        self.points = self.points[to_keep]
        self.values = self.values[to_keep]
        self.tree = KDTree(self.points.numpy())
        n_final = self.points.shape[0]
        n_removed = n_init - n_final
        print(f"[RBF] Removing {n_removed}/{n_init} basis functions ({100*n_removed/n_init:.1f}%)")
        print("[RBF] Recomputing weights")
        self.run()

    
    def _query_tree(self, x):
        if isinstance(x, np.ndarray):
            return self.tree.query_ball_point(x, self.alpha)
        elif isinstance(x, torch.Tensor):
            x_num = x.detach().cpu().numpy()
            return self.tree.query_ball_point(x_num, self.alpha)

    def forward(self, x):
        if self.weights is None: self.run()
        if len(x.shape)==1:
            return self._evaluate_rbf_single(x)
        elif len(x.shape)==2:
            return self._evaluate_rbf_bulk(x)
        else:
            raise Exception("bwaaaaaa", x.shape)

    def _evaluate_rbf_single(self, x : torch.Tensor):
        near_x = self._query_tree(x)
        if not near_x: return 0.
        dist_values = torch.norm(self.points[near_x]-x, dim=1)
        return torch.sum(self.weights[near_x] * self.rbf(dist_values))

    # def _evaluate_rbf_bulk(self, x):
    #     n_queries = x.shape[0]
    #     rbf_values = torch.zeros(n_queries)
    #     query_tree = KDTree(x.detach().cpu().numpy())
    #     near = query_tree.query_ball_tree(self.tree, self.alpha)
    #     for i,near_i in enumerate(near):
    #         if len(near_i)==0: continue
    #         dist = torch.norm(self.points[near_i]-x[i], dim=1)
    #         rbf_values[i] = torch.sum(self.weights[near_i] * self.rbf(dist))
    #     return rbf_values
    

    def _evaluate_rbf_bulk(self, x):
        n_queries = x.shape[0]
        rbf_values = torch.zeros(n_queries)
        query_tree = KDTree(x.detach().cpu().numpy(), compact_nodes=False)
        near = query_tree.query_ball_tree(self.tree, self.alpha)

        rows, cols = [], []
        for i,near_i in enumerate(near):
            rows.append(torch.full( (len(near_i),), fill_value=i, dtype=torch.int))
            cols.append(torch.tensor(near_i, dtype=torch.int))
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        dist = torch.norm(x[rows,:] - self.points[cols, :], dim=1)
        values = self.weights[cols] * self.rbf(dist)
        rbf_values = torch.zeros(n_queries)
        rbf_values.index_add_(0,rows, values)
        return rbf_values