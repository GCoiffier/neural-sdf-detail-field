import os
import argparse
import mouette as M
import numpy as np
from scipy import sparse as sp
from scipy.spatial import KDTree
from tqdm import tqdm

import implicitlab as IL
from time import time
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib import colors
import torch

class CompactSupportRBFInterpolantTorch(torch.nn.Module):

    def __init__(self, points: np.ndarray, values: np.ndarray, alpha: float, **kwargs):
        super().__init__()

        self.values = torch.Tensor(values)
        self.alpha : float = alpha
        self.tree : KDTree = kwargs.get("tree", KDTree(points))
        self.points = torch.Tensor(points)
        self.weights : torch.Tensor = None

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
    
    def rbf(self, r):
        r = r/self.alpha
        return torch.where(r>1., 0., torch.pow(1-r,2.))

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
        self.weights = sp.linalg.cg(A, self.values.numpy())[0]
        # self.weights = sp.linalg.spsolve(A, self.values)
        print(f"[RBF] Solved in {time()-t0:.3f} seconds")
        self.points = torch.Tensor(self.points).to("cpu")
        self.weights = torch.Tensor(self.weights).to("cpu")

    def prune(self, threshold:float):
        to_keep = torch.abs(self.weights)>threshold #*torch.abs(self.values.squeeze())
        print(f"[RBF] Pruning basis functions with |weight|<{threshold}")
        
        n_init = self.points.shape[0]
        tree_pruned : KDTree = KDTree(self.points[to_keep].numpy())
        pts_numpy = self.points.numpy()
        for i in trange(n_init):
            if to_keep[i] and len(tree_pruned.query_ball_point(pts_numpy[i], self.alpha/1.1))<2:
                to_keep[i] = False
        
        self.points = self.points[to_keep]
        self.values = self.values[to_keep]
        self.tree = KDTree(self.points.numpy())
        n_final = self.points.shape[0]
        n_removed = n_init - n_final
        print(f"[RBF] Removing {n_removed}/{n_init} basis functions ({100*n_removed/n_init:.1f}%)")
        print("[RBF] Recomputing weights")
        self.run()
        # self.weights = self.weights[to_keep]

    
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



OUTPUT_DIR = "RBF_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = IL.utils.get_device()
print("DEVICE:", DEVICE)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-n", "--n-points", type=int, default=3)
argument_parser.add_argument("-s", "--support-size", type=float, default=0.5)
args = argument_parser.parse_args()
N_pts = args.n_points
SUPPORT_SIZE = args.support_size

PLOT_DOMAIN = M.geometry.AABB([-1,-0.5], [1.,0.5])
X_centers = np.linspace(-0.7, 0.7, N_pts)
# Y_centers = 0.1*np.sin(4*np.pi*X_centers)
Y_centers = np.full_like(X_centers, fill_value=0.2)
points = np.vstack((X_centers,Y_centers)).T
SDF_VALUES = Y_centers # SDF function is dot(p, (0,1))

rbf = CompactSupportRBFInterpolantTorch(points, -SDF_VALUES, SUPPORT_SIZE)
rbf.run()
w = rbf.weights.detach().numpy()
print(f"RBF weights: min={np.min(w)} | max={np.max(w)} | mean={np.mean(w)}")

resolution = 2000
X = np.linspace(PLOT_DOMAIN.mini[0], PLOT_DOMAIN.maxi[0], resolution)
resY = round(resolution * PLOT_DOMAIN.span[1]/PLOT_DOMAIN.span[0])
Y = np.linspace(PLOT_DOMAIN.mini[1], PLOT_DOMAIN.maxi[1], resY)

pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
detail_values = rbf(torch.Tensor(pts)).detach().numpy()
base_values = pts[:,1]
total_values = base_values + detail_values

img_base = base_values.reshape((resolution, resY)).T
img_base = img_base[::-1,:]
print(np.min(img_base), np.max(img_base))

img_total = total_values.reshape((resolution,resY)).T
img_total = img_total[::-1,:]


plt.clf()
norm = colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
plt.imshow(img_total, cmap="bwr", norm=norm, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], origin='upper')
plt.axis("off")
# cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
# cs.changed()
plt.contour(img_base, levels=[0.], colors="red", linewidths=0.5, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], origin='upper')
plt.contour(img_total, levels=31, colors='k', linestyles="solid", linewidths=0.1, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], origin='upper')
plt.contour(img_total, levels=[0.], colors='k', linestyles="solid", linewidths=0.5, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], origin='upper')

plt.scatter(X_centers, Y_centers, s=5., marker="x", color="black")
for i in range(N_pts):
    plt.gca().add_patch(plt.Circle((X_centers[i], Y_centers[i]), rbf.alpha, color='b', fill=False))
plt.gca().set(xlim=(np.min(X), np.max(X)), ylim=(np.min(Y), np.max(Y)))

# plt.savefig(os.path.join(OUTPUT_DIR, f"contours_{SUPPORT_SIZE:.3f}.png"), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(OUTPUT_DIR, f"contours.png"), bbox_inches='tight', pad_inches=0, dpi=300)