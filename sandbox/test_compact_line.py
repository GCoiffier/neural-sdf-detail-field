import os, sys
import mouette as M
import numpy as np
from scipy import sparse as sp
from scipy.spatial import KDTree
from tqdm import tqdm

import implicitlab as IL

import matplotlib.pyplot as plt
from matplotlib import colors

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = IL.utils.get_device()
print("DEVICE:", DEVICE)


N_pts = 100
PLOT_DOMAIN = M.geometry.AABB([-1,-0.5], [1.,0.5])

X = np.linspace(-1, 1, N_pts)
Y = 0.1*np.sin(4*np.pi*X)
points = np.vstack((X,Y)).T
SDF_VALUES = Y # SDF function is dot(p, (1,0))

pc_init = M.mesh.from_arrays(points)
pc_init.vertices.register_array_as_attribute("sdf", Y)
M.mesh.save(pc_init, os.path.join(OUTPUT_DIR, "init_points.geogram_ascii"))


class CompactSupportPlateSpline(M.Worker):
    """
    Compact support with constant width (the function phi is the same for every point)
    """
    def __init__(self, points: np.ndarray, values: np.ndarray):
        super().__init__("CompactSplines", True)

        self.points : np.ndarray = np.asarray(points)
        self.values : np.ndarray = np.asarray(values)
        self.alpha : float = 3*np.max(np.abs(self.values))
        self.weights = None
        self.tree = KDTree(self.points)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def rbf(self, r):
        r = r/self.alpha
        return np.where(r>1, 0., np.pow(1-r,4)*(4*r+1))
    
    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.where(r>1-1e-5, 0., np.exp(1/(np.pow(r,2)-1)))

    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.maximum(1. - np.sign(r)*r, 0.)

    def _evaluate_rbf(self, x):
        near_x = self.tree.query_ball_point(x, self.alpha)
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

        print(N*N, A.nnz)
        self.log("Solve system")
        self.weights = sp.linalg.spsolve(A, self.values)
        self.log("Done")
        
    def __call__(self, x):
        if self.weights is None: self.run()
        return self._evaluate_rbf(x)
    

class CompactSupportPlateSpline2(M.Worker):
    """
    Compact support with variable width
    """
    def __init__(self, points: np.ndarray, values: np.ndarray):
        super().__init__("CompactSplines", True)

        self.points : np.ndarray = np.asarray(points)
        self.values : np.ndarray = np.asarray(values)
        self.widths : np.ndarray = 2*np.abs(np.asarray(values)) # distance from points to closest zero of the function to correct

        self.alpha : float = np.max(self.widths)
        self.weights = None
        self.tree = KDTree(self.points)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    # def rbf(self, r, w):
    #     r = 0.5*r/w
    #     return np.where(r>1, 0., np.pow(1-r,4)*(4*r+1))
    
    # def rbf(self, r, w):
    #     r = r/w
    #     return np.where(r>1-1e-5, 0., np.exp(1/(r*r-1)))

    def rbf(self, r, w):
        return np.maximum(1.01 - np.sign(r)*r/w, 0.)

    def _evaluate_rbf(self, x):
        near_x = self.tree.query_ball_point(x, self.alpha)
        dist_values = np.linalg.norm(self.points[near_x]-x, axis=1)
        return np.dot(self.weights[near_x], self.rbf(dist_values, self.widths[near_x]))

    def run(self):
        N = self.n_points
        vals, rows, cols = [], [], []
        self.log("Build system")
        for i in range(N):
            pi = self.points[i]
            near_i = self.tree.query_ball_point(self.points[i], self.alpha)
            for j in near_i:
                pj = self.points[j]
                phi_ij = self.rbf(M.geometry.distance(pi, pj), self.widths[i])
                vals.append(phi_ij)
                rows.append(i)
                cols.append(j)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        print(N*N, A.nnz)
        self.log("Solve system")
        self.weights = sp.linalg.spsolve(A, self.values)
        self.log("Done")
        
    def __call__(self, x):
        if self.weights is None: self.run()
        return self._evaluate_rbf(x)
    


rbf = CompactSupportPlateSpline(points, -SDF_VALUES)
# rbf = CompactSupportPlateSpline2(points, -SDF_VALUES)
rbf.run()


def render_detail_field(contour_path, details, domain : M.geometry.AABB, res=1000):
    assert domain.dim == 2

    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1]/domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    detail_values = []
    for pt in tqdm(pts, total=pts.shape[0]):
        detail_values.append(details(pt))
    detail_values = np.array(detail_values)
    total_values = pts[:,1] + detail_values

    img = total_values.reshape((res,resY)).T
    img = img[::-1,:]

    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.contour(img, levels=16, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)


render_detail_field(os.path.join(OUTPUT_DIR, "contour_final.png"), rbf, PLOT_DOMAIN)