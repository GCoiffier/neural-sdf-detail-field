import os, sys
import mouette as M
import numpy as np
from scipy import sparse as sp
from scipy.spatial import KDTree
import torch
from tqdm import tqdm

import implicitlab as IL
from implicitlab.data import PointSampler

import matplotlib.pyplot as plt
from matplotlib import colors

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
data = torch.load(sys.argv[2])
print(geometry.geom_type)
DEVICE = IL.utils.get_device()
print("DEVICE:", DEVICE)
NDF = IL.nn.DenseLipBjorck(geometry.dim, 128, 12).to(DEVICE)
NDF.load_state_dict(data)


N_pts = 10_000 if geometry.dim==2 else 50_000
PLOT_DOMAIN = M.geometry.AABB([-1.5,-1.5],[1.5,1.5])


class NeuralSDFValues(IL.fields.FieldGenerator):

    def __init__(self, ndf, device, batch_size=10_000):
        """A scalar field that corresponds to the output of a previously trained neural network.

        Args:
            ndf (torch.nn.Module): the neural network to query
            device (str): which device to use to call the neural network
            batch_size (int, optional): batch size for forward computation. Defaults to 10_000.
        """
        self.ndf = ndf
        self.device = device
        self.batch_size = batch_size

    def compute(self, query):
        return IL.utils.forward_in_batches(self.ndf, query, self.device, batch_size=self.batch_size)


train_sampler = PointSampler(
    geometry, 
    IL.sampling_strategy.UniformBox(geometry),
    NeuralSDFValues(NDF, DEVICE)
)
points, val = train_sampler.sample(N_pts,on_ratio=1.)

pc_init = M.mesh.from_arrays(points)
pc_init.vertices.register_array_as_attribute("ndf", val)
M.mesh.save(pc_init, os.path.join(OUTPUT_DIR, "init_points.geogram_ascii"))

MARGIN = np.max(np.abs(val))
print("Max error :", MARGIN)
IL.visualize.render_sdf_2d(None, os.path.join(OUTPUT_DIR, "contour_init.png"), os.path.join(OUTPUT_DIR, "grad_init.png"), NDF, PLOT_DOMAIN, device=DEVICE, batch_size=5000)

points_projected = IL.queries.project_onto_iso(points, NDF, 0., DEVICE, batch_size=10_000)

class CompactSupportPlateSpline(M.Worker):

    def __init__(self, points: np.ndarray, values: np.ndarray, **kwargs):
        super().__init__("CompactSplines", True)

        self.points : np.ndarray = np.asarray(points)
        self.values : np.ndarray = np.asarray(values)
        self.alpha : float = kwargs.get("alpha", 2*np.max(np.abs(self.values)))
        self.weights = None
        self.tree = KDTree(self.points)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.where(r>1, 0., np.pow(1-r,4)*(4*r+1))
    
    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.where(r>1, 0., np.exp(1./(np.pow(r,2)-1.)))

    # def rbf(self, r):
    #     r = r/self.alpha
    #     return np.maximum(1. - np.sign(r)*r, 0.)

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
    


rbf = CompactSupportPlateSpline(points, -val)
rbf.run()


def render_detail_field(contour_path, model, details, domain : M.geometry.AABB, device, res=1000, batch_size=1000):
    assert domain.dim == 2

    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1]/domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    dist_values = IL.utils.forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    detail_values = []
    for pt in tqdm(pts, total=pts.shape[0]):
        detail_values.append(details(pt))
    detail_values = np.array(detail_values)
    total_values = np.concatenate(dist_values) + detail_values

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


render_detail_field(os.path.join(OUTPUT_DIR, "contour_final.png"), NDF, rbf, PLOT_DOMAIN, DEVICE)