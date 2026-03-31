import os
import numpy as np
import argparse
import implicitlab as IL
import mouette as M

from src import CompactSupportRBFInterpolant, CompactSupportRBFInterpolantTorch, AdaptativeSupportRBFInterpolant, MetaData, load_model, ImplicitRepresentation
from src.utils import NeuralSDFValues

from tqdm import tqdm
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from matplotlib import colors

from scipy.spatial import KDTree

def render_detail_field_2D(contour_path, detail_path, support_path, model, details, domain : M.geometry.AABB, device, res=1000, batch_size=10_000):
    assert domain.dim == 2

    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1]/domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    dist_values = IL.utils.forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size, use_tqdm=True)
    detail_values = IL.utils.forward_in_batches(details, pts, "cpu", compute_grad=False, batch_size=1_000_000, use_tqdm=True)
    dist_values = np.squeeze(dist_values)
    detail_values = np.squeeze(detail_values)
    total_values = dist_values + detail_values

    if contour_path is not None:
        img = total_values.reshape((res,resY)).T
        img = img[::-1,:]
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.contour(img, levels=16, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if detail_path is not None:
        img = detail_values.reshape((res,resY)).T
        img = img[::-1,:]
        vmax = max(np.abs(np.amin(img)), np.abs(np.amax(img)))
        vmin = -vmax
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        plt.contour(img, levels=8, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(detail_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if support_path is not None:
        img = detail_values.reshape((res,resY)).T
        img = img[::-1,:]
        img = np.abs(img)>1e-10
        plt.clf()
        plt.imshow(img, cmap="Oranges")
        plt.axis("off")
        plt.savefig(support_path, bbox_inches='tight', pad_inches=0, dpi=200)
        

def render_detail_field_3D(iso_path, support_path, model, details, domain : M.geometry.AABB, device, res=300, batch_size=10_000, ignore_detail_threshold:float = 10.):
    print("Render final surface")
    assert domain.dim == 3

    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    dist_values = IL.utils.forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size, use_tqdm=True)
    dist_values = np.squeeze(dist_values)

    low_dist = np.abs(dist_values)<ignore_detail_threshold
    pts_low_dist = pts[low_dist,:]
    n_detail = pts_low_dist.shape[0]
    n_total = pts.shape[0]
    print(f"Detail needed for {n_detail}/{n_total} points ({100*n_detail/n_total:.1f}%)")
    detail_values_low_dist = IL.utils.forward_in_batches(details, pts_low_dist, "cpu", compute_grad=False, batch_size=1_000_000, use_tqdm=True)
    
    detail_values = np.zeros_like(dist_values)
    detail_values[low_dist] = detail_values_low_dist
    
    total_values = dist_values + detail_values


    if support_path is not None: 
        support_values = np.abs(detail_values)>1e-10
        support_values = support_values.reshape((res,res,res))

        ### Call marching cubes
        verts,faces,normals,values = marching_cubes(support_values, level=0.5)
        values = values[:, np.newaxis]
        mesh = M.mesh.RawMeshData()
        mesh.vertices += list(verts)
        mesh.faces += list(faces)
        mesh = M.mesh.SurfaceMesh(mesh)
        normal_attr = mesh.vertices.create_attribute("normals", float, 3, dense=True)
        normal_attr._data = normals
        values_attr = mesh.vertices.create_attribute("values", float, 1, dense=True)
        values_attr._data = values
        
        ### Reproject meshes to correct coordinates
        for v in mesh.id_vertices:
            pV = M.Vec(mesh.vertices[v])
            ix, iy, iz = int(pV.x), int(pV.y), int(pV.z)
            dx, dy, dz = pV.x%1, pV.y%1, pV.z%1

            ixn = ix+1 if ix<res-1 else res-1
            iyn = iy+1 if iy<res-1 else res-1
            izn = iz+1 if iz<res-1 else res-1

            vx = (1-dx)*L[0][ix] + dx * L[0][ixn]
            vy = (1-dy)*L[1][iy] + dy * L[1][iyn]
            vz = (1-dz)*L[2][iz] + dz * L[2][izn]
            mesh.vertices[v] = M.Vec(vx,vy,vz)
        M.mesh.save(mesh, support_path)
        del support_values
    
    if iso_path is not None: 
        total_values = total_values.reshape((res,res,res))
        ### Call marching cubes
        verts,faces,normals,values = marching_cubes(total_values, level=0.)
        values = values[:, np.newaxis]
        mesh = M.mesh.RawMeshData()
        mesh.vertices += list(verts)
        mesh.faces += list(faces)
        mesh = M.mesh.SurfaceMesh(mesh)
        normal_attr = mesh.vertices.create_attribute("normals", float, 3, dense=True)
        normal_attr._data = normals
        values_attr = mesh.vertices.create_attribute("values", float, 1, dense=True)
        values_attr._data = values
        
        ### Reproject meshes to correct coordinates
        for v in mesh.id_vertices:
            pV = M.Vec(mesh.vertices[v])
            ix, iy, iz = int(pV.x), int(pV.y), int(pV.z)
            dx, dy, dz = pV.x%1, pV.y%1, pV.z%1

            ixn = ix+1 if ix<res-1 else res-1
            iyn = iy+1 if iy<res-1 else res-1
            izn = iz+1 if iz<res-1 else res-1

            vx = (1-dx)*L[0][ix] + dx * L[0][ixn]
            vy = (1-dy)*L[1][iy] + dy * L[1][iyn]
            vz = (1-dz)*L[2][iz] + dz * L[2][izn]
            mesh.vertices[v] = M.Vec(vx,vy,vz)
        M.mesh.save(mesh, iso_path)
    

# def render_rbf_support_2D(file_path, rbf, domain, res=1000):
#     print("Render RBF support")
    
#     X = np.linspace(domain.mini[0], domain.maxi[0], res)
#     resY = round(res * domain.span[1]/domain.span[0])
#     Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

#     pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
#     kd_tree = KDTree(pts)

#     near = kd_tree.query_ball_tree(rbf.tree, rbf.alpha)
#     occupancy = np.array([len(near_i)>0 for near_i in near]).reshape((res,resY)).T
#     occupancy = occupancy[::-1,:]
#     plt.clf()
#     plt.imshow(occupancy, cmap="bwr")
#     plt.axis("off")
#     plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=200)


# def render_rbf_support_3D(file_path, rbf, domain : M.geometry.AABB, res=300):
#     print("Render RBF support")
#     L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
#     pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
#     kd_tree = KDTree(pts)

#     near = kd_tree.query_ball_tree(rbf.tree, rbf.alpha)
#     occupancy = np.array([len(near_i)>0 for near_i in near]).reshape((res,res,res))
    
#     ### Call marching cubes
#     verts,faces,normals,values = marching_cubes(occupancy, level=0)
#     values = values[:, np.newaxis]
#     mesh = M.mesh.RawMeshData()
#     mesh.vertices += list(verts)
#     mesh.faces += list(faces)
#     mesh = M.mesh.SurfaceMesh(mesh)
#     normal_attr = mesh.vertices.create_attribute("normals", float, 3, dense=True)
#     normal_attr._data = normals
#     values_attr = mesh.vertices.create_attribute("values", float, 1, dense=True)
#     values_attr._data = values
    
#     ### Reproject meshes to correct coordinates
#     for v in mesh.id_vertices:
#         pV = M.Vec(mesh.vertices[v])
#         ix, iy, iz = int(pV.x), int(pV.y), int(pV.z)
#         dx, dy, dz = pV.x%1, pV.y%1, pV.z%1

#         ixn = ix+1 if ix<res-1 else res-1
#         iyn = iy+1 if iy<res-1 else res-1
#         izn = iz+1 if iz<res-1 else res-1

#         vx = (1-dx)*L[0][ix] + dx * L[0][ixn]
#         vy = (1-dy)*L[1][iy] + dy * L[1][iyn]
#         vz = (1-dz)*L[2][iz] + dz * L[2][izn]
#         mesh.vertices[v] = M.Vec(vx,vy,vz)
#     M.mesh.save(mesh, file_path)

if __name__ == "__main__":

    ###### Parse commandline arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("folder", type=str, help="path to the folder. Should contain a metadata file and neural weights")
    argument_parser.add_argument("-np", "--n-points", type=int, default=3_000)
    argument_parser.add_argument("--no-gradient-correction", action="store_true")
    argument_parser.add_argument("-a", "--adaptative-support", action="store_true")
    argument_parser.add_argument("-res", "--mc-resolution", type=int, default=300)
    argument_parser.add_argument("-s", "--support-size", type=float, default=1.1)
    argument_parser.add_argument("-prune", action="store_true")
    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()
    print("Neural model will be loaded on the following device:", DEVICE)

    metadata = MetaData.load_from_file(os.path.join(args.folder, "metadata.toml"))
    metadata.n_primitives = args.n_points
    metadata.adaptative_support = False

    neural_model = load_model(args.folder, DEVICE, ignore_grad_correct=args.no_gradient_correction, ignore_detail_field=True)
    geometry = IL.data.load_geometry(os.path.join(args.folder, "input_geometry.obj"))

    surface_sampler = IL.data.OnGeometryPointSampler( geometry, NeuralSDFValues(neural_model, DEVICE))
    points, val = surface_sampler.sample(args.n_points)
    # take three time the number of points and only consider the points with largest absolute values
    # points, val = surface_sampler.sample(3*args.n_points)
    # val = np.squeeze(val)
    # points = points[np.abs(val).argsort(), :]
    # val.sort()
    # points = points[-args.n_points:, :]
    # val = val[-args.n_points:]

    points_projected = IL.queries.project_onto_iso(points, neural_model, 0., DEVICE, batch_size=10_000)
    pc_proj = M.mesh.from_arrays(points_projected)
    M.mesh.save(pc_proj,os.path.join(args.folder, "projected_centers.geogram_ascii"))
    M.mesh.save(M.procedural.vector_field(points, points_projected-points), os.path.join(args.folder, "dis.mesh"))
    distances_to_levelset = np.linalg.norm(points - points_projected, axis=1)

    print("Compute RBF interpolation")
    if args.adaptative_support:
        tree = KDTree(points)
        KNN = tree.query(points, 10)[0]
        distances_to_closest_RBF = KNN[:,-1]
        sizes = np.maximum(1.1*distances_to_levelset, distances_to_closest_RBF)
        rbf = AdaptativeSupportRBFInterpolant(points, -val, sizes, tree=tree)
        metadata.adaptative_support = True
        metadata.support_size = np.amax(sizes)
        print("Support size range:", "[", np.amin(sizes), ",", np.amax(sizes), "]")
    else:
        support_size = args.support_size*np.max(distances_to_levelset)
        # tree = KDTree(points)
        # KNN = tree.query(points, 10)[0]
        # distances_to_closest_RBF = KNN[:,-1]
        # print("Max distance to 10 closest RBF", np.amax(distances_to_closest_RBF))

        rbf = CompactSupportRBFInterpolantTorch(points, -val, alpha=support_size)
        metadata.adaptative_support = False
        metadata.support_size = support_size
        print("Support size:", support_size)
        
    print("Number of basis functions:", args.n_points)
    print("Max error on sampled points:", np.max(np.abs(val)))


    rbf.run()
    if args.prune:
        rbf.prune(5e-4)

    rbf.save_to_file(os.path.join(args.folder, "rbf.pt"))
    pc_init = M.mesh.from_arrays(rbf.points.numpy())
    pc_init.vertices.register_array_as_attribute("ndf", rbf.values)
    pc_init.vertices.register_array_as_attribute("weights", rbf.weights.detach().cpu().numpy())
    M.mesh.save(pc_init, os.path.join(args.folder, "rbf_centers.geogram_ascii"))

    if metadata.geometry_dim == 2:
        plot_domain =  M.geometry.AABB([-1.5]*2, [1.5]*2)
        render_detail_field_2D(
            os.path.join(args.folder, "contours_with_details.png"), 
            os.path.join(args.folder, "detail_field.png"), 
            os.path.join(args.folder, "RBF_support.png"), 
            neural_model, rbf, plot_domain, DEVICE, res=1000, batch_size=5000)
        # render_rbf_support_2D(os.path.join(args.folder, "rbf_support.png"), rbf, plot_domain, res=1000)
    elif metadata.geometry_dim == 3:
        plot_domain =  M.geometry.AABB.of_mesh(geometry).pad(0.1)
        render_detail_field_3D(
            os.path.join(args.folder, "surface_with_details.obj"),
            os.path.join(args.folder, "RBF_support.obj"),
            neural_model, rbf, plot_domain, DEVICE, res=args.mc_resolution, batch_size=10_000, 
            ignore_detail_threshold = support_size)
        # render_rbf_support_3D(os.path.join(args.folder, "RBF_support.obj"), rbf, plot_domain.pad(0.1), res=200)
    metadata.detail_field_computed = True
    metadata.save_to_file(os.path.join(args.folder, "metadata.toml"))

