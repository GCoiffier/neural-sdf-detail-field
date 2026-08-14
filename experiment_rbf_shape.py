import os
import numpy as np
import argparse
import implicitlab as IL
import mouette as M

from src import CompactSupportRBFInterpolant, MetaData, load_model
from src.utils import NeuralSDFValues

from skimage.measure import marching_cubes
    
if __name__ == "__main__":

    ###### Parse commandline arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("folder", type=str, help="path to the folder. Should contain a metadata file and neural weights")
    argument_parser.add_argument("-np", "--n-points", type=int, default=50_000)
    argument_parser.add_argument("-res", "--mc-resolution", type=int, default=400)
    argument_parser.add_argument("-prune", type=float, default=0.)
    args = argument_parser.parse_args()
    DEVICE = IL.utils.get_device()
    print("Neural model will be loaded on the following device:", DEVICE)

    metadata = MetaData.load_from_file(os.path.join(args.folder, "metadata.toml"))
    metadata.n_primitives = args.n_points
    metadata.adaptative_support = False

    neural_model = load_model(args.folder, DEVICE, ignore_grad_correct=True, ignore_detail_field=True)
    geometry = IL.data.load_geometry(os.path.join(args.folder, "input_geometry.obj"))

    surface_sampler = IL.data.OnGeometryPointSampler( geometry, NeuralSDFValues(neural_model, DEVICE))
    points, val = surface_sampler.sample(args.n_points)

    pc_init = M.mesh.from_arrays(points)
    pc_init.vertices.register_array_as_attribute("ndf", val)
    M.mesh.save(pc_init, os.path.join(args.folder, "rbf_centers.geogram_ascii"))

    points_projected = IL.queries.project_onto_iso(points, neural_model, 0., DEVICE, batch_size=10_000)
    distances_to_levelset = np.linalg.norm(points - points_projected, axis=1)
    support_size = 1.1*np.max(distances_to_levelset)

    domain =  M.geometry.AABB.of_mesh(geometry).pad(0.1)
    res = args.mc_resolution
    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    print("Compute neural output on the grid")
    dist_values = IL.utils.forward_in_batches(neural_model, pts, DEVICE, compute_grad=False, batch_size=10_000, use_tqdm=True)
    dist_values = np.squeeze(dist_values)
    low_dist = np.abs(dist_values)<0.1
    pts_low_dist = pts[low_dist,:]
    n_detail = pts_low_dist.shape[0]
    n_total = pts.shape[0]
    print(f"Detail needed for {n_detail}/{n_total} points ({100*n_detail/n_total:.1f}%)")


    print("Support size:", support_size)
    print("Number of basis functions:", args.n_points)
    print("Max error on sampled points:", np.max(np.abs(val)))
    for i_shape in range(1,6):
        rbf = CompactSupportRBFInterpolant(points, -val, alpha=support_size, rbf_shape=i_shape)
        rbf.run()
        if args.prune>0.:
            rbf.prune(args.prune)

        assert metadata.geometry_dim == 3      
        print("Compute detail output on the grid")
        detail_values_low_dist = IL.utils.forward_in_batches(rbf, pts_low_dist, "cpu", compute_grad=False, batch_size=1_000_000, use_tqdm=True)
    
        detail_values = np.zeros_like(dist_values)
        detail_values[low_dist] = detail_values_low_dist
        
        total_values = dist_values + detail_values
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
        M.mesh.save(mesh, os.path.join(args.folder, f"RBF_support_{i_shape}.obj"))
        del support_values
        
       
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
        M.mesh.save(mesh, os.path.join(args.folder, f"surface_with_details_{i_shape}.obj"))


