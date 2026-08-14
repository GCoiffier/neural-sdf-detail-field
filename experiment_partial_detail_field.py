import os
import numpy as np
import argparse
import implicitlab as IL
import mouette as M

from src import CompactSupportRBFInterpolant, MetaData, load_model
from src.utils import NeuralSDFValues

from skimage.measure import marching_cubes
from igl import fast_winding_number

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
    

if __name__ == "__main__":

    ###### Parse commandline arguments
    argument_parser = argparse.ArgumentParser()

    n_points = 10_000
    res = 400
    sigma = 6.
    folder_path = os.path.join("trained_models", "grayloc", "hkr")

    DEVICE = IL.utils.get_device()
    print("Neural model will be loaded on the following device:", DEVICE)

    metadata = MetaData.load_from_file(os.path.join(folder_path, "metadata.toml"))

    neural_model = load_model(folder_path, DEVICE, ignore_grad_correct=True, ignore_detail_field=True)
    geometry = IL.data.load_geometry(os.path.join(folder_path, "input_geometry.obj"))

    surface_sampler = IL.data.OnGeometryPointSampler( geometry, NeuralSDFValues(neural_model, DEVICE))
    points, val = surface_sampler.sample(10*n_points)

    allowed_volume = M.mesh.load(os.path.join("trained_models", "grayloc", "allowed_volume.obj"))
    winding = fast_winding_number(np.asarray(allowed_volume.vertices), np.asarray(allowed_volume.faces), points)

    points = points[winding>0.5]
    val = val[winding>0.5]

    points_projected = IL.queries.project_onto_iso(points, neural_model, 0., DEVICE, batch_size=10_000)
    distances_to_levelset = np.linalg.norm(points - points_projected, axis=1)

    print("Compute RBF interpolation")
    support_size = sigma*np.max(distances_to_levelset)
    rbf = CompactSupportRBFInterpolant(points, -val, alpha=support_size)
    metadata.adaptative_support = False
    metadata.support_size = support_size
    print("Support size:", support_size)
        
    print("Number of basis functions:", points.shape[0])
    print("Max error on sampled points:", np.max(np.abs(val)))


    rbf.run()
    rbf.save_to_file(os.path.join(folder_path, "rbf.pt"))
    pc_init = M.mesh.from_arrays(rbf.points.numpy())
    pc_init.vertices.register_array_as_attribute("ndf", rbf.values)
    pc_init.vertices.register_array_as_attribute("weights", rbf.weights.detach().cpu().numpy())
    M.mesh.save(pc_init, os.path.join(folder_path, "rbf_centers.geogram_ascii"))


    plot_domain =  M.geometry.AABB.of_mesh(geometry).pad(0.1)
    render_detail_field_3D(
        os.path.join(folder_path, "surface_with_details.obj"),
        os.path.join(folder_path, "RBF_support.obj"),
        neural_model, rbf, plot_domain, DEVICE, res=res, batch_size=10_000, 
        ignore_detail_threshold = 2*support_size)
    metadata.detail_field_computed = True
    metadata.save_to_file(os.path.join(folder_path, "metadata.toml"))

