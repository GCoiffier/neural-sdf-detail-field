import numpy as np
import implicitlab as IL
import mouette as M
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from matplotlib import colors


def render_detail_field_2D(
        contour_path: str, 
        detail_path: str, 
        support_path: str, 
        neural_model, detail_model, 
        domain : M.geometry.AABB, device, res: int = 1000, 
        batch_size: int = 10_000
    ):
    assert domain.dim == 2

    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1]/domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    dist_values = IL.utils.forward_in_batches(neural_model, pts, device, compute_grad=False, batch_size=batch_size, use_tqdm=True)
    detail_values = IL.utils.forward_in_batches(detail_model, pts, "cpu", compute_grad=False, batch_size=batch_size, use_tqdm=True)
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
        

def render_detail_field_3D(
        iso_path: str, support_path: str, 
        neural_model, details, 
        domain : M.geometry.AABB, device: str, res: int, 
        batch_size: int = 10_000, 
        ignore_detail_threshold: float = 10.
    ):
    print("Extract final surface with marching cubes")
    assert domain.dim == 3

    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    print("Compute neural values on the grid")
    dist_values = IL.utils.forward_in_batches(neural_model, pts, device, compute_grad=False, batch_size=batch_size, use_tqdm=True)
    dist_values = np.squeeze(dist_values)

    low_dist = np.abs(dist_values)<ignore_detail_threshold
    pts_low_dist = pts[low_dist,:]
    n_detail = pts_low_dist.shape[0]
    n_total = pts.shape[0]
    print(f"Detail needed for {n_detail}/{n_total} points ({100*n_detail/n_total:.1f}%)")
    print("Compute detail values where needed")
    detail_values_low_dist = IL.utils.forward_in_batches(details, pts_low_dist, "cpu", compute_grad=False, batch_size=batch_size, use_tqdm=True)
    
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
    

###########################################################################################################################################################

def render_detail_field_2D_implicit(contour_path, detail_path, support_path, model, details, domain : M.geometry.AABB, device, res=1000, batch_size=10_000):
    assert domain.dim == 2

    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1]/domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    dist_values = IL.utils.forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    detail_values = details(pts)
    total_values = np.concatenate(dist_values) + detail_values

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
        plt.imshow(img, cmap="bwr")
        plt.axis("off")
        plt.savefig(support_path, bbox_inches='tight', pad_inches=0, dpi=200)
        

def render_detail_field_3D_implicit(iso_path, model, displacement_field, domain : M.geometry.AABB, device, res=300, batch_size=10_000, ignore_detail_threshold:float = 10.):
    assert domain.dim == 3

    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    dist_values, grad0 = IL.utils.forward_in_batches(model, pts, device, compute_grad=True, batch_size=batch_size, use_tqdm=True)

    dist_values = np.squeeze(dist_values)
    low_dist = np.abs(dist_values)<ignore_detail_threshold

    pts_low_dist = pts[low_dist,:]
    n_detail = pts_low_dist.shape[0]
    n_total = pts.shape[0]
    print(f"Detail needed for {n_detail}/{n_total} points ({100*n_detail/n_total:.1f}%)")

    displacement_values = IL.utils.forward_in_batches(displacement_field, pts_low_dist, "cpu", compute_grad=False, batch_size=200_000, use_tqdm=True)
    pts_disp = pts_low_dist + grad0[low_dist,:]*displacement_values[:, np.newaxis]/np.linalg.norm(grad0[low_dist,:], axis=1)[:,np.newaxis]
    dist_values_with_disp = IL.utils.forward_in_batches(model, pts_disp, device, compute_grad=False, batch_size=batch_size, use_tqdm=True)
    dist_values_with_disp = np.squeeze(dist_values_with_disp)
    
    dist_values[low_dist] = dist_values_with_disp
    dist_values = dist_values.reshape((res,res,res))
    ### Call marching cubes
    verts,faces,normals,values = marching_cubes(dist_values, level=0.)
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