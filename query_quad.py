import argparse
import os
import torch

import numpy as np
import mouette as M
import implicitlab as IL
from src import MetaData, load_model
import matplotlib.pyplot as plt
from matplotlib import colors


def render_sdf_quad(contour_path, gradient_path, no_detail_path, support_path, model, P0, P1, P2, device, res=800, batch_size=5000, **kwargs):    
    dx = P1 - P0
    dy = P2 - P0
    X = np.linspace(0,1, res)
    resY = round(res * M.geometry.norm(dy)/M.geometry.norm(dx))
    Y = np.linspace(0,1, resY)

    pts = []
    for ax in X:
        for ay in Y:
            p = P0 + ax*dx + ay*dy
            pts.append(p)
    pts = np.array(pts)    
    if gradient_path is not None:
        dist_values, grad_dist_values = IL.utils.forward_in_batches( model.neural_model, pts, device, compute_grad=True, batch_size=batch_size, use_tqdm=True)
        detail_values, grad_detail_values = IL.utils.forward_in_batches(model.detail_model, pts, "cpu", compute_grad=True, batch_size=100_000, use_tqdm=True)
    else:
        dist_values = IL.utils.forward_in_batches(model.neural_model, pts, device, compute_grad=False, batch_size=batch_size, use_tqdm=True)
        detail_values = IL.utils.forward_in_batches(model.detail_model, pts, "cpu", compute_grad=False, batch_size=1_000_000, use_tqdm=True)
    
    dist_values = np.squeeze(dist_values)
    detail_values = np.squeeze(detail_values)
    total_values = dist_values + detail_values

    if no_detail_path is not None:
        img = dist_values.reshape((res,resY)).T
        img = img[::-1,:]
        vmin = np.amin(img)
        vmax = np.amax(img)
        if vmin>0 or vmax<0:
            vmin,vmax = -1, 1
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.contour(img, levels=kwargs.get("n_levels", 16), colors='k', linestyles="solid", linewidths=0.5)
        plt.contour(img, levels=[kwargs.get("zero_contour_offset", 0.)], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(no_detail_path, bbox_inches='tight', pad_inches=0, dpi=kwargs.get("dpi", 500))

    if support_path is not None:
        img = detail_values.reshape((res,resY)).T
        img = img[::-1,:]
        img = np.abs(img)>1e-10
        plt.clf()
        plt.imshow(img, cmap="bwr")
        plt.axis("off")
        plt.savefig(support_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if contour_path is not None:
        img = total_values.reshape((res,resY)).T
        img = img[::-1,:]
        vmin = np.amin(img)
        vmax = np.amax(img)
        if vmin>0 or vmax<0:
            vmin,vmax = -1, 1
        print(vmin, vmax)
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.contour(img, levels=kwargs.get("n_levels", 16), colors='k', linestyles="solid", linewidths=0.5)
        plt.contour(img, levels=[kwargs.get("zero_contour_offset", 0.)], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=kwargs.get("dpi", 500))

    if gradient_path is not None:
        grad_values = grad_dist_values + grad_detail_values
        grad_norms = np.linalg.norm(grad_values,axis=1)
        grad_img = grad_norms.reshape((res,resY)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr")
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )

    parser.add_argument("folder_path", type=str, help="path to the model '.pt' file")
    parser.add_argument("quad",  type=str, help="path to the quad")
    parser.add_argument("-p", "--path", type=str, default="", help="path_to_output")
    parser.add_argument("-res", "--resolution", type=int, default=1000, help="grid resolution to consider")
    parser.add_argument("-cpu", action="store_true", help="force CPU computation")
    parser.add_argument("-bs", "--batch-size", type=int, default=10_000, help="batch size")
    parser.add_argument("-i", "--n-iso", type=int, default=14, help="number of iso contours")
    args = parser.parse_args()

    device = IL.utils.get_device(args.cpu)
    print("DEVICE:", device)

    sdf = load_model(args.folder_path, device)
    quad = M.mesh.load(args.quad)
    assert len(quad.vertices)==4
    uvs = quad.vertices.create_attribute("uv_coords", float, 2) 
    uvs = M.attributes.average_corners_to_vertices(quad, quad.face_corners.get_attribute("uv_coords"), uvs)

    iP0 = np.argmin([M.geometry.distance(M.Vec(0.,0.), uvs[i]) for i in quad.id_vertices])
    iP1 = np.argmin([M.geometry.distance(M.Vec(1.,0.), uvs[i]) for i in quad.id_vertices])
    iP2 = np.argmin([M.geometry.distance(M.Vec(0.,1.), uvs[i]) for i in quad.id_vertices])
    assert iP0 != iP1 and iP0 != iP2 and iP1 != iP2

    contour_path = os.path.join(args.path, "contours.png")
    gradient_path = os.path.join(args.path, "gradient.png")
    no_detail_path = os.path.join(args.path, "no_details.png")
    support_path = os.path.join(args.path, "rbf_support.png")

    render_sdf_quad(contour_path, gradient_path, no_detail_path, support_path, 
        sdf, quad.vertices[iP0], quad.vertices[iP1], quad.vertices[iP2], 
        device, res=args.resolution, batch_size=args.batch_size, n_levels=args.n_iso, dpi=500, zero_contour_offset=0.)