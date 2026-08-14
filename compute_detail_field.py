import os
import numpy as np
import argparse
import implicitlab as IL
import mouette as M

from src import CompactSupportRBFInterpolant, MetaData, load_model, NeuralSDFValues
from src import render_detail_field_2D, render_detail_field_3D, render_detail_field_2D_implicit, render_detail_field_3D_implicit

np.random.seed(42)

if __name__ == "__main__":

    ###### Parse commandline arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("folder", type=str, help="path to the folder. Should contain a metadata file and neural weights")
    argument_parser.add_argument("-nc", "--n-centers", type=int, default=100_000, help="Number of RBF centers")
    argument_parser.add_argument("-res", "--mc-resolution", type=int, default=300, help="Marching cube resolution.")
    argument_parser.add_argument("-s", "--support-size", type=float, default=1.1, help="support size sigma.")
    argument_parser.add_argument("-prune", type=float, default=0., help="pruning threshold for small RBF weights. If set to 0., nothing will be pruned.")
    argument_parser.add_argument("--rbf-shape-id", type=int, default=1, help="shape of the RBF function to consider. Default is 1 for (1-r)^2. See src/rbf.py to see all the shapes.")
    argument_parser.add_argument("--query-batch-size", type=int, default=50_000)
    argument_parser.add_argument("--implicit", action="store_true")
    args = argument_parser.parse_args()

    DEVICE = IL.utils.get_device()
    print("Neural model will be loaded on the following device:", DEVICE)

    metadata = MetaData.load_from_file(os.path.join(args.folder, "metadata.toml"))
    metadata.n_centers = args.n_centers
    metadata.sigma = args.support_size
    metadata.rbf_shape_id = args.rbf_shape_id
    metadata.implicit = args.implicit

    neural_model = load_model(metadata, args.folder, DEVICE, ignore_detail_field=True)
    geometry = IL.data.load_geometry(os.path.join(args.folder, "input_geometry.obj"))

    surface_sampler = IL.data.OnGeometryPointSampler( geometry, NeuralSDFValues(neural_model, DEVICE))
    points, val = surface_sampler.sample(args.n_centers)

    points_projected = IL.queries.project_onto_iso(points, neural_model, 0., DEVICE, batch_size=args.query_batch_size)
    pc_proj = M.mesh.from_arrays(points_projected)
    M.mesh.save(pc_proj, os.path.join(args.folder, "projected_centers.geogram_ascii"))
    # M.mesh.save(M.procedural.vector_field(points, points_projected-points), os.path.join(args.folder, "centers_to_projected.mesh"))
    
    
    if not metadata.implicit:
        distances_to_levelset = np.linalg.norm(points - points_projected, axis=1)
        support_size = args.support_size*np.max(distances_to_levelset)
        metadata.support_size = support_size
        rbf = CompactSupportRBFInterpolant(points, -val, alpha=support_size, rbf_shape=metadata.rbf_shape_id)
    else:
        displacement_vector = points_projected - points
        displacement_values =  np.linalg.norm(displacement_vector, axis=1)
        _, normals = IL.utils.forward_in_batches(neural_model, points, DEVICE, compute_grad=True, batch_size=args.query_batch_size, use_tqdm=True)
        normals /= np.linalg.norm(normals, axis=1)[:,np.newaxis]
        sign_values = -np.sign(np.sum(displacement_vector/displacement_values[:,np.newaxis] * normals, axis=1))
        displacement_values *= sign_values
        support_size = args.support_size*np.max(np.abs(displacement_values))
        metadata.support_size = support_size
        rbf = CompactSupportRBFInterpolant(points, -displacement_values, alpha=support_size, rbf_shape=metadata.rbf_shape_id)

    print("Compute RBF interpolation")
    print("Support size:", support_size)
    print("Number of basis functions:", args.n_centers)
    print("Max error on sampled points:", np.max(np.abs(val)))

    rbf.run()
    if args.prune>0.:
        rbf.prune(args.prune)

    rbf.save_to_file(os.path.join(args.folder, "rbf.pt"))
    pc_init = M.mesh.from_arrays(rbf.points.numpy())
    pc_init.vertices.register_array_as_attribute("neural_dist_field", rbf.values)
    pc_init.vertices.register_array_as_attribute("weights", rbf.weights.detach().cpu().numpy())
    M.mesh.save(pc_init, os.path.join(args.folder, "rbf_centers.geogram_ascii"))

    if metadata.geometry_dim == 2:
        plot_domain =  M.geometry.AABB([-1.5]*2, [1.5]*2)
        if metadata.implicit:
            render_detail_field_2D_implicit(
                os.path.join(args.folder, "contours_with_details.png"), 
                os.path.join(args.folder, "detail_field.png"), 
                os.path.join(args.folder, "RBF_support.png"), 
                neural_model, rbf, plot_domain, DEVICE, res=1000, batch_size=args.query_batch_size)
        else:
            render_detail_field_2D(
                os.path.join(args.folder, "contours_with_details.png"), 
                os.path.join(args.folder, "detail_field.png"), 
                os.path.join(args.folder, "RBF_support.png"), 
                neural_model, rbf, plot_domain, DEVICE, res=1000, batch_size=args.query_batch_size)
            
    elif metadata.geometry_dim == 3:
        plot_domain =  M.geometry.AABB.of_mesh(geometry).pad(0.1)
        if metadata.implicit:
            render_detail_field_3D_implicit(
                os.path.join(args.folder, "surface_with_details_implicit.obj"),
                neural_model, rbf, plot_domain, DEVICE, res=args.mc_resolution, batch_size=args.query_batch_size, 
                ignore_detail_threshold=support_size)
        else:
            render_detail_field_3D(
                os.path.join(args.folder, "surface_with_details.obj"),
                os.path.join(args.folder, "RBF_support.obj"),
                neural_model, rbf, plot_domain, DEVICE, res=args.mc_resolution, batch_size=args.query_batch_size,
                ignore_detail_threshold = support_size)
    metadata.has_detail_field = True
    metadata.save_to_file(os.path.join(args.folder, "metadata.toml"))

