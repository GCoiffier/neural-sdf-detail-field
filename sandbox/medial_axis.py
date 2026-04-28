import mouette as M
import triangle
import numpy as np
import sys
import implicitlab as IL
import polyscope as ps


geom = IL.load_geometry(sys.argv[1])
assert geom.dim == 2

points = IL.data.OnGeometryPointSampler(geom).sample(1_000)

vor_pts, vor_edges, _, _ = triangle.voronoi(points)

polyline = M.mesh.RawMeshData()
n_edge = 0
for (a,b) in vor_edges:
    if M.geometry.norm(vor_pts[a])<2 and M.geometry.norm(vor_pts[b])<2:
        polyline.vertices += (vor_pts[a], vor_pts[b])
        polyline.edges.append((2*n_edge, 2*n_edge+1))
        n_edge += 1
polyline = M.mesh.PolyLine(polyline)

ps.init()
ps.set_navigation_style("planar")
ps.register_point_cloud("points", points)
vor_pts_near = []
for a in vor_pts:
    if M.geometry.norm(a)<2:
        vor_pts_near.append(a)
vor_pts_near = np.array(vor_pts_near)
ps.register_point_cloud("near_vor_pts", vor_pts_near)
ps.register_curve_network("voronoi", np.asarray(polyline.vertices), np.asarray(polyline.edges))

ps.show()