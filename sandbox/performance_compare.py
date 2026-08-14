import numpy as np
from scipy.spatial import KDTree
import pynanoflann as flann
from time import time

np.random.seed(42)

N_POINTS = 100_000
N_QUERIES = 400*400*400
points = np.random.random((N_POINTS, 3))
queries = np.random.random((N_QUERIES, 3))

radius = 5e-3

print("Scipy query_ball_point (1 worker)")
t0 = time()
tree_seed = KDTree(points)
t1 = time()
neigh = tree_seed.query_ball_point(queries, radius, workers=1)
print("With construction:", time() - t0)
print("Query only:", time() - t1)
print("")


print("Scipy query_ball_point (16 workers)")
t0 = time()
tree_seed = KDTree(points)
t1 = time()
neigh = tree_seed.query_ball_point(queries, radius, workers=16)
print("With construction:", time() - t0)
print("Query only:", time() - t1)
print("")


print("Scipy query_ball_tree (point over query)")
t0 = time()
tree_seed = KDTree(points)
t1 = time()
tree_queries = KDTree(queries, compact_nodes=False)
q = tree_seed.query_ball_tree(tree_queries, radius)
print("With construction:", time() - t0)
print("Query only:", time() - t1)
print("")


# print("Scipy query_ball_tree (query over point)")
# t0 = time()
# tree_seed = KDTree(points)
# t1 = time()
# tree_queries = KDTree(queries)
# q = tree_queries.query_ball_tree(tree_seed, radius)
# print("With construction:", time() - t0)
# print("Query only:", time() - t1)
# print("")


print("pynanoflann")
t0 = time()
tree_seed = flann.KDTree(radius=radius)
tree_seed.fit(points)
t1 = time()
q = tree_seed.radius_neighbors(queries, return_distance=False, n_jobs=8)
print("With construction:", time() - t0)
print("Query only:", time() - t1)
print("")