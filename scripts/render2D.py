import os
import sys
sys.path.append(os.getcwd())

import implicitlab as IL
import mouette as M
from src.io import load_model

DEVICE = IL.utils.get_device()
domain = M.geometry.AABB([-1.2]*2, [1.2]*2)
model = load_model(sys.argv[1], DEVICE, ignore_detail_field=True)
IL.visualize.render_sdf_2d(None, "contour.png", "gradient.png", model, domain, DEVICE, n_contours=32, res=1500)

model = load_model(sys.argv[1], DEVICE)
IL.visualize.render_sdf_2d(None, "contour_details.png", "gradient_details.png", model, domain, DEVICE, n_contours=32, res=1500)

    