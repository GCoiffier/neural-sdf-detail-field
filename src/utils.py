import os
import numpy as np
import mouette as M
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

import implicitlab as IL
from implicitlab.training import Callback


def sample_circle(center: np.ndarray, radius: float, n_points: int) -> np.ndarray:
    """Samples n points from a uniform distribution over a circle of center `center` and radius `radius`

    Args:
        center (np.ndarray): position of the circle's center.
        radius (float): radius of the circle
        n_points (int): number of points to sample

    Returns:
        np.ndarray: (n_points, 2)-shaped array of sampled points
    """
    theta = 2*np.pi*np.random.random(size=n_points)
    points = np.asarray(center) + radius*np.array([[np.cos(t), np.sin(t)] for t in theta])
    return points


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

    
class SaveTrainingPointsCB(Callback):
    def __init__(self, save_folder: str, freq: int):
        """A Callback responsible for saving the current point cloud dataset into a .xyz file

        Args:
            save_folder (str): folder into which the point will be saved. The format will be point_e<epoch>_<on/out>.xyz
            freq (int): frequency of the saving.
        """
        self.save_folder: str = save_folder
        self.freq = freq

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch>0 and epoch%self.freq==0:
            points_out = M.mesh.from_arrays(trainer.points["out"])
            M.mesh.save(points_out, os.path.join(self.save_folder, f"points_e{epoch}_out.xyz"))

            if trainer.resample_surface:
                # If no resampling is performed, there is no need to save the surface points
                points_on = M.mesh.from_arrays(trainer.points["on"])
                M.mesh.save(points_on, os.path.join(self.save_folder, f"points_e{epoch}_on.xyz"))
