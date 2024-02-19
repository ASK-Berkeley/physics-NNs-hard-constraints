import yaml
from .navier_stokes import Navier_Stokes_2D_Dataloaders
from .diffusion_sorption import Diffusion_Sorption1d_Dataloaders
from .data_utils import DatasetConfig


def get_numerical_solver_data(cfg: DatasetConfig):
    if cfg.name == 'navier-stokes-2d':
        return Navier_Stokes_2D_Dataloaders(cfg)
    elif cfg.name == 'diffusion-sorption-1d':
        return Diffusion_Sorption1d_Dataloaders(cfg)