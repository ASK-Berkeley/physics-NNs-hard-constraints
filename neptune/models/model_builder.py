import yaml
from typing import Dict
from jaxtyping import PRNGKeyArray

from .sequential import SequentialModel
from .model_utils import ModelConfig
from .fno2d import FNO2D, FNO2DConfig
from .fno3d import FNO3D, FNO3DConfig
from .constraints import HardConstraintLayer, ConstraintConfig
from ..types import ModelInput
from ..geometry import Function
from ..pdes import AbstractPDE
from .MoE import MoEConfig


def _recurse_dict(d: dict):
    for k in d.keys():
        if isinstance(d[k], dict):
            d[k] = _recurse_dict(d[k])
        elif isinstance(d[k], list):
            d[k] = tuple(d[k])
    return d


def get_model_config(path: str, model_name: str = None):
    with open(path, 'r') as fp:
        cfg_dict: dict = yaml.safe_load(fp)
    for k in cfg_dict.keys():
        if isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
        elif isinstance(cfg_dict[k], dict):
            cfg_dict[k] = _recurse_dict(cfg_dict[k])
    if model_name is None:
        if 'model' not in cfg_dict.keys():
            raise ModuleNotFoundError()
        model_name = cfg_dict['model']
    return dict_to_model_config(cfg_dict)


def dict_to_model_config(cfg: Dict):
    if 'model' not in cfg.keys():
        raise ModuleNotFoundError()
    model_name = cfg['model']
    if 'constraint' in cfg.keys():
        cfg['constraint'] = ConstraintConfig(**cfg['constraint'])
    if 'moe_config' in cfg.keys():
        cfg['moe_config'] = MoEConfig(**cfg['moe_config'])
        if cfg['moe_config'].split == 'none':
            cfg['moe_config'] = None
    if model_name == 'fno2d':
        return FNO2DConfig(**cfg)
    elif model_name == 'fno3d':
        return FNO3DConfig(**cfg)
    raise ModuleNotFoundError()


def get_unconstrained_model(cfg: ModelConfig):
    if cfg.model == 'fno2d':
        return FNO2D
    elif cfg.model == 'fno3d':
        return FNO3D
    raise NotImplementedError()


def get_constraint(cfg: ModelConfig, pde: AbstractPDE):
    return HardConstraintLayer(cfg.constraint, pde)


def get_model(cfg: ModelConfig, pde, rng_key: PRNGKeyArray, model_input: ModelInput):
    model_cls = get_unconstrained_model(cfg)
    processed_input = SequentialModel(
        None, None, None, pde).prepare_input(model_input)
    model = model_cls(cfg, rng_key, processed_input)
    constraint = None
    MoE_config = cfg.moe_config
    if cfg.constraint is not None and cfg.constraint.system != 'none' and cfg.constraint.system != 'soft':
        constraint = get_constraint(cfg, pde)
    return SequentialModel(model, constraint, MoE_config, pde)
