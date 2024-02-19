import argparse

from .pdes import NavierStokes2D, DiffusionSorption1D
from .models import FNO3DConfig, MoEConfig, ConstraintConfig, FNO2DConfig
from .datasets import DatasetConfig
from .trainers.train_utils import OptimizerConfig, TrainingConfig


def get_parser():
    parser = argparse.ArgumentParser(description='Params')

    #### Model Specific ####
    parser.add_argument('-model', type=str, default='fno3d')
    parser.add_argument('-modes1', type=int, default=8)
    parser.add_argument('-modes2', type=int, default=8)
    parser.add_argument('-modes3', type=int, default=8)
    parser.add_argument('-layers', type=int, default=8)
    parser.add_argument('-fc_dim', type=int, default=128,)
    parser.add_argument('-layer_dim', type=int, default=64)
    # Number of basis functions
    parser.add_argument('-out_dim', type=int, default=1,)
    parser.add_argument('-activation', type=str, default='tanh',)
    parser.add_argument('-activate_last_layer', action='store_true',)
    parser.add_argument('-mollifier', type=float, default=None,)

    #### Constraint Specific ####
    parser.add_argument('-system', type=str, default='none', choices=[
                        'equalityqp', 'soft', 'none', 'levenbergmarquardt', 'bfgs', 'gaussnewton'])
    parser.add_argument('-num_sampled_points', type=int,
                        default=200, help='number of sampled points')
    parser.add_argument('-atol', type=float, default=1e-4,
                        help='Abs. tolerance for constraint solver')
    parser.add_argument('-rtol', type=float, default=1e-4,
                        help='Rel. tolerance for constraint solver')
    parser.add_argument('-maxiter', type=int, default=50,
                        help='maxiter for constraint solver')
    parser.add_argument('-refine_regularization', type=float, default=0.,
                        help='DEPRECATED: Regularization for refinement.')
    parser.add_argument('-refine_maxiter', type=int, default=5,
                        help='DEPRECATED: Maxiter for refinement.')
    parser.add_argument('-linear_solver', type=str, default='gmres',
                        help='DEPRECATED: Use system arg')
    parser.add_argument('-ridge', type=float, default=1e-4,
                        help='DEPRECATED: ridge regularization for constraint solver')
    parser.add_argument('-damping_parameter', type=float, default=1.,
                        help='damping parameter for levenberg marquardt')
    parser.add_argument('-reweight_fn', type=str, default='none',
                        help='reweighting function for constraint solver')
    parser.add_argument('-mask_boundary_conditions', action='store_true',
                        help='Whether to mask out the boundary conditions dict - uses the same mask as the residual terms')

    #### MoE Specific ####
    parser.add_argument('-split', type=str, default='none', choices=['none', 'spatialtemporal'],
                        help='Type of MoE split')
    parser.add_argument('-expert_t', type=int,
                        help='num experts in t', default=1)
    parser.add_argument('-expert_x', type=int,
                        help='num experts in x', default=1)
    parser.add_argument('-expert_y', type=int,
                        help='num experts in y', default=1)

    #### Training Specific ####
    parser.add_argument('-num_epochs', type=int, default=50)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-learning_rate', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('-seed', type=int, default=3)
    parser.add_argument('-no_checkpoints', action='store_true')
    parser.add_argument('-no_jit', action='store_true')
    parser.add_argument('-log_every_n_steps', type=int, default=100)
    parser.add_argument('-num_callback_points', type=int, default=64)
    parser.add_argument('-ic_loss_weight', type=float, default=0.)
    parser.add_argument('-bc_loss_weight', type=float, default=0.)
    parser.add_argument('-interior_loss_weight', type=float, default=0.)
    parser.add_argument('-pde_loss_weight', type=float, default=1.)
    parser.add_argument('-no_data_loss_normalize', action='store_true')
    parser.add_argument('-no_icbc_loss_normalize', action='store_true')

    #### Dataset Specific ####
    parser.add_argument('-dataset', type=str, default='navier-stokes-2d')
    parser.add_argument('-data_root', type=str,
                        default='')
    parser.add_argument('-batch_size', type=int, default=2)

    #### W&B Args ####
    parser.add_argument('-use_tqdm', action='store_true',
                        help='Whether to use tqdm or not.')

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    model_cfg = None

    constraint_cfg = ConstraintConfig(
        system=args.system,
        num_sampled_points=args.num_sampled_points,
        atol=args.atol,
        rtol=args.rtol,
        maxiter=args.maxiter,
        refine_regularization=args.refine_regularization,
        refine_maxiter=args.refine_maxiter,
        linear_solver=args.linear_solver,
        ridge=args.ridge,
        damping_parameter=args.damping_parameter,
        mask_boundary_conditions=args.mask_boundary_conditions,
    )

    if args.model.lower() == 'fno3d':
        model_cfg = FNO3DConfig(
            model=args.model,
            modes1=tuple(args.modes1 for _ in range(args.layers)),
            modes2=tuple(args.modes2 for _ in range(args.layers)),
            modes3=tuple(args.modes3 for _ in range(args.layers)),
            layers=tuple(args.layer_dim for _ in range(args.layers + 1)),
            fc_dim=args.fc_dim,
            out_dim=args.out_dim,
            activation=args.activation,
            activate_last_layer=args.activate_last_layer,
            mollifier=args.mollifier,
            constraint=constraint_cfg,
            moe_config=MoEConfig(
                split=args.split,
                num_experts=(args.expert_t, args.expert_x, args.expert_y)
            )
        )
    elif args.model.lower() == 'fno2d':
        model_cfg = FNO2DConfig(
            model=args.model,
            modes1=tuple(args.modes1 for _ in range(args.layers)),
            modes2=tuple(args.modes2 for _ in range(args.layers)),
            layers=tuple(args.layer_dim for _ in range(args.layers + 1)),
            fc_dim=args.fc_dim,
            out_dim=args.out_dim,
            activation=args.activation,
            activate_last_layer=args.activate_last_layer,
            constraint=constraint_cfg,
            moe_config=MoEConfig(
                split=args.split,
                num_experts=(args.expert_t, args.expert_x)
            )
        )
    else:
        raise NotImplementedError(f'Could not find model {args.model}')
    training_cfg = TrainingConfig(
        num_epochs=args.num_epochs,
        optimizer=OptimizerConfig(
            name=args.optimizer,
            learning_rate=args.learning_rate
        ),
        seed=args.seed,
        save_checkpoints=not args.no_checkpoints,
        jit=not args.no_jit,
        log_every_n_steps=args.log_every_n_steps,
        num_callback_points=args.num_callback_points,
        ic_loss_weight=args.ic_loss_weight,
        bc_loss_weight=args.bc_loss_weight,
        interior_loss_weight=args.interior_loss_weight,
        pde_loss_weight=args.pde_loss_weight,
        data_loss_normalize=not args.no_data_loss_normalize,
        icbc_loss_normalize=not args.no_icbc_loss_normalize,
        use_tqdm=args.use_tqdm
    )

    dataset_cfg = DatasetConfig(
        name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size
    )
    pde = None
    if 'navier-stokes' in args.dataset.lower():
        pde = NavierStokes2D()
    elif 'diffusion-sorption' in args.dataset.lower():
        pde = DiffusionSorption1D()

    return model_cfg, training_cfg, dataset_cfg, pde
