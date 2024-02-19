import jax
import jax.numpy as jnp
import typing
import dataclasses
import equinox as eqx

from .geometry import Grid, Function, Mask, SparseGrid, DenseGrid


class_type = typing.TypeVar('class_type')
Callback = typing.Callable[[], typing.Any]
Loss = typing.Callable[[jax.Array, jax.Array], jax.Array]
Forward_func = typing.Callable[[jax.Array, typing.Any], jax.Array]


@dataclasses.dataclass
class ModelInput:
    domain: Grid
    pde_param: Function
    initial_condition: Function


def model_input_tree_flatten(model_input: ModelInput) -> typing.Tuple[typing.Iterable, typing.Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('domain'), model_input.domain),
        (jax.tree_util.GetAttrKey('pde_param'), model_input.pde_param),
        (jax.tree_util.GetAttrKey('initial_condition'),
            model_input.initial_condition),
    ]
    return flat_contents, dict()


def model_input_tree_unflatten(aux_data: typing.Mapping, data: typing.Iterable) -> ModelInput:
    return ModelInput(*data)


jax.tree_util.register_pytree_with_keys(
    ModelInput, model_input_tree_flatten, model_input_tree_unflatten)


@dataclasses.dataclass
class ModelOutput:
    solution: Function
    evaluation_partials: typing.Optional[typing.Dict[typing.Any, Grid]] = None
    evaluation_mask: typing.Optional[Mask] = None
    weight: typing.Optional[jax.Array] = None
    solver_status: typing.Optional[eqx.Enumeration] = None
    solver_error: typing.Optional[jax.Array] = None
    solver_iter: typing.Optional[int] = None


def model_output_tree_flatten(model_output: ModelOutput) -> typing.Tuple[typing.Iterable, typing.Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('solution'), model_output.solution),
        (jax.tree_util.GetAttrKey('evaluation_partials'),
            model_output.evaluation_partials),
        (jax.tree_util.GetAttrKey('evaluation_mask'),
            model_output.evaluation_mask),
        (jax.tree_util.GetAttrKey('weight'), model_output.weight),
        (jax.tree_util.GetAttrKey('solver_status'), model_output.solver_status),
        (jax.tree_util.GetAttrKey('solver_error'), model_output.solver_error),
        (jax.tree_util.GetAttrKey('solver_iter'), model_output.solver_iter),
    ]
    return flat_contents, dict()


def model_output_tree_unflatten(aux_data: typing.Mapping, data: typing.Iterable) -> ModelOutput:
    return ModelOutput(*data)


jax.tree_util.register_pytree_with_keys(
    ModelOutput, model_output_tree_flatten, model_output_tree_unflatten)


def merge_functions(fn1, fn2):
    return Function(DenseGrid(jnp.concatenate((fn1.domain.grid, fn2.domain.grid), axis=0)),
                    jnp.concatenate((fn1.image, fn2.image), axis=0))


@dataclasses.dataclass
class CallbackPayload:
    epoch: int
    loss: int
    predicted_solution: Function
    pde_sol: Function
    pde: typing.Any
    split: str  # One of ['train', 'validation']
    pde_param: Function

    def merge(self, other):
        return CallbackPayload(epoch=self.epoch,
                               loss=self.loss + other.loss,
                               predicted_solution=merge_functions(
                                   self.predicted_solution, other.predicted_solution),
                               pde_sol=merge_functions(
                                   self.pde_sol, other.pde_sol),
                               pde=self.pde,
                               split=self.split,
                               pde_param=merge_functions(self.pde_param, other.pde_param))


def callback_payload_tree_flatten(callback_payload: CallbackPayload) -> typing.Tuple[typing.Iterable, typing.Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('epoch'), callback_payload.epoch),
        (jax.tree_util.GetAttrKey('loss'), callback_payload.loss),
        (jax.tree_util.GetAttrKey('predicted_solution'),
            callback_payload.predicted_solution),
        (jax.tree_util.GetAttrKey('pde_sol'), callback_payload.pde_sol),
        (jax.tree_util.GetAttrKey('pde'), callback_payload.pde),
        (jax.tree_util.GetAttrKey('split'), callback_payload.split),
        (jax.tree_util.GetAttrKey('pde_param'), callback_payload.pde_param)
    ]
    return flat_contents, dict()


def callback_payload_tree_unflatten(aux_data: typing.Mapping, data: typing.Iterable) -> CallbackPayload:
    return CallbackPayload(*data)


jax.tree_util.register_pytree_with_keys(
    CallbackPayload, callback_payload_tree_flatten, callback_payload_tree_unflatten)


@dataclasses.dataclass
class LoggingMetrics:
    predicted_solution: Function
    loss: float
    ic_loss: float
    left_bc_loss: float
    right_bc_loss: float
    data_loss: float
    pde_loss: typing.Optional[float] = None
    solver_weight: typing.Optional[jax.Array] = None
    solver_status: typing.Optional[float] = None
    solver_iter: typing.Optional[int] = None
    grad_norm: typing.Optional[float] = None
    weight_deltas: typing.Optional[typing.List[jax.Array]] = None
    gt_residual: typing.Optional[float] = None
    raw_l2_loss: typing.Optional[float] = None


def logging_metrics_tree_flatten(logging_metrics: LoggingMetrics) -> typing.Tuple[typing.Iterable, typing.Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('predicted_solution'),
            logging_metrics.predicted_solution),
        (jax.tree_util.GetAttrKey('loss'), logging_metrics.loss),
        (jax.tree_util.GetAttrKey('ic_loss'), logging_metrics.ic_loss),
        (jax.tree_util.GetAttrKey('left_bc_loss'),
            logging_metrics.left_bc_loss),
        (jax.tree_util.GetAttrKey('right_bc_loss'),
            logging_metrics.right_bc_loss),
        (jax.tree_util.GetAttrKey('data_loss'), logging_metrics.data_loss),
        (jax.tree_util.GetAttrKey('pde_loss'), logging_metrics.pde_loss),
        (jax.tree_util.GetAttrKey('solver_weight'),
            logging_metrics.solver_weight),
        (jax.tree_util.GetAttrKey('solver_status'),
            logging_metrics.solver_status),
        (jax.tree_util.GetAttrKey('solver_iter'),
            logging_metrics.solver_iter),
        (jax.tree_util.GetAttrKey('grad_norm'), logging_metrics.grad_norm),
        (jax.tree_util.GetAttrKey('weight_deltas'),
            logging_metrics.weight_deltas),
        (jax.tree_util.GetAttrKey('gt_residual'),
         logging_metrics.gt_residual),
        (jax.tree_util.GetAttrKey('raw_l2_loss'),
         logging_metrics.raw_l2_loss),
    ]
    return flat_contents, dict()


def logging_metrics_tree_unflatten(aux_data: typing.Mapping, data: typing.Iterable) -> LoggingMetrics:
    return LoggingMetrics(*data)


jax.tree_util.register_pytree_with_keys(
    LoggingMetrics, logging_metrics_tree_flatten, logging_metrics_tree_unflatten)
