from loguru import logger as local_log
import jax


def DEBUG(msg, *args):
    local_log.opt(depth=1).debug(msg, *args)


def INFO(msg, *args):
    local_log.opt(depth=1).info(msg, *args)


def WARNING():
    pass


def reweight_matrix(weight, matrix, activation_func=lambda x: x):
    return activation_func(matrix @ weight)


def pytree_print(pytree):
    elements = jax.tree_util.tree_flatten_with_path(pytree)[0]
    def fmt_shape(x): return '[' + ','.join([str(i) for i in x]) + ']'
    for path, e in elements:
        if e is None:
            print(type(pytree).__name__ + jax.tree_util.keystr(path) + ': None')
        else:
            print(type(pytree).__name__ + jax.tree_util.keystr(path) +
                  ': ' + fmt_shape(e.shape))

def pytree_in_axes(pytree, in_axes):
    # Returns a pytree def with in_axes broadcasted
    return jax.tree_map(lambda _: in_axes, pytree)