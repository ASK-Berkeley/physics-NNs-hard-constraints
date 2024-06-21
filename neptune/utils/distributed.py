import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

global_devices = mesh_utils.create_device_mesh((jax.device_count(),))
global_sharding = PositionalSharding(global_devices)

def shard_array(x, sharding=global_sharding):
    # For an array of shape N x .. x .. shards across N devices
    new_shape = (jax.device_count(),) + \
        tuple(1 for _ in range(len(x.shape[1:])))
    return jax.lax.with_sharding_constraint(x, sharding.reshape(new_shape))

def replicate_array(x, sharding=global_sharding):
    return jax.lax.with_sharding_constraint(x, sharding.replicate())
