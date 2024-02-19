from jax import nn
import jax.numpy as jnp


def Activation_Function(activation: str):
    if activation == 'tanh':
        return jnp.tanh
    if activation == 'gelu':
        return nn.gelu
    if activation == 'relu':
        return nn.relu
    raise ValueError(f'Activation function {activation} not implemented')
