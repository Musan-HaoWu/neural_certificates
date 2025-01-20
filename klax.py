from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.jax_utils import prefetch_to_device
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf
from flax.training import train_state  # Useful dataclass to keep train state
import flax
import numpy as onp  # Ordinary NumPy
import optax  # Optimizers
from functools import partial
from gymnasium import spaces

klax_config = {"eps": 1e-3}

def project(x):
    '''
    apply h1 projection to point in R^n
    '''
    return x/(jnp.sum(jnp.abs(x))+1)

def v_project(x_list):
    return jax.vmap(project)(x_list)

def inverse_project(y):
    '''
    apply h1^-1 projection to point in [-1,1]^n
    '''
    # if jnp.sum(jnp.abs(y)) >= 1:
    #     print("Domain Error") 
    return y/(1-jnp.sum(jnp.abs(y)))

def v_inverse_project(y_list):
    return jax.vmap(inverse_project)(y_list)

def pretty_time(elapsed):
    if elapsed > 60 * 60:
        h = elapsed // (60 * 60)
        mins = (elapsed // 60) % 60
        return f"{h}h {mins:02d} min"
    elif elapsed > 60:
        mins = elapsed // 60
        secs = int(elapsed) % 60
        return f"{mins:0.0f}min {secs}s"
    elif elapsed < 1:
        return f"{elapsed*1000:0.1f}ms"
    else:
        return f"{elapsed:0.1f}s"


def make_unsafe_spaces(obs_space, unsafe_bounds):
    unsafe_spaces = []
    dims = obs_space.shape[0]
    for i in range(dims):
        low = onp.array(obs_space.low)
        high = onp.array(obs_space.high)
        high[i] = -unsafe_bounds[i]
        if not onp.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=onp.float32))

        high = onp.array(obs_space.high)
        low = onp.array(obs_space.low)
        low[i] = unsafe_bounds[i]
        if not onp.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=onp.float32))
    return unsafe_spaces


@jax.jit
def clip_grad_norm(grad, max_norm):
    '''
    Gradient clipping is a technique used to prevent the gradients from exploding during training. 
    It limits the magnitude of the gradient updates, ensuring that the optimization process remains stable.
    '''
    norm = jnp.linalg.norm(
        jnp.asarray((jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad))))
    )
    factor = jnp.minimum(max_norm, max_norm / (norm + 1e-6))
    return jax.tree_map((lambda x: x * factor), grad)


def contained_in_any(spaces, state):
    for space in spaces:
        if space.contains(state):
            return True
    return False


def triangular(rng_key, shape):
    '''
    sample from a triangular distribution over [-1,1]
    '''
    U = jax.random.uniform(rng_key, shape=shape)
    p1 = -1 + jnp.sqrt(2 * U)
    p2 = 1 - jnp.sqrt((1 - U) * 2)
    return jnp.where(U <= 0.5, p1, p2)


def softhuber(x):
    return jnp.sqrt(1 + jnp.square(x)) - 1


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
    softplus_output: bool = False
    
    @nn.compact
    def __call__(self, x):
        if len(self.features)==0:
            # empty MLP with no parameters
            print("here")
            return x
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            if self.activation == "relu":
                x = nn.relu(x)
            else:
                x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        if self.softplus_output:
            x = jax.nn.softplus(x)
        return x

# Must be called "Dense" because flax uses self.__class__.__name__ to name variables
# Implements a dense layer customized for interval bound propagation (IBP)
class Dense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, inputs):
        lower_bound_head, upper_bound_head = inputs
        kernel = self.param(
            "kernel",
            jax.nn.initializers.glorot_uniform(),
            (lower_bound_head.shape[-1], self.features),
        )  # shape info.
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        # Center and width
        center_prev = 0.5 * (upper_bound_head + lower_bound_head)
        edge_len_prev = 0.5 * jnp.maximum(
            upper_bound_head - lower_bound_head, 0
        )  # avoid numerical issues

        # Two matrix multiplications
        center = jnp.matmul(center_prev, kernel) + bias
        edge_len = jnp.matmul(edge_len_prev, jnp.abs(kernel))  # Edge length has no bias

        # New bounds
        lower_bound_head = center - edge_len
        upper_bound_head = center + edge_len
        # self.sow("intermediates", "edge_len", edge_len)
        return [lower_bound_head, upper_bound_head]


class IBPMLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
    softplus_output: bool = False

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = Dense(feat)(x)
            if self.activation == "relu":
                x = [nn.relu(x[0]), nn.relu(x[1])]
            else:
                x = [nn.tanh(x[0]), nn.tanh(x[1])]
        x = Dense(self.features[-1])(x)
        if self.softplus_output:
            x = [jax.nn.softplus(x[0]), jax.nn.softplus(x[1])]
        return x


def create_train_state(model, rng, in_dim, learning_rate):
    """
    Creates initial TrainState.
    """
    params = model.init(rng, jnp.ones([1, in_dim]))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def non_neg_loss(l):
    return jnp.mean(jnp.maximum(-l, 0))


def zero_at_zero_loss(l_at_zero):
    return jnp.mean(jnp.square(l_at_zero))


def martingale_loss(l, l_next, eps):
    diff = l_next - l
    return jnp.mean(jnp.maximum(diff + eps, 0))


def np_load(filename):
    arr = onp.load(filename)
    return {k: arr[k] for k in arr.files}


def jax_save(params, filename):
    bytes_v = flax.serialization.to_bytes(params)
    with open(filename, "wb") as f:
        f.write(bytes_v)


def jax_load(params, filename):
    with open(filename, "rb") as f:
        bytes_v = f.read()
    params = flax.serialization.from_bytes(params, bytes_v)
    return params


def lipschitz_l1_jax(params):
    lipschitz_l1 = 1
    # Max over input axis
    for i, (k, v) in enumerate(params["params"].items()):
        lipschitz_l1 *= jnp.max(jnp.sum(jnp.abs(v["kernel"]), axis=0))

    return lipschitz_l1
