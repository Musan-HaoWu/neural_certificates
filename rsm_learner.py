import jax
import jax.numpy as jnp
from functools import partial

import tensorflow as tf

from klax import (
    jax_save,
    jax_load,
    lipschitz_l1_jax,
    # martingale_loss,
    triangular,
    IBPMLP,
    MLP,
    create_train_state,
    zero_at_zero_loss,
    clip_grad_norm,
)
import numpy as np


class Learner:
    def __init__(
        self,
        env,
        l_hidden,
        l_lip,
        eps, # epsilon in the expected decrease condition
        gamma_decrease, #???
        reach_prob,
        softplus_l_output=True,
    ) -> None:
        self.env = env
        self.eps = jnp.float32(eps)
        self.gamma_decrease = gamma_decrease #???
        self.reach_prob = jnp.float32(reach_prob) 
        
        obs_dim = self.env.observation_space.shape[0]

        l_net = MLP(l_hidden + [1], activation="relu", softplus_output=softplus_l_output)
        
        # l_ibp: interval back propagation, used in verification
        self.l_ibp = IBPMLP(
            l_hidden + [1], activation="relu", softplus_output=softplus_l_output
        )
        
        self.l_state = create_train_state(l_net, jax.random.PRNGKey(1), obs_dim, 0.0005)
        self.l_lip = jnp.float32(l_lip)

        self.rng = jax.random.PRNGKey(777)

        self._debug_init = []
        self._debug_unsafe = []

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_init(self, rng, n):
        rngs = jax.random.split(rng, len(self.env.init_spaces))
        per_space_n = n // len(self.env.init_spaces)

        batch = []
        for i in range(len(self.env.init_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_space.shape[0]),
                minval=self.env.init_spaces[i].low,
                maxval=self.env.init_spaces[i].high,
            )
            batch.append(x)
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_unsafe(self, rng, n):
        rngs = jax.random.split(rng, len(self.env.unsafe_spaces))
        per_space_n = n // len(self.env.unsafe_spaces)

        batch = []
        for i in range(len(self.env.unsafe_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_space.shape[0]),
                minval=self.env.unsafe_spaces[i].low,
                maxval=self.env.unsafe_spaces[i].high,
            )
            batch.append(x)
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_target(self, rng, n):
        rngs = jax.random.split(rng, len(self.env.target_spaces))
        per_space_n = n // len(self.env.target_spaces)

        batch = []
        for i in range(len(self.env.target_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_space.shape[0]),
                minval=self.env.target_spaces[i].low,
                maxval=self.env.target_spaces[i].high,
            )
            batch.append(x)
        return jnp.concatenate(batch, axis=0)
    
    @partial(jax.jit, static_argnums=(0, ))
    def train_step(self, l_state, state, lip, rng, current_delta):
        """
            Train for a single step.
            
            state: train states
        """
        rngs = jax.random.split(rng, 4)
        
        # to evaluate the network 
        init_samples = self.sample_init(rngs[1], 256)
        unsafe_samples = self.sample_unsafe(rngs[2], 256)
        target_samples = self.sample_target(rngs[3], 64)
        
        # Adds a bit of randomization to the grid
        # s_random = jax.random.uniform(rngs[4], state.shape, minval=-0.5, maxval=0.5)
        # state = state + current_delta * s_random

        def loss_fn(l_params):
            # l has softplus at output -> always negative
            l = l_state.apply_fn(l_params, state)
            l_at_init = l_state.apply_fn(l_params, init_samples)
            l_at_unsafe = l_state.apply_fn(l_params, unsafe_samples)
            l_at_target = l_state.apply_fn(l_params, target_samples)
            max_at_init = jnp.max(l_at_init)
            min_at_init = jnp.min(l_at_init)
            min_at_unsafe = jnp.min(l_at_unsafe)
            min_at_target = jnp.min(l_at_target)
            

            # loss_decrease
            N = 16
            s_next_det = self.env.v_next(state)
            sample_rngs = jax.random.split(rngs[0], N*len(state))
            s_next = self.env.v_add_noise(jnp.repeat(s_next_det, N, axis=0), sample_rngs)
            s_next_fn = jax.vmap(l_state.apply_fn, in_axes=(None, 0))
            l_next = s_next_fn(l_state.params, s_next)
            l_next = l_next.reshape(len(state), N)

            exp_l_next = jnp.mean(l_next, axis=1)
            dec_loss = jnp.mean(jnp.maximum(exp_l_next - l + self.eps, 0))
            loss = dec_loss 

            violations = (exp_l_next >= l).astype(jnp.float32)
            violations = jnp.mean(violations)

            # loss_init
            loss += jnp.maximum(max_at_init - 1, 0)
            
            # loss_unsafe
            loss += jnp.maximum( 1/(1-self.reach_prob) - min_at_unsafe , 0)

            # loss_lipschitz
            loss += lip * jnp.maximum(lipschitz_l1_jax(l_params) - self.l_lip, 0)  

            # loss_auxiliary
            #  guide the learner towards a candidate that attains the 
            #  global minimum in a state that is contained within the target set
            point = self.env.target_spaces[0].sample()
            l_at_zero = l_state.apply_fn(l_params, point) # returns an one element array
            loss += jnp.sum(jnp.maximum(jnp.abs(l_at_zero)-0.3, 0)) 
            loss += jnp.maximum(min_at_target - min_at_init, 0)
            loss += jnp.maximum(min_at_target - min_at_unsafe, 0)
            return loss, (dec_loss, violations)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (dec_loss, violations)), l_grad = grad_fn(l_state.params)
        l_grad = clip_grad_norm(l_grad, 1)
        l_state = l_state.apply_gradients(grads=l_grad)
        metrics = {"loss": loss, "dec_loss": dec_loss, "train_violations": violations}
        return l_state, metrics

    def train_epoch(
        self, train_ds, current_delta=0, lip=0.1
    ):
        """Train for a single epoch."""
        lip = jnp.float32(lip)
        current_delta = jnp.float32(current_delta)
        batch_metrics = []

        #???
        if isinstance(train_ds, tf.data.Dataset):
            iterator = train_ds.as_numpy_iterator()
        else:
            iterator = range(80)
            train_ds = jnp.array(train_ds)
        
        for state in iterator:
            if isinstance(train_ds, tf.data.Dataset):
                state = jnp.array(state)
            else:
                state = train_ds
            self.rng, rng = jax.random.split(self.rng, 2)

            new_l_state, metrics = self.train_step(
                self.l_state,
                state,
                lip,
                rng,
                current_delta
            )
            self.l_state = new_l_state
            batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }

        return epoch_metrics_np

    def save(self, filename):
        jax_save({"martingale": self.l_state},filename)

    def load(self, filename, force_load_all=False):
        try:
            params = jax_load({"martingale": self.l_state,},filename)
            self.l_state = params["martingale"]
        except KeyError as e:
            print("Error loading model")