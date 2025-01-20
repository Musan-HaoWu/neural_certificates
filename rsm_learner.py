import jax
import jax.numpy as jnp
from functools import partial

import tensorflow as tf

from klax import (
    project,
    inverse_project,
    v_project,
    v_inverse_project,
    jax_save,
    jax_load,
    lipschitz_l1_jax,
    # martingale_loss,
    # triangular,
    IBPMLP,
    MLP,
    create_train_state,
    # zero_at_zero_loss,
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
        reach_prob,
        softplus_l_output=True,
        normalize = False,
        rng = jax.random.PRNGKey(111),
        debug=False,
    ) -> None:
        self.env = env
        self.eps = jnp.float32(eps)
        self.reach_prob = jnp.float32(reach_prob) 
        self.normalize = normalize
        self.Nmax = 10000 # bound for sampling
        
        obs_dim = self.env.observation_space.shape[0]

        l_net = MLP(l_hidden + [1], activation="relu", softplus_output=softplus_l_output)
        
        # l_ibp: interval back propagation, used in verification
        self.l_ibp = IBPMLP(
            l_hidden + [1], activation="relu", softplus_output=softplus_l_output
        )
        self.rng = rng
        self.rng, rng = jax.random.split(self.rng, 2)
        self.l_state = create_train_state(l_net, rng, obs_dim, learning_rate=0.01)
        self.l_lip = jnp.float32(l_lip)

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
                minval=np.maximum(self.env.init_spaces[i].low, -self.Nmax),
                maxval=np.minimum(self.env.init_spaces[i].high, self.Nmax),
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
                minval=np.maximum(self.env.unsafe_spaces[i].low, -self.Nmax),
                maxval=np.minimum(self.env.unsafe_spaces[i].high, self.Nmax),
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
                minval=np.maximum(self.env.target_spaces[i].low, -self.Nmax),
                maxval=np.minimum(self.env.target_spaces[i].high, self.Nmax),
            )
            batch.append(x)
        return jnp.concatenate(batch, axis=0)
    
    # @partial(jax.jit, static_argnums=(0, ))
    def train_step(self, l_state, ds, lip, rng, current_delta):
        """
            Train for a single step.
            
            ds: train states
        """
        rngs = jax.random.split(rng, 5)
        
        # to evaluate the network 
        init_samples = self.sample_init(rngs[1], 256)
        unsafe_samples = self.sample_unsafe(rngs[2], 256)
        target_samples = self.sample_target(rngs[3], 64)
        
        # Adds a bit of randomization to the grid
        s_random = jax.random.uniform(rngs[4], ds.shape, minval=-0.01, maxval=0.01)
        ds = ds + current_delta * s_random
        
        s_next_fn = jax.vmap(l_state.apply_fn, in_axes=(None, 0))

        def loss_fn(l_params):
            # l has softplus at output -> always negative
            l = l_state.apply_fn(l_params, ds).flatten()
            l_at_init = l_state.apply_fn(l_params, init_samples)
            l_at_unsafe = l_state.apply_fn(l_params, unsafe_samples)
            l_at_target = l_state.apply_fn(l_params, target_samples)
            max_at_init = jnp.max(l_at_init)
            min_at_init = jnp.min(l_at_init)
            min_at_unsafe = jnp.min(l_at_unsafe)
            min_at_target = jnp.min(l_at_target)
            
            # loss_decrease
            N = 16
            s_next_det = self.env.v_next(ds)
            sample_rngs = jax.random.split(rngs[0], N*len(ds))
            s_next = self.env.v_add_noise(jnp.repeat(s_next_det, N, axis=0), sample_rngs)
            # s_next_fn = jax.vmap(l_state.apply_fn, in_axes=(None, 0))
            l_next = s_next_fn(l_state.params, s_next)
            l_next = l_next.reshape(len(ds), N)

            exp_l_next = jnp.mean(l_next, axis=1)
            dec_loss = jnp.mean(jnp.maximum(exp_l_next - l + self.eps, 0))
            loss = dec_loss 
            
            
            l_less = (l<= 1/(1-self.reach_prob))
            l_larger = (exp_l_next + self.eps >= l)
            violations = jnp.logical_and(l_less, l_larger)
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

    # @partial(jax.jit, static_argnums=(0, ))
    def train_step_normalize(self, l_state, ds, lip, rng, current_delta):
        rngs = jax.random.split(rng, 5)
        # to evaluate the network 
        init_samples = v_project(self.sample_init(rngs[1], 256))
        unsafe_samples = v_project(self.sample_unsafe(rngs[2], 256))
        target_samples = v_project(self.sample_target(rngs[3], 64))
        
        # s_random = jax.random.uniform(rngs[4], ds.shape, minval=-0.01, maxval=0.01)
        # ds = ds + current_delta * s_random
        s_next_fn = jax.vmap(l_state.apply_fn, in_axes=(None, 0))

        def loss_fn(l_params):
            # l has softplus at output -> always negative
            l = l_state.apply_fn(l_params, ds).flatten()
            l_at_init = l_state.apply_fn(l_params, init_samples)
            l_at_unsafe = l_state.apply_fn(l_params, unsafe_samples)
            l_at_target = l_state.apply_fn(l_params, target_samples)
            max_at_init = jnp.max(l_at_init)
            min_at_init = jnp.min(l_at_init)
            min_at_unsafe = jnp.min(l_at_unsafe)
            min_at_target = jnp.min(l_at_target)
            # loss_decrease
            N = 16
            s_next_det = self.env.v_next(v_inverse_project(ds))
            sample_rngs = jax.random.split(rngs[0], N*len(ds))
            s_next = self.env.v_add_noise(jnp.repeat(s_next_det, N, axis=0), sample_rngs)
            l_next = s_next_fn(l_state.params, v_project(s_next))
            l_next = l_next.reshape(len(ds), N)

            exp_l_next = jnp.mean(l_next, axis=1)
            
            dec_loss = jnp.mean(jnp.maximum(exp_l_next - l + self.eps, 0))
            loss = dec_loss 
        
            l_less = (l<= 1/(1-self.reach_prob))
            l_larger = (exp_l_next + self.eps >= l)
            violations = jnp.logical_and(l_less, l_larger)

            # print(l.shape, exp_l_next.shape)
            # raise ValueError
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
            # point = self.env.target_spaces[0].sample()
            point = jax.random.uniform(
                rngs[-1],
                (self.env.observation_space.shape[0]),
                minval=np.maximum(self.env.target_spaces[0].low, -100),
                maxval=np.minimum(self.env.target_spaces[0].high, 100),
            )            
            l_at_zero = l_state.apply_fn(l_params, project(point)) # returns an one element array
            loss += jnp.sum(jnp.maximum(jnp.abs(l_at_zero)-0.3, 0)) 
            loss += jnp.maximum(min_at_target - min_at_init, 0)
            loss += jnp.maximum(min_at_target - min_at_unsafe, 0)
            if jnp.isnan(loss):
                print("Nan loss")
                for i, val in enumerate(exp_l_next):
                    if jnp.isnan(val):
                        print("state", ds[i])
                raise ValueError
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

        iterator = train_ds.as_numpy_iterator()
        for state in iterator:
            state = jnp.array(state)
            self.rng, rng = jax.random.split(self.rng, 2)
            if self.normalize:
                new_l_state, metrics = self.train_step_normalize(
                    self.l_state,
                    state,
                    lip,
                    rng,
                    current_delta
                )
            else:   
                new_l_state, metrics = self.train_step(
                    self.l_state,
                    state,
                    lip,
                    rng,
                    current_delta
                )
            self.l_state = new_l_state
            batch_metrics.append(metrics)

        # debug
        # rng = jax.random.PRNGKey(0)
        # unsafe_samples = v_project(self.sample_unsafe(rng, 256))
        # l_at_unsafe = self.l_state.apply_fn(self.l_state.params, unsafe_samples)
        # min_at_unsafe = jnp.min(l_at_unsafe)
        # print("sample min unsafe:", min_at_unsafe)
        
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