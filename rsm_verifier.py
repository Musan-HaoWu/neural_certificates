import os
import time
from functools import partial

import gymnasium.spaces

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import tensorflow as tf
import jax.numpy as jnp

from tqdm import tqdm
import numpy as np

from klax import (
    project,
    inverse_project,
    v_project,
    v_inverse_project,
)
# gpus = tf.config.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def v_contains(boxs, states):
    '''
    Check if a vector of states is contained in any of the boxes
        (support unbounded boxes(low=-inf, high=inf)) 
    Return:
        contains: a boolean vector of length states.shape[0]
    '''
    contains = jnp.zeros(states.shape[0], dtype=jnp.bool)
    for box in boxs:
        b_low = jnp.expand_dims(box.low, axis=0)
        b_high = jnp.expand_dims(box.high, axis=0)
        mask = jnp.logical_and(
            jnp.all(states >= b_low, axis=1), jnp.all(states <= b_high, axis=1)
        )
        contains = jnp.logical_or(mask, contains)
    return contains

def v_intersect(boxs, lb, ub):
    '''
    Check if a vector of boxs (described by lb and ub) intersect with any of the boxes
        (support unbounded boxes(low=-inf, high=inf)) 
    Return:
        contains: a boolean vector of length states.shape[0]
    '''
    contains = jnp.zeros(lb.shape[0], dtype=jnp.bool)
    for box in boxs:
        non_intersect = jnp.logical_or(ub<box.low, lb>box.high)
        intersect = jnp.logical_not(non_intersect)
        mask = jnp.all(intersect, axis=1)
        contains = jnp.logical_or(mask, contains)
    return contains
    # contains = jnp.zeros(lb.shape[0], dtype=jnp.bool)
    # for box in boxs:
    #     b_low = jnp.expand_dims(box.low, axis=0)
    #     b_high = jnp.expand_dims(box.high, axis=0)
    #     contain_lb = jnp.logical_and(lb >= b_low, lb <= b_high)
    #     contain_ub = jnp.logical_and(ub >= b_low, ub <= b_high)
    #     mask = jnp.all(jnp.logical_or(contain_lb, contain_ub), axis=1)
    #     contains = jnp.logical_or(mask, contains)
    # return contains

class TrainBuffer:
    def __init__(self, max_size=3_000_000):
        self.s = []
        self.max_size = max_size

    def append(self, s):
        if self.max_size is not None and len(self) > self.max_size:
            return
        self.s.append(np.array(s))

    def extend(self, lst):
        for e in lst:
            self.append(e)

    def __len__(self):
        if len(self.s) == 0:
            return 0
        return sum([s.shape[0] for s in self.s])

    @property
    def in_dim(self):
        return len(self.s[0])

    def as_tfds(self, rng, batch_size=512):
        train_s = np.concatenate(self.s, axis=0)   
        train_s = np.unique(train_s, axis=0) 
        print("Train set size: ", train_s.shape)  
        seeds = jax.random.randint(rng, minval=0, maxval=9, shape=(2,))

        np_rng = np.random.default_rng(int(seeds[0]))
        train_s = np_rng.permutation(train_s)
        train_ds = tf.data.Dataset.from_tensor_slices(train_s)
        train_ds = train_ds.shuffle(50000, seed=int(seeds[1])).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds


class Verifier:
    def __init__(
        self,
        learner,
        env,
        batch_size,
        reach_prob,
        grid_size, # decide grid size
        normalize = False,
        rng = jax.random.PRNGKey(222),
        debug = False,
    ):
        self.learner = learner
        self.env = env
        self.reach_prob = jnp.float32(reach_prob)
        self.normalize = normalize
  
        self.batch_size = batch_size
        self.refinement_enabled = True
        
        self.grid_size = grid_size
        self.pmass_n = 2
        self.grid_stream_size = 512*512 #1024 * 1024 # TODO try smaller
        
        self._cached_pmass_grid = None
        self._cached_state_grid = None
        self._debug_violations = None
        self.hard_constraint_violation_buffer = None
        self.train_buffer = TrainBuffer()
        self._perf_stats = {
            "apply": 0.0,
            "loop": 0.0,
        }
        self.v_get_grid_item = jax.vmap(
            self.get_grid_item, in_axes=(0, None), out_axes=0
        )

    def prefill_train_buffer(self):
        '''
            create a grid,
            add all center points into the train buffer
            return delta
        '''
        if self.env.observation_space.shape[0] == 2:
            n = 100
        elif self.env.observation_space.shape[0] == 3:
            n = 100
        else:
            n = 50
        state_grid, _, _ = self.get_unfiltered_grid(n=n)
        
        if self.normalize: 
            mask = jnp.abs(state_grid[:, 0]) + jnp.abs(state_grid[:, 1]) < 1
            sub_grid = state_grid[mask]
            contain_domain = v_contains([self.env.observation_space], v_inverse_project(sub_grid))                
            self.train_buffer.append(np.array(sub_grid[contain_domain]))
            delta = 2.0 / n
        else:
            self.train_buffer.append(np.array(state_grid))
            delta = (self.env.observation_space.high[0] - self.env.observation_space.low[0]) / n   
        return delta

    @partial(jax.jit, static_argnums=(0, 2))
    def get_grid_item(self, idx, n):
        '''
        Generates a grid item based on the given index and grid size.
        Parameters:
        idx (int): The index of the grid item to retrieve.
        n (int): The number of divisions in each dimension of the grid.
        Returns:
        jnp.array: The coordinates of the grid center.
        '''
        dims = self.env.observation_space.shape[0]
        target_points = [
            jnp.linspace(
                start=self.env.observation_space.low[i],
                stop=self.env.observation_space.high[i],
                num=n,
                retstep=True,
                endpoint=False,
            )
            for i in range(dims)
        ]
        target_points, retsteps = zip(*target_points)
        target_points = list(target_points)
        for i in range(dims):
            target_points[i] = target_points[i] + 0.5 * retsteps[i]
        inds = []
        for i in range(dims):
            inds.append(idx % n)
            idx = idx // n
        return jnp.array([target_points[i][inds[i]] for i in range(dims)])

    def get_refined_grid_template(self, delta, n):
        dims = self.env.observation_space.shape[0]
        grid, new_deltas = [], []
        for i in range(dims):
            samples, new_delta = jnp.linspace(
                -0.5 * delta,
                +0.5 * delta,
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples.flatten() + new_delta * 0.5)
            new_deltas.append(new_delta)
        grid = jnp.meshgrid(*grid)
        grid = jnp.stack(grid, axis=1)
        return grid, new_deltas[0]

    def get_pmass_grid(self):
        '''
        grid the noise space, and compute the probability mass of each cell
        '''
        if self._cached_pmass_grid is not None and not self.normalize:
            return self._cached_pmass_grid
        dims = len(self.env.noise_bounds[0])
        grid, steps = [], []
        for i in range(dims):
            samples, step = jnp.linspace(
                self.env.noise_bounds[0][i]*self.env.noise_coef[i],
                self.env.noise_bounds[1][i]*self.env.noise_coef[i],
                self.pmass_n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        grid_lb = jnp.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid_lb]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]
        batched_grid_lb = jnp.stack(grid_lb, axis=1)
        batched_grid_ub = jnp.stack(grid_ub, axis=1)
        
        pmass = self.env.integrate_noise(grid_lb, grid_ub)
        self._cached_pmass_grid = (pmass, batched_grid_lb, batched_grid_ub)
        return pmass, batched_grid_lb, batched_grid_ub

    def get_unfiltered_grid(self, n=200):
        '''
        Generate a grid of points in the observation space of the environment,
        with `n` points in each dimension. 
        It returns the centers, lower bounds, and upper bounds of the grid cells.
        '''
        dims = self.env.observation_space.shape[0]
        grid, steps = [], []
        for i in range(dims):
            if self.normalize:
                lb, ub = -1.0, 1.0
            else:
                lb, ub = self.env.observation_space.low[i], self.env.observation_space.high[i]
            samples, step = jnp.linspace(
                lb,
                ub,
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        
        grid = jnp.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]
        grid_centers = [grid_lb[i] + steps[i] / 2 for i in range(dims)]

        grid_lb = jnp.stack(grid_lb, axis=1)
        grid_ub = jnp.stack(grid_ub, axis=1)
        grid_centers = jnp.stack(grid_centers, axis=1)

        return grid_centers, grid_lb, grid_ub

    def compute_bound_l(self, spaces):
        lb = [spaces[i].low for i in range(len(spaces))]
        ub = [spaces[i].high for i in range(len(spaces))]
        lbs, ubs = [], []
        for i in range(len(lb)):
            t1, t2 = self.finer_grid(lb[i], ub[i], N=10)
            lbs.append(t1)
            ubs.append(t2)
        x_boxes = self.make_box(lbs, ubs)
        y_lb, y_ub = self.v_get_y_box(x_boxes)
        return self.compute_bounds_on_set(jnp.array(y_lb), jnp.array(y_ub))

    def finer_grid(self, lb, ub, N=50):
        dims = self.env.observation_space.shape[0]
        grid, steps = [], []
        for i in range(dims):
            samples, step = jnp.linspace(
                lb[i],
                ub[i],
                N,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)

        grid = jnp.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]
        # grid_centers = [grid_lb[i] + steps[i] / 2 for i in range(dims)]

        grid_lb = jnp.stack(grid_lb, axis=1)
        grid_ub = jnp.stack(grid_ub, axis=1)
        # grid_centers = jnp.stack(grid_centers, axis=1)
        return grid_lb, grid_ub

    def compute_bound_init(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)
        if self.normalize:
            grid_lb2, grid_ub2 = v_inverse_project(grid_lb), v_inverse_project(grid_ub)
        else:
            grid_lb2, grid_ub2 = grid_lb, grid_ub
        mask = v_intersect(self.env.init_spaces, grid_lb2, grid_ub2)
        # Exclude if both lb AND ub are in the target set
        contains_lb = v_contains(self.env.target_spaces, grid_lb2)
        contains_ub = v_contains(self.env.target_spaces, grid_ub2)
        mask = np.logical_and(
            mask, np.logical_not(np.logical_and(contains_lb, contains_ub))
        )

        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0

        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_unsafe(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)
        if self.normalize:
            grid_lb2, grid_ub2 = v_inverse_project(grid_lb), v_inverse_project(grid_ub)
        else:
            grid_lb2, grid_ub2 = grid_lb, grid_ub
        mask = v_intersect(self.env.unsafe_spaces, grid_lb2, grid_ub2)

        # unsafe and target spaces are disjoint
        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_domain(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)
        if self.normalize:
            grid_lb2, grid_ub2 = v_inverse_project(grid_lb), v_inverse_project(grid_ub)
        else:
            grid_lb2, grid_ub2 = grid_lb, grid_ub
        mask_domain = v_intersect([self.env.observation_space], grid_lb2, grid_ub2)
        # Exclude if both lb AND ub are in the target set
        # unsafe and target spaces are disjoint
        # contains_lb = v_contains(self.env.target_spaces, grid_lb2)
        # contains_ub = v_contains(self.env.target_spaces, grid_ub2)
        # mask_target = np.logical_not(np.logical_and(contains_lb, contains_ub))
        # mask = np.logical_and(mask_domain, mask_target)
        
        # grid_lb = grid_lb[mask]
        # grid_ub = grid_ub[mask]
        grid_lb = grid_lb[mask_domain]
        grid_ub = grid_ub[mask_domain]

        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def get_domain_partition(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        # Exclude if both lb AND ub are in the target set
        contains_lb = v_contains(self.env.target_spaces, grid_lb)
        contains_ub = v_contains(self.env.target_spaces, grid_ub)
        mask = np.logical_not(np.logical_and(contains_lb, contains_ub))

        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return grid_lb, grid_ub

    def get_domain_jitter_grid(self, n):
        '''
        Generate a grid, exclude those cells that are contained in the target set
        '''
        grid_center, grid_lb, grid_ub = self.get_unfiltered_grid(n)
        stepsize = grid_ub[0, 0] - grid_lb[0, 0]
        contains = v_contains(self.env.target_spaces, grid_center)
        mask = np.logical_not(contains)
        grid_center = grid_center[mask]
        # assert grid_center.shape[0] > 0
        return grid_center, stepsize

    def compute_lipschitz_bound_on_domain(self, ibp_fn, params, n):
        grid_lb, grid_ub = self.get_domain_partition(n=n)
        lb, ub = ibp_fn.apply(params, [grid_lb, grid_ub])
        delta = (grid_ub[0] - grid_lb[0]).sum()

        max_dist = jnp.max(jnp.sum(ub - lb, axis=1))
        lip2 = max_dist / delta
        return lip2

    def compute_bounds_on_set(self, grid_lb, grid_ub):
        """
        Computes the global minimum and maximum bounds over a set of input intervals.

        Args:
            grid_lb (array-like): Lower bounds of the input intervals.
            grid_ub (array-like): Upper bounds of the input intervals.

        Returns:
            tuple: A tuple containing the global minimum and maximum bounds as floats.
        """
        global_min = jnp.inf
        global_max = -jnp.inf #jnp.NINF
        for i in range(int(np.ceil(grid_ub.shape[0] / self.batch_size))):
            start = i * self.batch_size
            end = np.minimum((i + 1) * self.batch_size, grid_ub.shape[0])
            batch_lb = jnp.array(grid_lb[start:end])
            batch_ub = jnp.array(grid_ub[start:end])
            lb, ub = self.learner.l_ibp.apply(
                self.learner.l_state.params, [batch_lb, batch_ub]
            )
            global_min = jnp.minimum(global_min, jnp.min(lb))
            global_max = jnp.maximum(global_max, jnp.max(ub))
        return float(global_min), float(global_max)

    # @partial(jax.jit, static_argnums=(0,))
    def compute_expected_l(self, params, s):
        pmass, batched_grid_lb, batched_grid_ub = self._cached_pmass_grid
        deterministic_s_next = self.env.v_next(s)
        batch_size = s.shape[0]
        ibp_size = batched_grid_lb.shape[0]
        obs_dim = self.env.observation_space.shape[0]

        # Broadcasting happens here, that's why we don't do directly vmap (although it's probably possible somehow)
        deterministic_s_next = deterministic_s_next.reshape((batch_size, 1, obs_dim))
        
        batched_grid_lb = batched_grid_lb.reshape((1, ibp_size, obs_dim))
        batched_grid_ub = batched_grid_ub.reshape((1, ibp_size, obs_dim))

        batched_grid_lb = batched_grid_lb + deterministic_s_next
        batched_grid_ub = batched_grid_ub + deterministic_s_next

        batched_grid_lb = batched_grid_lb.reshape((-1, obs_dim))
        batched_grid_ub = batched_grid_ub.reshape((-1, obs_dim))

        lb, ub = self.learner.l_ibp.apply(params, [batched_grid_lb, batched_grid_ub])
        ub = ub.reshape((batch_size, ibp_size))

        pmass = pmass.reshape((1, ibp_size))  # Boradcast to batch size
        exp_terms = pmass * ub
        expected_value = jnp.sum(exp_terms, axis=1)
        return expected_value
    
    def make_box(self, grid_lb, grid_ub):
        # grid_lb = grid_lb.T
        # grid_ub = grid_ub.T
        boxs = []
        for i in range(len(grid_lb)):
            box = []
            for j1 in range(len(grid_lb[i])):
                for j2 in range(len(grid_ub[i])):
                    box.append((grid_lb[i][j1], grid_ub[i][j2]))
            boxs.append(box) 
        return jnp.array(boxs)

    def get_y_box(self, x_box):
        y_polytope = v_project(x_box)
        y_lb = np.min(y_polytope, axis=0)
        y_ub = np.max(y_polytope, axis=0)
        return y_lb, y_ub

    def v_get_y_box(self, x_boxes):
        return jax.vmap(self.get_y_box)(x_boxes)

    def compute_expected_l_normalize_each(self, params, y):
        deterministic_x_next = self.env.next(inverse_project(y))
        grid, steps = [], []
        dims = len(self.env.noise_bounds[0])
        for i in range(dims):
            samples, step = jnp.linspace(
                self.env.noise_bounds[0][i]*self.env.noise_coef[i],
                self.env.noise_bounds[1][i]*self.env.noise_coef[i],
                self.pmass_n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        pgrid_lb = jnp.meshgrid(*grid)
        pgrid_lb = [x.flatten() for x in pgrid_lb]
        pgrid_ub = [pgrid_lb[i] + steps[i] for i in range(dims)]
        pmass = self.env.integrate_noise(pgrid_lb, pgrid_ub)
        
        xgrid_lb = jnp.array(pgrid_lb) + deterministic_x_next[:, jnp.newaxis]
        xgrid_ub = jnp.array(pgrid_ub) + deterministic_x_next[:, jnp.newaxis]
        x_boxes = self.make_box(xgrid_lb.T, xgrid_ub.T)
        y_lb, y_ub = self.v_get_y_box(x_boxes)
            
        _, ub = self.learner.l_ibp.apply(params, [y_lb, y_ub])
        exp_terms = pmass * ub
        expected_value = jnp.sum(exp_terms)
   
        return expected_value

    def compute_expected_l_normalize(self, params, ys):
        '''
        pmass is determined by state s 
        '''
        # pmass, batched_grid_lb, batched_grid_ub = self._cached_pmass_grid
        
        return jax.vmap(self.compute_expected_l_normalize_each, in_axes=(None, 0))(params, ys)


    @partial(jax.jit, static_argnums=(0,))
    def _check_dec_batch(self, l_params, grid_batch, l_batch, K):
        
        if self.normalize:
            e = self.compute_expected_l_normalize(
                l_params,
                grid_batch
            )
        else:
            e = self.compute_expected_l(
                l_params,
                grid_batch
            )

        violating_indices = (e >= l_batch - K)
        v = violating_indices.astype(jnp.int32).sum()

        hard_violating_indices = (e >= l_batch)
        hard_v = hard_violating_indices.astype(jnp.int32).sum()
        
        # average_decrease = jnp.mean(jnp.maximum(e - l_batch, 0))
        # print(f"Average decrease: {average_decrease}")
        return v, violating_indices, hard_v, hard_violating_indices

    @partial(jax.jit, static_argnums=(0,))
    def normalize_rsm(self, l, ub_init, lb_domain):
        l = l - lb_domain
        ub_init = ub_init - lb_domain
        # now min = 0
        l = l / jnp.maximum(ub_init, 1e-6)
        # now init max = 1
        return l

    def check_dec_cond(self, lipschitz_k):
        print("Checking expected decrease condition...")
        loop_start_time = time.perf_counter()

        dims = self.env.observation_space.shape[0]
        grid_total_size = self.grid_size ** dims
        
        info_dict = {}
        if self.normalize:  
            delta = 2.0 / self.grid_size
        else:
            delta = (self.env.observation_space.high[0] - self.env.observation_space.low[0]) / self.grid_size

        # estimate the ub_init and lb_domain, used for normalization
        if self.env.observation_space.shape[0] == 2:
            n = 100
        elif self.env.observation_space.shape[0] == 3:
            n = 100
        else:
            n = 50
        lb_domain, _ = self.compute_bound_domain(n)

        if self.normalize:
            _, ub_init = self.compute_bound_l(self.env.init_spaces)
            lb_unsafe, _ = self.compute_bound_l(self.env.unsafe_spaces)
            print(f"lb_unsafe={lb_unsafe:0.4f}")
        else:
            _, ub_init = self.compute_bound_init(n)
        

        K = jnp.float32(lipschitz_k * delta)
        info_dict["delta"] = delta
        info_dict["K"] = K
        print(f"Checking grid of size {self.grid_size}")
        print(f"ub_init={ub_init:0.4f}, lb_domain={lb_domain:0.4f}")
        print(f"lipschitz_k={lipschitz_k:.4f} (without delta)")
        print(f"delta={delta:.4f}")
        print(f"lipschitz_k*delta={K:.4f}")
        
        if not self.normalize:
            self.get_pmass_grid()  # cache pmass grid

        # expected_l_next = []
        # avg_decrease = []
        # # self.hard_constraint_violation_buffer = None
        # # _perf_loop_start = time.perf_counter()

        # grid_violating_indices = []
        # grid_hard_violating_indices = []
        # total_kernel_time = 0
        # total_kernel_iters = 0
        # stream size should not be too large
        grid_stream_size = min(grid_total_size, self.grid_stream_size)
        
        refinement_buffer = [] # grid needs to be refined
        violations = 0
        hard_violations = 0
        violation_buffer = []
        hard_violation_buffer = []

        pbar = tqdm(total=grid_total_size // grid_stream_size)        
        for i in range(0, grid_total_size, grid_stream_size):
            
            idx = jnp.arange(i, i + grid_stream_size)
            sub_grid = jnp.array(self.v_get_grid_item(idx, self.grid_size))
            if self.normalize:
                contains = v_contains(self.env.target_spaces, v_inverse_project(sub_grid))
            else:
                contains = v_contains(self.env.target_spaces, sub_grid)

            sub_grid = sub_grid[jnp.logical_not(contains)]

            # kernel_start = time.perf_counter()
            for start in range(0, sub_grid.shape[0], self.batch_size):
                end = min(start + self.batch_size, sub_grid.shape[0])
                grid_batch = jnp.array(sub_grid[start:end])

                # TODO: later optimize this by filtering the entire stream first
                l_batch = self.learner.l_state.apply_fn(
                    self.learner.l_state.params, grid_batch
                ).flatten()
                normalized_l_batch = self.normalize_rsm(l_batch, ub_init, lb_domain)
                # less_than_p = normalized_l_batch - K < 1 / (1 - self.reach_prob) #TODO: why -K here
                less_than_p = normalized_l_batch < 1 / (1 - self.reach_prob) 
                if jnp.sum(less_than_p.astype(np.int32)) == 0:
                    continue
                
                grid_batch = grid_batch[less_than_p]
                l_batch = l_batch[less_than_p]
                (
                    v,
                    violating_indices,
                    hard_v,
                    hard_violating_indices,
                ) = self._check_dec_batch(
                    self.learner.l_state.params,
                    grid_batch,
                    l_batch,
                    K
                )
                hard_violations += hard_v
                violations += v
                if v > 0:
                    refinement_buffer.append(grid_batch[violating_indices])
                if hard_v > 0:
                    hard_violation_buffer.append(grid_batch[hard_violating_indices])
            pbar.update(1)

            # if i > 0:
            #     total_kernel_time += time.perf_counter() - kernel_start
            #     total_kernel_iters += sub_grid.shape[0] // self.batch_size
            #     ints_per_sec = (
            #         total_kernel_iters * self.batch_size / total_kernel_time / 1000
            #     )
            #     pbar.set_description(
            #         f"kernel_t: {total_kernel_time*1e6/total_kernel_iters:0.2f}us/iter ({ints_per_sec:0.2f} Kints/s)"
            #     )
            # if self.fail_check_fast and violations > 0:
                # break
        pbar.close()
        print(f"violations={violations}")
        print(f"hard_violations={hard_violations}")

        # print(hard_violation_buffer)
        self.train_buffer.extend(hard_violation_buffer)

        loop_time = time.perf_counter() - loop_start_time
        
        if hard_violations == 0 and violations > 0:
            print(f"Zero hard violations -> refinement of {violations} soft violations")
            _perf_refine_buffer = time.perf_counter()
            refinement_buffer = [np.array(g) for g in refinement_buffer]
            refinement_buffer = np.concatenate(refinement_buffer)
            print(
                f"Took {time.perf_counter()-_perf_refine_buffer:0.2f}s to build refinement buffer"
            )
            print(
                f"Refine took {time.perf_counter()-_perf_refine_buffer:0.2f}s in total"
            )
            if self.refine_grid(
                refinement_buffer, lipschitz_k, delta, ub_init, lb_domain
            ):
                print("Refinement successful!")
                return True, 0, info_dict
            else:
                print("Refinement unsuccessful")
        
        # info_dict["avg_increase"] = (
        #     np.mean(avg_decrease) if len(avg_decrease) > 0 else 0
        # )
        info_dict["violations"] = f"{violations}/{grid_total_size}"
        info_dict["hard_violations"] = f"{hard_violations}/{grid_total_size}"
        print(f"{violations}/{grid_total_size} violations")
        print(f"{hard_violations}/{grid_total_size} hard violations")
        # print(f"Train buffer len: {len(self.train_buffer)}")
        if loop_time > 60:
            print(f"Grid runtime={loop_time/60:0.0f} min")
        else:
            print(f"Grid runtime={loop_time:0.2f} s")

        return violations == 0, hard_violations, info_dict

    def refine_grid(
        self, refinement_buffer, lipschitz_k, current_delta, ub_init, lb_domain
    ):
        n_dims = self.env.observation_space.shape[0]
        batch_size = 10 if self.env.observation_space.shape[0] == 2 else 5
        n = 10 if self.env.observation_space.shape[0] == 2 else 4

        template_batch, new_delta = self.get_refined_grid_template(current_delta, n)
        K = jnp.float32(lipschitz_k * new_delta)
        
        template_batch = template_batch.reshape((1, -1, n_dims))
        for i in tqdm(range(int(np.ceil(refinement_buffer.shape[0] / batch_size)))):
            start = i * batch_size
            end = np.minimum((i + 1) * batch_size, refinement_buffer.shape[0])
            s_batch = jnp.array(refinement_buffer[start:end])
            s_batch = s_batch.reshape((-1, 1, n_dims))
            r_batch = s_batch + template_batch
            r_batch = r_batch.reshape((-1, self.env.observation_space.shape[0]))  # flatten

            l_batch = self.learner.l_state.apply_fn(
                self.learner.l_state.params, r_batch
            ).flatten()
            normalized_l_batch = self.normalize_rsm(l_batch, ub_init, lb_domain)
            less_than_p = normalized_l_batch - K < 1 / (1 - self.reach_prob)
            if jnp.sum(less_than_p.astype(np.int32)) == 0:
                continue
            r_batch = r_batch[less_than_p]
            l_batch = l_batch[less_than_p]

            (
                v,
                violating_indices,
                hard_v,
                hard_violating_indices,
            ) = self._check_dec_batch(
                self.learner.l_state.params,
                r_batch,
                l_batch,
                K,
            )
            if v > 0:
                return False
        return True
