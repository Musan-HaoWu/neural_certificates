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

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def v_contains(boxs, states):
    '''
    Check if a vector of states is contained in any of the boxes
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
    Return:
        contains: a boolean vector of length states.shape[0]
    '''
    contains = jnp.zeros(lb.shape[0], dtype=jnp.bool)
    for box in boxs:
        b_low = jnp.expand_dims(box.low, axis=0)
        b_high = jnp.expand_dims(box.high, axis=0)
        contain_lb = jnp.logical_and(lb >= b_low, lb <= b_high)
        contain_ub = jnp.logical_and(ub >= b_low, ub <= b_high)
        mask = jnp.all(jnp.logical_or(contain_lb, contain_ub), axis=1)
        contains = jnp.logical_or(mask, contains)
    return contains

class TrainBuffer:
    def __init__(self, max_size=3_000_000):
        self.s = []
        self.max_size = max_size
        self._cached_ds = None

    def append(self, s):
        if self.max_size is not None and len(self) > self.max_size:
            return
        self.s.append(np.array(s))
        self._cached_ds = None
        # print(f"XXX Adding item of size {s.shape} to ds (now of size {len(self)})")
        # if self.max_size is not None and len(self) > self.max_size:
        #     self.s.pop(0)

    def extend(self, lst):
        '''
            add a list of elements
        '''
        for s in lst:
            self.append(s)

    def __len__(self):
        if len(self.s) == 0:
            return 0
        return sum([s.shape[0] for s in self.s])

    @property
    def in_dim(self):
        return len(self.s[0])

    def as_tfds(self, batch_size=32):
        if self._cached_ds is not None:
            return self._cached_ds
        train_s = np.concatenate(self.s, axis=0)
        train_s = np.random.default_rng().permutation(train_s)
        train_ds = tf.data.Dataset.from_tensor_slices(train_s)
        train_ds = train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self._cached_ds = train_ds
        return train_ds


class Verifier:
    def __init__(
        self,
        learner,
        env,
        batch_size,
        reach_prob,
        fail_check_fast, #???
        grid_factor, # decide grid size
        small_mem, # can be removed
    ):
        self.learner = learner
        self.env = env
        self.reach_prob = jnp.float32(reach_prob)
        self._small_mem = small_mem # can be removed
        self.fail_check_fast = fail_check_fast
        # self.batch_size = batch_size
        # self.grid_stream_size = grid_stream_size

        self.batch_size = batch_size
        self.refinement_enabled = True
        
        self.grid_size = int(grid_factor * 500)
        self.pmass_n = 10
        self.grid_stream_size = 1024 * 1024
        
        # if env.observation_space.shape[0] == 2:
        #     self.grid_size = int(grid_factor * 500)
        #     self.pmass_n = 10
        #     self.grid_stream_size = 1024 * 1024
        # elif env.observation_space.shape[0] == 3:
        #     self.grid_size = int(grid_factor * 200)
        #     self.pmass_n = 10
        #     self.grid_stream_size = 2 * 1024 * 1024
        # else:
        #     self.refinement_enabled = False
        #     self.grid_size = int(grid_factor * 100)
        #     self.pmass_n = 10
        #     self.grid_stream_size = 8 * 1024 * 1024
        
        self._cached_pmass_grid = None
        self._cached_state_grid = None
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
            n = 200
        elif self.env.observation_space.shape[0] == 3:
            n = 100
        else:
            n = 50
        state_grid, _, _ = self.get_unfiltered_grid(n=n)
        self.train_buffer.append(np.array(state_grid))
        return (self.env.observation_space.high[0] - self.env.observation_space.low[0]) / n

    @partial(jax.jit, static_argnums=(0, 2))
    def get_grid_item(self, idx, n):
        '''
        Generates a grid item based on the given index and grid size.
        Parameters:
        idx (int): The index of the grid item to retrieve.
        n (int): The number of divisions in each dimension of the grid.
        Returns:
        jnp.array: The coordinates of the grid item corresponding to the given index.
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
            # grid.append(samples.flatten())
            new_deltas.append(new_delta)
        grid = jnp.meshgrid(*grid)
        grid = jnp.stack(grid, axis=1)
        return grid, new_deltas[0]

    def get_pmass_grid(self):
        '''
        grid the noise space, and compute the probability mass of each cell
        '''
        if self._cached_pmass_grid is not None:
            return self._cached_pmass_grid
        
        dims = len(self.env.noise_bounds[0])
        grid, steps = [], []
        for i in range(dims):
            samples, step = jnp.linspace(
                self.env.noise_bounds[0][i],
                self.env.noise_bounds[1][i],
                self.pmass_n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        grid_lb = jnp.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid_lb]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]

        if dims < self.env.observation_space.shape[0]:
            # Fill remaining dimensions with 0
            remaining = self.env.observation_space.shape[0] - len(self.env.noise_bounds)
            for i in range(remaining):
                grid_lb.append(jnp.zeros_like(grid_lb[0]))
                grid_ub.append(jnp.zeros_like(grid_lb[0]))
        
        batched_grid_lb = jnp.stack(grid_lb, axis=1)
        batched_grid_ub = jnp.stack(grid_ub, axis=1)
        
        pmass = self.env.integrate_noise(grid_lb, grid_ub)
        self._cached_pmass_grid = (pmass, batched_grid_lb, batched_grid_ub)
        return pmass, batched_grid_lb, batched_grid_ub

    def get_unfiltered_grid(self, n=100):
        '''
        Generate a grid of points in the observation space of the environment,
        with `n` points in each dimension. 
        It returns the centers, lower bounds, and upper bounds of the grid cells.
        '''
        dims = self.env.observation_space.shape[0]
        grid, steps = [], []
        for i in range(dims):
            samples, step = jnp.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
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

    def compute_bound_init(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)
        mask = v_intersect(self.env.init_spaces, grid_lb, grid_ub)
        
        # Exclude if both lb AND ub are in the target set
        contains_lb = v_contains(self.env.target_spaces, grid_lb)
        contains_ub = v_contains(self.env.target_spaces, grid_ub)
        mask = np.logical_and(
            mask, np.logical_not(np.logical_and(contains_lb, contains_ub))
        )

        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0

        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_unsafe(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)
        mask = v_intersect(self.env.unsafe_spaces, grid_lb, grid_ub)

        # unsafe and target spaces are disjoint
        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_domain(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        # Exclude if both lb AND ub are in the target set
        contains_lb = v_contains(self.env.target_spaces, grid_lb)
        contains_ub = v_contains(self.env.target_spaces, grid_ub)
        mask = np.logical_not(np.logical_and(contains_lb, contains_ub))

        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
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
        contains = v_contains(self.env.target_spaces[0], grid_center)
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
        for i in tqdm(range(int(np.ceil(grid_ub.shape[0] / self.batch_size)))):
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

    @partial(jax.jit, static_argnums=(0,))
    def compute_expected_l(self, params, s, pmass, batched_grid_lb, batched_grid_ub):
        '''
        TODO: normalize 
        '''
        deterministic_s_next = self.env.v_next(s)
        batch_size = s.shape[0]
        ibp_size = batched_grid_lb.shape[0]
        
        obs_dim = self.env.observation_space.shape[0]
        # s has (batch_size,2)

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

    @partial(jax.jit, static_argnums=(0,))
    def _check_dec_batch(self, l_params, grid_batch, l_batch, K):
        
        # if self.normalziation = True

        pmass, batched_grid_lb, batched_grid_ub = self._cached_pmass_grid
        e = self.compute_expected_l(
            l_params,
            grid_batch,
            pmass,
            batched_grid_lb,
            batched_grid_ub,
        )

        violating_indices = (e >= l_batch - K)
        v = violating_indices.astype(jnp.int32).sum()

        hard_violating_indices = (e >= l_batch)
        hard_v = hard_violating_indices.astype(jnp.int32).sum()
        
        return v, violating_indices, hard_v, hard_violating_indices

    @partial(jax.jit, static_argnums=(0,))
    def normalize_rsm(self, l, ub_init, domain_min):
        l = l - domain_min
        ub_init = ub_init - domain_min
        # now min = 0
        l = l / jnp.maximum(ub_init, 1e-6)
        # now init max = 1
        return l

    def check_dec_cond(self, lipschitz_k):
        loop_start_time = time.perf_counter()

        dims = self.env.observation_space.shape[0]
        grid_total_size = self.grid_size ** dims
        
        info_dict = {}
        delta = (self.env.observation_space.high[0] - self.env.observation_space.low[0]) / (
            self.grid_size - 1
        )

        # estimate the ub_init and domain_min, used for normalization
        if self.env.observation_space.shape[0] == 2:
            n = 200
        elif self.env.observation_space.shape[0] == 3:
            n = 100
        else:
            n = 50
        _, ub_init = self.compute_bound_init(n)
        domain_min, _ = self.compute_bound_domain(n)

        K = lipschitz_k * delta
        info_dict["delta"] = delta
        info_dict["K"] = K
        print(f"lipschitz_k={lipschitz_k} (without delta)")
        print(f"delta={delta}")
        print(f"K={K} (with delta)")
        print(f"Checking GRID of size {self.grid_size}")

        self.get_pmass_grid()  # cache pmass grid

        K = jnp.float32(K)

        violations = 0
        hard_violations = 0
        expected_l_next = []
        avg_decrease = []
        refinement_buffer = [] # grid needs to be refined
        violation_buffer = []
        hard_violation_buffer = []
        # self.hard_constraint_violation_buffer = None
        # _perf_loop_start = time.perf_counter()

        grid_violating_indices = []
        grid_hard_violating_indices = []

        # stream size should not be too large
        grid_stream_size = min(grid_total_size, self.grid_stream_size)
        total_kernel_time = 0
        total_kernel_iters = 0

        pbar = tqdm(total=grid_total_size // grid_stream_size)
        
        for i in range(0, grid_total_size, grid_stream_size):
            
            idx = jnp.arange(i, i + grid_stream_size)
            sub_grid = jnp.array(self.v_get_grid_item(idx, self.grid_size))
            contains = v_contains(self.env.target_spaces, sub_grid)
            sub_grid = sub_grid[jnp.logical_not(contains)]

            kernel_start = time.perf_counter()
            for start in range(0, sub_grid.shape[0], self.batch_size):
                end = min(start + self.batch_size, sub_grid.shape[0])
                grid_batch = jnp.array(sub_grid[start:end])

                # TODO: later optimize this by filtering the entire stream first
                l_batch = self.learner.l_state.apply_fn(
                    self.learner.l_state.params, grid_batch
                ).flatten()
                normalized_l_batch = self.normalize_rsm(l_batch, ub_init, domain_min)
                less_than_p = normalized_l_batch - K < 1 / (1 - self.reach_prob)
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
                if self.refinement_enabled and v > 0:
                    refinement_buffer.append(grid_batch[violating_indices])
                if hard_v > 0:
                    hard_violation_buffer.append(grid_batch[hard_violating_indices])
            pbar.update(1)

            if i > 0:
                total_kernel_time += time.perf_counter() - kernel_start
                total_kernel_iters += sub_grid.shape[0] // self.batch_size
                ints_per_sec = (
                    total_kernel_iters * self.batch_size / total_kernel_time / 1000
                )
                pbar.set_description(
                    f"kernel_t: {total_kernel_time*1e6/total_kernel_iters:0.2f}us/iter ({ints_per_sec:0.2f} Kints/s)"
                )
            # if self.fail_check_fast and violations > 0:
                # break
        pbar.close()
        print(f"violations={violations}")
        print(f"hard_violations={hard_violations}")

        self.train_buffer.extend(hard_violation_buffer)

        loop_time = time.perf_counter() - loop_start_time
        
        if self.refinement_enabled and hard_violations == 0 and violations > 0:
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
                refinement_buffer, lipschitz_k, delta, ub_init, domain_min
            ):
                print("Refinement successful!")
                return True, 0, info_dict
            else:
                print("Refinement unsuccessful")
        
        info_dict["avg_increase"] = (
            np.mean(avg_decrease) if len(avg_decrease) > 0 else 0
        )
        info_dict["dec_violations"] = f"{violations}/{grid_total_size}"
        info_dict["hard_violations"] = f"{hard_violations}/{grid_total_size}"
        print(f"{violations}/{grid_total_size} violated decrease condition")
        print(f"{hard_violations}/{grid_total_size} hard violations")
        print(f"Train buffer len: {len(self.train_buffer)}")
        if loop_time > 60:
            print(f"Grid runtime={loop_time/60:0.0f} min")
        else:
            print(f"Grid runtime={loop_time:0.2f} s")

        return violations == 0, hard_violations, info_dict

    def refine_grid(
        self, refinement_buffer, lipschitz_k, current_delta, ub_init, domain_min
    ):
        # refinement_buffer = [np.array(r) for r in refinement_buffer]
        # refinement_buffer = np.concatenate(refinement_buffer, axis=0)
        pmass, batched_grid_lb, batched_grid_ub = self.get_pmass_grid()
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
            normalized_l_batch = self.normalize_rsm(l_batch, ub_init, domain_min)
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
