import argparse
import os
import sys
import jax
import jax.random
import jax.numpy as jnp
from tqdm import tqdm

from klax import lipschitz_l1_jax #, triangular
import gymnasium as gym
from gymnasium import spaces
from rl_environments import Vandelpol, LinearLQR, NonPoly1, NonPoly2
from rsm_learner import Learner
from rsm_verifier import Verifier
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RSMLoop:
    def __init__(
        self,
        learner,
        verifier,
        env,
        lip_factor,
        normalize = False,
        rng = jax.random.PRNGKey(333),
        debug = False
    ):
        self.env = env
        self.learner = learner
        self.verifier = verifier
        self.normalize = normalize
        self.debug = debug
        self.lip_factor = lip_factor 
        self.prefill_delta = 0 
        self.rng = rng
        self.iter = 0
        self.info = {} 

    def train(self):
        self.rng, rng = jax.random.split(self.rng)
        train_ds = self.verifier.train_buffer.as_tfds(rng, batch_size=4096)
        current_delta = self.prefill_delta

        start_metrics = None
        num_epochs = 100 

        start_time = time.time()
        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            metrics = self.learner.train_epoch(
                train_ds, current_delta, self.lip_factor
            )
            if start_metrics is None:
                start_metrics = metrics
            pbar.update(1)
            pbar.set_description_str(
                f"Train: loss={metrics['loss']:0.3g}, dec_loss={metrics['dec_loss']:0.3g}, violations={metrics['train_violations']:0.3g}"
            )
        pbar.n = num_epochs
        pbar.refresh()
        pbar.close()
        self.info["ds_size"] = len(self.verifier.train_buffer)

        elapsed = time.time() - start_time
        if elapsed > 60:
            elapsed = f"{elapsed/60:0.1f} minutes"
        else:
            elapsed = f"{elapsed:0.1f} seconds"

        print(
            f"Trained on {len(self.verifier.train_buffer)} samples, start_loss={start_metrics['loss']:0.3g}, end_loss={metrics['loss']:0.3g}, start_violations={start_metrics['train_violations']:0.3g}, end_violations={metrics['train_violations']:0.3g} in {elapsed}"
        )
        # debug
        # self.learner.save("save1")
        # raise KeyError
        return metrics['loss'].astype(np.float32)

    def run(self, timeout):
        start_time = time.time()
        last_saved = time.time()
        self.prefill_delta = self.verifier.prefill_train_buffer()
        
        best_reach_prob = 0
        best_params = None

        while True:
            runtime = time.time() - start_time
            if runtime > timeout:
                print("Timeout!")
                # self.learner.save(f"saved/{self.env.name}_loop.jax")
                return False, best_reach_prob, best_params
            
            print(f"\n### Iteration {self.iter} ({runtime // 60:0.0f}:{runtime % 60:02.0f} elapsed) ###")
            
            self.train()
            
            if self.normalize:
                K_f = self.env.lipschitz_constant_normalize
            else:
                K_f = self.env.lipschitz_constant
            K_l = lipschitz_l1_jax(self.learner.l_state.params).item()
            lipschitz_k = float(K_l * K_f)
    
            self.info["lipschitz_k"] = lipschitz_k
            self.info["K_f"] = K_f
            self.info["K_l"] = K_l
            self.info["iter"] = self.iter
            self.info["runtime"] = runtime

            sat, hard_violations, info = self.verifier.check_dec_cond(lipschitz_k)
            for k, v in info.items():
                self.info[k] = v
            print("info=", str(self.info), flush=True)
            
            # break

            # if the expected decrease condition is satisfied
            # then check other conditions
            # o.w. refine the grid
            if sat:
                print("Expected decrease condition fulfilled!")
                # self.learner.save(f"saved/{self.env.name}_loop.jax")
                # print("[SAVED]")
                
                if self.env.observation_space.shape[0] == 2:
                    n = 200
                elif self.env.observation_space.shape[0] == 3:
                    n = 100
                else:
                    n = 50
                lb_domain, _ = self.verifier.compute_bound_domain(n)

                if self.normalize:
                    _, ub_init = self.verifier.compute_bound_l(self.env.init_spaces)
                    lb_unsafe, _ = self.verifier.compute_bound_l(self.env.unsafe_spaces)
                else:
                    _, ub_init = self.verifier.compute_bound_init(n)
                    lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
                

                print(f"Init   max = {ub_init:0.6g}")
                print(f"Unsafe min = {lb_unsafe:0.6g}")
                print(f"domain min = {lb_domain:0.6g}")
                self.info["ub_init"] = ub_init
                self.info["lb_unsafe"] = lb_unsafe
                self.info["lb_domain"] = lb_domain
                bound_correct = True
                if lb_unsafe < ub_init:
                    bound_correct = False
                    print(
                        "RSM is lower at unsafe than in init. No probabilistic guarantees can be obtained."
                    )
                else:                
                    # normalize to min = 0
                    ub_init = ub_init - lb_domain
                    lb_unsafe = lb_unsafe - lb_domain
                    # normalize to init=1
                    lb_unsafe = lb_unsafe / ub_init
                    actual_reach_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                    self.info["actual_reach_prob"] = actual_reach_prob
                    if not bound_correct:
                        self.info["actual_reach_prob"] = "UNSAFE"
                    print(
                        f"Probability of reaching the target safely is at least {actual_reach_prob*100:0.3f}% (higher is better)"
                    )
                    if actual_reach_prob > best_reach_prob:
                        best_reach_prob = actual_reach_prob
                        best_params = self.learner.l_state.params
                        self.learner.save(f"plots/{self.env.name}.jax")
                    

                    if (best_reach_prob >= self.verifier.reach_prob) or (actual_reach_prob < best_reach_prob-0.03):
                        runtime = time.time() - start_time
                        print(f"Total Time: ({runtime // 60:0.0f}:{runtime % 60:02.0f} elapsed) ###")
                        return True, best_reach_prob, best_params

            
            if hard_violations == 0 and self.iter > 4 and self.iter % 2 == 0: #???
                print("Refining grid")
                if self.env.observation_space.shape[0] == 2:
                    self.verifier.grid_size *= 2
                elif self.env.observation_space.shape[0] == 3:
                    self.verifier.grid_size *= int(1.5 * self.verifier.grid_size)
                else:
                    self.verifier.grid_size = int(1.35 * self.verifier.grid_size)

            runtime = time.time() - start_time
            print(f"Total Time: ({runtime // 60:0.0f}:{runtime % 60:02.0f} elapsed)")

            sys.stdout.flush()
            # if True:
                # self.plot_l(f"loop/{self.env.name}_{self.iter:04d}.png")
            self.iter += 1

    def rollout(self):
        '''
        generate a trace of the system
        '''
        
        self.rng, rng1, rng2, rng3 = jax.random.split(self.rng, 4)
        index = jax.random.randint(rng1, shape=(), minval=0, maxval=len(self.env.init_spaces))
        state  = jax.random.uniform(
                rng2,
                (self.env.observation_space.shape[0]),
                minval=np.maximum(self.env.init_spaces[index].low, -10000),
                maxval=np.minimum(self.env.init_spaces[index].high, 10000),
            )
        trace = [np.array(state)]
        rngs = jax.random.split(rng3, 200)
        for i in range(200):
            state = self.env.next(state)
            state = self.env.add_noise(state,rngs[i])
            trace.append(np.array(state))
        return np.stack(trace, axis=0)

    def plot_l(self, filename, rollout=True):
        if self.env.observation_space.shape[0] > 2:
            print("Cannot plot when > 2 dimensions")
            return
        
        grid, _, _ = self.verifier.get_unfiltered_grid(n=50)
        l = self.learner.l_state.apply_fn(self.learner.l_state.params, grid).flatten()
        l = np.array(l)

        # sns.set()
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(grid[:, 0], grid[:, 1], marker="s", c=l, zorder=1, alpha=0.7)
        fig.colorbar(sc)
        ax.set_title(f"L at iter {self.iter} for {self.env.name}")

        if rollout:
            terminals_x, terminals_y = [], []
            for i in range(5):
                trace = self.rollout()
                ax.plot(
                    trace[:, 0],
                    trace[:, 1],
                    color=sns.color_palette()[0],
                    zorder=2,
                    alpha=0.3,
                )
                ax.scatter(
                    trace[:, 0],
                    trace[:, 1],
                    color=sns.color_palette()[0],
                    zorder=2,
                    marker=".",
                )
                terminals_x.append(float(trace[-1, 0]))
                terminals_y.append(float(trace[-1, 1]))
            ax.scatter(terminals_x, terminals_y, color="white", marker="x", zorder=5)
        
        for init in self.env.init_spaces:
            x = [
                init.low[0],
                init.high[0],
                init.high[0],
                init.low[0],
                init.low[0],
            ]
            y = [
                init.low[1],
                init.low[1],
                init.high[1],
                init.high[1],
                init.low[1],
            ]
            ax.plot(x, y, color="cyan", alpha=0.5, zorder=7)
        for unsafe in self.env.unsafe_spaces:
            x = [
                unsafe.low[0],
                unsafe.high[0],
                unsafe.high[0],
                unsafe.low[0],
                unsafe.low[0],
            ]
            y = [
                unsafe.low[1],
                unsafe.low[1],
                unsafe.high[1],
                unsafe.high[1],
                unsafe.low[1],
            ]
            ax.plot(x, y, color="red", alpha=0.5, zorder=7)
        for target in self.env.target_spaces:
            x = [
                target.low[0],
                target.high[0],
                target.high[0],
                target.low[0],
                target.low[0],
            ]
            y = [
                target.low[1],
                target.low[1],
                target.high[1],
                target.high[1],
                target.low[1],
            ]
            ax.plot(x, y, color="magenta", alpha=0.5, zorder=7)

        xl = np.maximum(self.env.observation_space.low[0], -5.0)
        xu = np.minimum(self.env.observation_space.high[0], 5.0)
        yl = np.maximum(self.env.observation_space.low[1], -5.0)
        yu = np.minimum(self.env.observation_space.high[1], 5.0)
        ax.set_xlim([xl, xu])
        ax.set_ylim([yl, yu])
        # ax.set_xlim([self.env.observation_space.low[0], self.env.observation_space.high[0]])
        # ax.set_ylim([self.env.observation_space.low[1], self.env.observation_space.high[1]])
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='System Normalization for Neural Certificates',
        description='Learning a reach-avoid supermartingale neural network with normalization techniques'
    )

    # problem formulation    
    parser.add_argument("--env", default="Vandelpol", help='dynamical system')
    parser.add_argument("--difficulty", default=1, type=int, help='difficulty of the problem, set to INF if > 100')
    parser.add_argument("--timeout", default=60, type=int, help='max time limit in minutes') 
    parser.add_argument("--reach_prob", default=0.8, type=float, help='reach-avoid probability')
    
    # learner 
    parser.add_argument("--hidden", default=16, type=int, help='hidden neurons in each layer')
    parser.add_argument("--num_layers", default=2, type=int, help='number of hidden layers')
    parser.add_argument("--batch_size", default=512, type=int, help='batch size in training and verification')
    parser.add_argument("--continue_rsm", action='store_true', help='use an existing network')
    
    # verifier
    parser.add_argument("--eps", default=0.05, type=float, help='epsilon in the RASM condition') #0.005
    parser.add_argument("--lip", default=0.1, type=float, help='regularization term for lipschitz constant') 
    parser.add_argument("--l_lip", default=1.0, type=float, help='target lipschitz constant of the neural network') 
    parser.add_argument("--grid_size", default=200, type=int, help='grid size for verification')
    
    parser.add_argument("--normalize", action='store_true', help='Nomalize the system')
    parser.add_argument("--debug", action='store_true', help='Debug mode')
    parser.add_argument("--seed", default=2025, type=int, help='random seed')

    args = parser.parse_args()
    
    if args.difficulty > 100:
        args.difficulty = np.inf

    if args.env == 'Vandelpol':
        env = Vandelpol(difficulty=args.difficulty)
    elif args.env == 'NonPoly1':
        env = NonPoly1(difficulty=args.difficulty)
    elif args.env == 'NonPoly2':
        env = NonPoly2(difficulty=args.difficulty)
    elif args.env == 'LinearLQR':
        env = LinearLQR()
    else:
        raise ValueError(f'Unknown environment "{args.env}"')
    
    if args.normalize:
        dim = env.observation_space.shape[0]
        env.observation_space = gym.spaces.Box(
            low = np.array([-np.inf]*dim, dtype=jnp.float32),  
            high = np.array([np.inf]*dim, dtype=jnp.float32),
            dtype = jnp.float32)
        
    print(f'Dynamical System: {args.env} (difficulty={args.difficulty})')
    print(f'System Normaliztion: {args.normalize}')
    # print(f'Observation Space: {env.observation_space}')
    # print(f'Initial Space: {env.init_spaces}')

    os.makedirs("saved", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("loop", exist_ok=True)
    
    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, 3)

    learner = Learner(
        env=env,
        l_hidden=[args.hidden] * args.num_layers,
        l_lip=args.l_lip,
        eps=args.eps,
        reach_prob=args.reach_prob,
        softplus_l_output=True,
        normalize = args.normalize,
        rng = rngs[0],
        debug = args.debug,
    )
    
    verifier = Verifier(
        learner=learner,
        env=env,
        batch_size=args.batch_size,
        reach_prob=args.reach_prob,
        grid_size=args.grid_size,
        normalize=args.normalize,
        rng = rngs[1],
        debug = args.debug,
    )

    if args.continue_rsm:
        print("Loading existing neural network.")
        learner.load(f"saved/{args.env}_loop.jax")

    loop = RSMLoop( 
        learner,
        verifier,
        env,
        lip_factor=args.lip,
        normalize=args.normalize,
        rng = rngs[2],
        debug = args.debug,
    )

    loop.plot_l(f"plots/{args.env}_{args.difficulty}_start.png")
    
    sat, best_reach_prob, best_params = loop.run(args.timeout * 60)
    print("SAT=", sat)
    print("Reach Prob=", best_reach_prob)

    # loop.plot_l(f"plots/{args.env}_{args.difficulty}_end.png")
  