import argparse
import os
import sys
import jax
import jax.numpy as jnp
import jax.random
from gymnasium import spaces
from tqdm import tqdm

from klax import lipschitz_l1_jax, triangular
from rl_environments import Vandelpol
# from rl_environments import LDSEnv, InvertedPendulum, CollisionAvoidanceEnv, Vandelpol
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
        plot,
        jitter_grid,#???
        soft_constraint, #???
        train_p=True, #???
    ):
        self.env = env
        self.learner = learner
        self.verifier = verifier

        self.soft_constraint = soft_constraint #???
        self.jitter_grid = jitter_grid #???

        self.lip_factor = lip_factor 
        
        self.plot = plot
        self.prefill_delta = 0 #???
        self.iter = 0 # number of CEGIS iterations
        self.info = {} 

    def train(self):
        if self.jitter_grid:
            if self.env.observation_space.shape[0] == 2:
                n = 200
            elif self.env.observation_space.shape[0] == 3:
                n = 100
            else:
                n = 50
            print("create grid.")
            train_ds, stepsize = self.verifier.get_domain_jitter_grid(n)#???
            current_delta = stepsize
        else:
            print("use train buffer.")
            train_ds = self.verifier.train_buffer.as_tfds(batch_size=4096)#???
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

    def run(self, timeout):
        start_time = time.time()
        last_saved = time.time()
        self.prefill_delta = self.verifier.prefill_train_buffer()
        
        while True:
            runtime = time.time() - start_time
            if runtime > timeout:
                print("Timeout!")
                self.learner.save(f"saved/{self.env.name}_loop.jax")
                return False
            if time.time() - last_saved > 60 * 60:
                # save every hour
                last_saved = time.time()
                self.learner.save(f"saved/{self.env.name}_loop.jax")
                print("[SAVED]")
            
            print(f"\n## Iteration {self.iter} ({runtime // 60:0.0f}:{runtime % 60:02.0f} elapsed) ##")
            
            self.train()
        
            K_f = self.env.lipschitz_constant
            K_l = lipschitz_l1_jax(self.learner.l_state.params).item()
            lipschitz_k = float(K_l * K_f) # to check
    
            self.info["lipschitz_k"] = lipschitz_k
            self.info["K_f"] = K_f
            self.info["K_l"] = K_l
            self.info["iter"] = self.iter
            self.info["runtime"] = runtime

            sat, hard_violations, info = self.verifier.check_dec_cond(lipschitz_k)
            for k, v in info.items():
                self.info[k] = v
            print("info=", str(self.info), flush=True)
            
            # if the expected decrease condition is satisfied
            # then check other conditions
            # o.w. refine the grid
            if sat:
                print("Expected decrease condition fulfilled!")
                self.learner.save(f"saved/{self.env.name}_loop.jax")
                print("[SAVED]")
                
                if self.env.observation_space.shape[0] == 2:
                    n = 200
                elif self.env.observation_space.shape[0] == 3:
                    n = 100
                else:
                    n = 50
                _, ub_init = self.verifier.compute_bound_init(n)
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
                lb_domain, _ = self.verifier.compute_bound_domain(n)
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
                if ( self.soft_constraint or actual_reach_prob >= self.verifier.reach_prob):
                    return True
            
            if hard_violations == 0 and self.iter > 4 and self.iter % 2 == 0: #???
                print("Refining grid")
                if self.env.observation_space.shape[0] == 2:
                    self.verifier.grid_size *= 2
                elif self.env.observation_space.shape[0] == 3:
                    self.verifier.grid_size *= int(1.5 * self.verifier.grid_size)
                else:
                    self.verifier.grid_size = int(1.35 * self.verifier.grid_size)

            sys.stdout.flush()
            if self.plot:
                self.plot_l(f"loop/{self.env.name}_{self.iter:04d}.png")
            self.iter += 1

    def rollout(self):
        '''
        generate a trace of the system
        '''
        rng = np.random.default_rng().integers(0, 10000)
        rng = jax.random.PRNGKey(rng)
        rngs = jax.random.split(rng, 200)

        index = jax.random.randint(rng, shape=(), minval=0, maxval=len(self.env.init_spaces))
        state = self.env.init_spaces[index].sample()
    
        trace = [np.array(state)]
        for i in range(200):
            state = self.env.next(state)
            state = self.env.add_noise(state,rngs[i])
            trace.append(np.array(state))
        # print(trace)
        return np.stack(trace, axis=0)

    def plot_l(self, filename):
        if self.env.observation_space.shape[0] > 2:
            print("Cannot plot in more than 2 dimensions")
            return
        
        grid, _, _ = self.verifier.get_unfiltered_grid(n=50)
        l = self.learner.l_state.apply_fn(self.learner.l_state.params, grid).flatten()
        l = np.array(l)

        # sns.set()
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(grid[:, 0], grid[:, 1], marker="s", c=l, zorder=1, alpha=0.7)
        fig.colorbar(sc)
        ax.set_title(f"L at iter {self.iter} for {self.env.name}")

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
        # if self.verifier.hard_constraint_violation_buffer is not None:
        #     print(
        #         "self.verifier.hard_constraint_violation_buffer: ",
        #         self.verifier.hard_constraint_violation_buffer[0:10],
        #     )
        #     ax.scatter(
        #         self.verifier.hard_constraint_violation_buffer[:, 0],
        #         self.verifier.hard_constraint_violation_buffer[:, 1],
        #         color="green",
        #         marker="s",
        #         alpha=0.7,
        #         zorder=6,
        #     )
        # if self.verifier._debug_violations is not None:
        #     ax.scatter(
        #         self.verifier._debug_violations[:, 0],
        #         self.verifier._debug_violations[:, 1],
        #         color="cyan",
        #         marker="s",
        #         alpha=0.7,
        #         zorder=6,
        #     )
        
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
            ax.plot(x, y, color="magenta", alpha=0.5, zorder=7)
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
            ax.plot(x, y, color="green", alpha=0.5, zorder=7)
        
        # x = [
        #     self.env.safe_space.low[0],
        #     self.env.safe_space.high[0],
        #     self.env.safe_space.high[0],
        #     self.env.safe_space.low[0],
        #     self.env.safe_space.low[0],
        # ]
        # y = [
        #     self.env.safe_space.low[1],
        #     self.env.safe_space.low[1],
        #     self.env.safe_space.high[1],
        #     self.env.safe_space.high[1],
        #     self.env.safe_space.low[1],
        # ]
        # ax.plot(x, y, color="green", alpha=0.5, zorder=7)

        ax.set_xlim([self.env.observation_space.low[0], self.env.observation_space.high[0]])
        ax.set_ylim([self.env.observation_space.low[1], self.env.observation_space.high[1]])
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='System Normalization for Neural Certificates',
        description='Learning a reach-avoid supermartingale neural network with normalization techniques',
        epilog='By Musan (Hao Wu) @ SKLCS, Institute of Software, UCAS'
    )
    # problem formulation    
    parser.add_argument("--env", default="vandelpol", help='control system')
    parser.add_argument("--timeout", default=60, type=int, help='max time limit in minutes') 
    parser.add_argument("--reach_prob", default=0.8, type=float, help='reach-avoid probability')
    
    # neural network and training
    parser.add_argument("--hidden", default=128, type=int, help='hidden neurons in each layer')
    parser.add_argument("--num_layers", default=2, type=int, help='number of hidden layers')

    # learner 
    parser.add_argument("--continue_rsm", type=int, default=0, help='use an existing network')
    
    # verifier
    parser.add_argument("--eps", default=0.05, type=float) # ???
    parser.add_argument("--lip", default=0.01, type=float) # ???
    # parser.add_argument("--p_lip", default=0.0, type=float) 
    parser.add_argument("--l_lip", default=4.0, type=float) # ???
    parser.add_argument("--fail_check_fast", type=int, default=0)
    parser.add_argument("--grid_factor", default=1.0, type=float)

    parser.add_argument("--batch_size", default=512, type=int)
    # parser.add_argument("--ppo_iters", default=50, type=int)
    # parser.add_argument("--policy", default="policies/lds0_zero.jax")

    # parser.add_argument("--train_p", type=int, default=1)
    parser.add_argument("--square_l_output", default=True)
    parser.add_argument("--jitter_grid", type=int, default=0)
    parser.add_argument("--soft_constraint", type=int, default=1)
    parser.add_argument("--gamma_decrease", default=1.0, type=float)

    parser.add_argument("--debug_k0", action="store_true")
    parser.add_argument("--gen_plot", action="store_true")
    parser.add_argument("--no_refinement", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--small_mem", action="store_true")


    args = parser.parse_args()
    if args.env == 'vandelpol':
        env = Vandelpol()
    else:
        raise ValueError(f'Unknown environment "{args.env}"')
    print(f'Control System: {args.env}')
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("saved", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("loop", exist_ok=True)
    
    learner = Learner(
        env=env,
        l_hidden=[args.hidden] * args.num_layers,
        l_lip=args.l_lip,
        eps=args.eps,
        # gamma_decrease=args.gamma_decrease,
        reach_prob=args.reach_prob,
        softplus_l_output=args.square_l_output,
    )
    
    verifier = Verifier(
        learner=learner,
        env=env,
        batch_size=args.batch_size,
        reach_prob=args.reach_prob,
        fail_check_fast=bool(args.fail_check_fast),
        grid_factor=args.grid_factor,
        small_mem=args.small_mem,
    )

    if args.continue_rsm > 0:
        learner.load(f"saved/{args.env}_loop.jax")
        verifier.grid_size *= args.continue_rsm #refine grid

    loop = RSMLoop( 
        learner,
        verifier,
        env,
        lip_factor=args.lip,
        plot=args.plot,
        jitter_grid=bool(args.jitter_grid),#???
        soft_constraint=bool(args.soft_constraint),#???
    )

    loop.plot_l(f"plots/{args.env}_start.png")
    
    sat = loop.run(args.timeout * 60)
    print("SAT=", sat)

    loop.plot_l(f"plots/{args.env}_end.png")

    # with open("info.log", "a") as f:
    #     f.write("args=" + str(vars(args)) + "\n")
    #     f.write("sat=" + str(sat) + "\n")
    #     f.write("info=" + str(loop.info) + "\n\n\n") 7