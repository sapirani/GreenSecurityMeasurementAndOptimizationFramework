import argparse
import os
import time
import numpy as np
import torch
import TD

def train_offline(RL_agent, env, eval_env, args):
    # Reward
    evals = []
    biases = []
    variances = []

    # Time
    times = []

    # Loss
    losses = []

    # Load offline dataset.
    RL_agent.conscious_replay.load_D4RL(d4rl.qlearning_dataset(env))
    start_time = time.time()

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, biases, variances, times, losses, t, start_time, args)

        # Train.
        RL_agent.train()

# Train online RL agent.
def train_online(RL_agent, env, eval_env, args):
    # Reward
    evals = []
    biases = []
    variances = []

    # Time
    times = []

    # Loss
    losses = []

    # Initialize
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset()[0], False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        RL_agent.t += 1
        maybe_evaluate_and_print(RL_agent, eval_env, evals, biases, variances, times, losses, t, start_time, args)

        # Select action.
        if allow_train:
            action = RL_agent.select_action(np.array(state), deterministic=False)
        else:
            action = env.env.action_space.sample()

        # Do a step.
        next_state, reward, done, trunc, _ = env.step({"action": action, "robust_type": args.noise_factor, "robust_config": args})

        ep_total_reward += reward
        ep_timesteps += 1
        ep_finished = float(done or trunc)

        # Store tuple.
        RL_agent.conscious_replay.add(torch.tensor(state).unsqueeze(0), torch.tensor(action), torch.tensor(next_state), reward, reward, 0, done)

        state = next_state

        if allow_train:
            for _ in range(args.UTD):
                # Train.
                RL_agent.train()

        if ep_finished:
            if t >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset()[0], False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1


# Logs.
def maybe_evaluate_and_print(RL_agent, eval_env, evals, biases, variances, times, losses, t, start_time, args):
    if t % args.eval_freq == 0:
        # Rewards
        q_values = np.zeros(args.eval_eps)
        discounted_reward = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            state = eval_env.reset()

            if args.offline == 0:
                state = state[0]

            done = False
            action = RL_agent.select_action(state, deterministic=True)

            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).to(RL_agent.args.device).unsqueeze(0)
                action = torch.tensor(action, dtype=torch.float).to(RL_agent.args.device).unsqueeze(0)

                fixed_target_zs = RL_agent.fixed_encoder_target.zs(state)
                fixed_target_zsa = RL_agent.fixed_encoder_target.zsa(fixed_target_zs, action)

                Q_target = RL_agent.critic_target(state, action, fixed_target_zsa, fixed_target_zs).min(1, keepdim=True)[0]

            q_values[ep] = Q_target

            step = 0

            # Episode
            while not done:
                # Action selection.
                action = RL_agent.select_action(state, deterministic=True)

                # Step.
                if args.offline == 0:
                    state, reward, done, trunc, _ = eval_env.step({"action": action, "robust_type": args.noise_factor, "robust_config": args})

                    done = done or trunc
                else:
                    state, reward, done, _ = eval_env.step(action)

                # Reward sum.
                discounted_reward[ep] += reward * RL_agent.args.discount ** step

                step += 1

        # Time
        time_total = (time.time() - start_time) / 60

        # Reward
        score = discounted_reward.mean().item()
        q_score = q_values.mean().item()
        bias = torch.tensor(score - q_score).abs().item()
        variance = discounted_reward.std().item()

        # Loss
        loss_tot = RL_agent.estimate_loss(replay=RL_agent.conscious_replay)

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tScore: {score:,.1f}\tQ-Score: {q_score:,.1f}\tBias: {bias:,.1f}\t"
              f"Variance: {variance:,.1f}\t"
              f"Loss: {loss_tot:,.1f}")

        # Reward
        evals.append(score)
        biases.append(bias)
        variances.append(variance)

        # Time
        times.append(time_total)

        # Loss
        losses.append(loss_tot)

        # file.
        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}\n{times}\n{losses}\n{biases}\n{variances}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Algorithm.
    parser.add_argument("--policy", default="DDPG", type=str)
    parser.add_argument('--offline', default=0, type=int)

    # Exploration.
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--exploration_noise", default=.1, type=float)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--UTD", default=1, type=int)

    parser.add_argument("--W", default=5, type=int)
    parser.add_argument("--rho", default=0.7, type=float)

    parser.add_argument("--buffer_size", default=1e6, type=int)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)

    # Environment.
    parser.add_argument("--env", default="Humanoid-v5", type=str)
    # InvertedDoublePendulum, InvertedPendulum, Reacher, Pusher, HumanoidStandup, Humanoid, Swimmer, Hopper, Walker2d, HalfCheetah, Ant

    parser.add_argument("--noise_factor", default="action", type=str)
    # "robust_force", "robust_shape", "action"

    parser.add_argument("--noise_type", default="gauss", type=str)
    # "uniform"

    parser.add_argument("--noise_mu", default=0, type=float)
    parser.add_argument("--noise_sigma", default=0, type=float)

    # Evaluation
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)

    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = f"{args.policy}_{args.seed}"

    # Offline.
    if args.offline == 1:
        import d4rl
        import gym

        d4rl.set_dataset_path(args.d4rl_path)
        args.use_checkpoints = False

        # environment
        env = gym.make(args.env)
        eval_env = gym.make(args.env)
    else:
        import sys

        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # strip all args

        import robust_gymnasium as gym  # safe import

        env = gym.make(args.env)
        eval_env = gym.make(args.env)

    if not os.path.exists(f"./results/{args.env}"):
        os.makedirs(f"./results/{args.env}")

    # Seed.
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, N: {args.N}, Environment: {args.env}, Seed: {args.seed}, "
          f"Device: {RL_agent.device}")
    print("---------------------------------------")

    # Optimize.
    if args.offline == 1:
        train_offline(RL_agent, env, eval_env, args)
    else:
        train_online(RL_agent, env, eval_env, args)