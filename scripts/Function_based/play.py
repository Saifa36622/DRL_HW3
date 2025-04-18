# """Script to train RL agent."""

# """Launch Isaac Sim Simulator first."""

# import argparse
# import sys
# import os

# from isaaclab.app import AppLauncher

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from RL_Algorithm.Function_Aproximation.DQN import DQN

# from tqdm import tqdm

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# args_cli, hydra_args = parser.parse_known_args()

# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # clear out sys.argv for Hydra
# sys.argv = [sys.argv[0]] + hydra_args

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import gymnasium as gym
# import torch
# from datetime import datetime
# import random

# from RL_Algorithm.Function_based.DQN import DQN

# from RL_Algorithm.Function_based.Linear_Q import Linear_QN

# from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE

# from RL_Algorithm.Function_based.AC import Actor_Critic

# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# from itertools import count
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np


# from isaaclab.envs import (
#     DirectMARLEnv,
#     DirectMARLEnvCfg,
#     DirectRLEnvCfg,
#     ManagerBasedRLEnvCfg,
#     multi_agent_to_single_agent,
# )
# # from omni.isaac.lab.utils.dict import print_dict
# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# from isaaclab_tasks.utils.hydra import hydra_task_config

# # Import extensions to set up environment tasks
# import CartPole.tasks  # noqa: F401

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False

# steps_done = 0

# @hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
# def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
#     """Train with stable-baselines agent."""
#     # randomly sample a seed if seed = -1
#     if args_cli.seed == -1:
#         args_cli.seed = random.randint(0, 10000)

#     # override configurations with non-hydra CLI arguments
#     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

#     # set the environment seed
#     # note: certain randomizations occur in the environment initialization so we set the seed here
#     env_cfg.seed = agent_cfg["seed"]
#     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

#     # create isaac environment
#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

#     # ==================================================================== #
#     # ========================= Can be modified ========================== #

#     # hyperparameters
#     num_of_action = 7
#     action_range = [-15, 15]  

#     learning_rate = 0.3

#     hidden_dim = None
#     n_episodes = 10000
#     initial_epsilon = 1.0
#     epsilon_decay = 0.9997  
#     final_epsilon = 0.01
#     discount = 0.5

#     buffer_size = None
#     batch_size = None


#     # set up matplotlib
#     is_ipython = 'inline' in matplotlib.get_backend()
#     if is_ipython:
#         from IPython import display

#     plt.ion()

#     # if GPU is to be used
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else
#         "mps" if torch.backends.mps.is_available() else
#         "cpu"
#     )

#     print("device: ", device)

#     agent = DQN(
#         device=device,
#         num_of_action=num_of_action,
#         action_range=action_range,
#         learning_rate=learning_rate,
#         hidden_dim=hidden_dim,
#         initial_epsilon = initial_epsilon,
#         epsilon_decay = epsilon_decay,
#         final_epsilon = final_epsilon,
#         discount_factor = discount,
#         buffer_size = buffer_size,
#         batch_size = batch_size,
#     )

#     task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
#     Algorithm_name = "DQN"  
#     episode = 0
#     q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.json"
#     full_path = os.path.join(f"w/{task_name}", Algorithm_name)
#     agent.load_w(full_path, q_value_file)

#     # reset environment
#     obs, _ = env.reset()
#     timestep = 0
#     # simulate environment
#     while simulation_app.is_running():
#         # run everything in inference mode
#         with torch.inference_mode():
        
#             for episode in range(n_episodes):
#                 obs, _ = env.reset()
#                 done = False

#                 while not done:
#                     # agent stepping
#                     action, action_idx = agent.get_action(obs)

#                     # env stepping
#                     next_obs, reward, terminated, truncated, _ = env.step(action)

#                     done = terminated or truncated
#                     obs = next_obs
            
#         if args_cli.video:
#             timestep += 1
#             # Exit the play loop after recording one video
#             if timestep == args_cli.video_length:
#                 break

#         break
#     # ==================================================================== #

#     # close the simulator
#     env.close()

# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()


"""Script to play RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN
from RL_Algorithm.Function_based.Linear_Q import Linear_QN
from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
from RL_Algorithm.Function_based.AC import Actor_Critic

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main():
    """Play trained RL agent."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Set parameters (match training)
    num_of_action = 7
    action_range = [-12.0, 12.0]
    learning_rate = 3e-4
    hidden_dim = 256
    discount = 0.99
    initial_epsilon = 0.01
    epsilon_decay = 0.9997
    final_epsilon = 0.01
    buffer_size = 10000
    batch_size = 256

    # MC
    n_observations = 4
    dropout = 0.3

    # AC
    tau = 0.005

    # DQN Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Algorithm_name = "Linear_QN"
    name_plot = "LQ_Normal"

    if Algorithm_name == "DQN" :

        agent = DQN(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            initial_epsilon = initial_epsilon,
            epsilon_decay = epsilon_decay,
            final_epsilon = final_epsilon,
            discount_factor = discount,
            buffer_size = buffer_size,
            batch_size = batch_size,
        )

    if Algorithm_name == "Linear_QN" :

        agent = Linear_QN(
            # device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            # hidden_dim=hidden_dim,
            initial_epsilon = initial_epsilon,
            epsilon_decay = epsilon_decay,
            final_epsilon = final_epsilon,
            discount_factor = discount,
            
            # buffer_size = buffer_size,
            # batch_size = batch_size,

        )
        max_steps = 1000

    
    if Algorithm_name == "MC" :

        agent = MC_REINFORCE(
            # device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            dropout=dropout,
            discount_factor = discount,
            # buffer_size = buffer_size,
            # batch_size = batch_size,

        )
        max_steps = 1000

    if Algorithm_name == "AC" :

        agent = Actor_Critic(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            dropout=dropout,
            discount_factor = discount,
            buffer_size = buffer_size,
            batch_size = batch_size,
            tau=tau,

        )

    task_name = str(args_cli.task).split('-')[0]
    Algorithm_name = "DQN"
    name_plot = "LQ_Normal"
    model_episode = 25000  # Update this to match saved model
    weight_file = f"{Algorithm_name}_{model_episode}_{num_of_action}_{action_range[1]}.pth"
    model_path = os.path.join(f"{task_name}", Algorithm_name, name_plot, weight_file)

    if Algorithm_name == "DQN" :
        agent.load_w_DQN(model_path)

    if Algorithm_name == "MC" :
        agent.load_model(model_path)

    if Algorithm_name == "AC" :
        agent.load_model(model_path)

    if Algorithm_name == "Linear_QN" :
        agent.load_w(model_path)


    # reset environment
    obs, _ = env.reset()
    timestep = 0
    n_episodes = 10  # only a few episodes needed to play

    while simulation_app.is_running():

        with torch.inference_mode():

            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                observer_log = []

                while not done:
                    policy_tensor = obs["policy"]
                    policy_values = policy_tensor.cpu().numpy().flatten()

                    cart_pos     = policy_values[0]
                    pole_angle   = policy_values[1]
                    cart_vel     = policy_values[2]
                    pole_vel     = policy_values[3]

                    observer_log.append({
                        "cart_pos": float(cart_pos),
                        "pole_angle": float(pole_angle),
                        "cart_vel": float(cart_vel),
                        "pole_vel": float(pole_vel),
                    })
                    if Algorithm_name == "DQN" :
                        agent.learn(env)
                        agent.decay_epsilon()

                    if Algorithm_name == "Linear_QN" :
                        agent.learn(env,max_steps)
                        # agent.sum_count += agent.count

                    if Algorithm_name == "MC" :
                        agent.learn(env)
                        # sum_loss += loss
                        # sum_ep += episode_return

                    if Algorithm_name == "AC" :
                        episode_return = agent.learn(env,1000,1)
                        # sum_ep += episode_return

                # save log
                log_dir = os.path.join("observer_log", Algorithm_name, name_plot)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"{Algorithm_name}_{name_plot}.json")
                with open(log_path, "w") as f:
                    json.dump(observer_log, f, indent=4)
                print(f"Saved observer log to {log_path}")

                break  # comment this line to play more episodes

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
