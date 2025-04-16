from __future__ import annotations
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm
import wandb

class Linear_QN(BaseAlgorithm):
    def __init__(
        self,
        num_of_action: int = 7,        # e.g. 7 discrete actions
        action_range: list = [-2.5, 2.5],
        learning_rate: float = 0.01,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,

        

    ) -> None:
        """
        Initialize a Q-learning agent with linear function approximation,
        but discretize 7 actions in [-2.5..+2.5].
        """
        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            count = 0,
            sum_count= 0
            
        )
        
        last_discrete_action_idx :int = 0
        # We'll store the last discrete action index (0..6)
        # so that we can update Q-values properly.

    def scale_action(self, action_idx: int) -> torch.Tensor:
        """
        Maps an integer action index (0..num_of_action-1)
        to a continuous value in [action_min, action_max].

        Returns:
            torch.Tensor of shape (1,1) containing the continuous action,
            e.g. [[-2.5]] or [[1.7]] on the same device as environment.
        """
        action_min, action_max = self.action_range
        num_bins = self.num_of_action - 1  # e.g. 6 if num_of_action=7

        # If action_idx=0 => scaled_value= action_min
        # If action_idx=num_bins => scaled_value= action_max
        scaled_value = action_min + (action_idx / num_bins) * (action_max - action_min)

        # Return a (1,1) float tensor. Adjust device if needed.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor([[scaled_value]], dtype=torch.float32, device=device)

    def select_action(self, obs: dict) -> torch.Tensor:
        """
        Epsilon-greedy selection over discrete actions {0..num_of_action-1}.
        Then convert that discrete index to a continuous action for the environment.

        Returns:
            A (1,1) float32 tensor in [-2.5, +2.5] suitable for env.step(...).
        """
        if np.random.rand() < self.epsilon:
            # Explore: pick random discrete index
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            # Exploit: pick action with max Q(s,a)
            x_s = obs["policy"].cpu().numpy().flatten()   # shape (state_dim,)
            # w.shape = (state_dim, num_of_action)
            q_vals = np.dot(x_s, self.w)                  # shape (num_of_action,)
            action_idx = int(np.argmax(q_vals))           # best discrete action

        # Remember this discrete index for the Q-update
        self.last_discrete_action_idx = action_idx

        # Convert discrete index to a continuous action for the environment
        return self.scale_action(action_idx)

    def update(
        self,
        obs: dict,
        discrete_action_idx: int,
        reward: float | torch.Tensor,
        next_obs: dict,
        terminated: bool
    ):
        """
        Q-learning update of the weight vector, using discrete_action_idx in 0..(num_of_action-1).
        """

        # If reward is a single GPU tensor, convert it to float
        if isinstance(reward, torch.Tensor):
            reward = float(reward.cpu().item())

        # Convert the current state to a NumPy vector
        x_s = obs["policy"].cpu().numpy().flatten()  # shape (state_dim,)

        # Current Q(s,a)
        current_q = np.dot(x_s, self.w[:, discrete_action_idx])

        # If episode is done, no bootstrap
        if terminated:
            best_next_q = 0.0
        else:
            x_s_next = next_obs["policy"].cpu().numpy().flatten()
            # Q(s', a') for a' in 0..num_of_action-1
            q_next_all = np.dot(x_s_next, self.w)  # shape (num_of_action,)
            best_next_q = np.max(q_next_all)

        # TD target & error
        td_target = reward + self.discount_factor * best_next_q
        td_error = td_target - current_q

        # Gradient step to update w for that action dimension
        self.w[:, discrete_action_idx] += self.lr * td_error * x_s

    def learn(self, env, max_steps: int):
        """
        Run one training episode in the given environment (assuming it needs continuous actions).
        """
        obs, _info = env.reset()
        episode_return = 0.0
        self.count = 0
         
        for step in range(max_steps):
            # 1) Select an action (continuous) but store discrete idx internally
            continuous_action = self.select_action(obs)

            # 2) Step environment with the continuous action
            next_obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated
            # 3) Q-update using the discrete index we stored
            self.update(obs, self.last_discrete_action_idx, reward, next_obs, terminated)

            # 4) Move to next state
            obs = next_obs
            episode_return += reward
            self.count += 1

            if done:
                break

        # Decay epsilon after each episode
        self.decay_epsilon()

        self.reward_sum += reward

        return episode_return
