from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """        

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        # next_action: int,
        terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #
        x_s = obs["policy"].cpu().numpy().flatten()

        # Compute the current Q(s,a)
        current_q = np.dot(x_s, self.w[:, action])

        # For Q-learning, we use max over actions in the next state
        # If the episode is terminated, there's no bootstrap from next state
        if terminated:
            # No future Q if episode ended
            best_next_q = 0.0
        else:
            # Flatten next state
            x_s_next = next_obs["policy"].cpu().numpy().flatten()
            # Q(s',a') for all possible a'
            q_next_all = np.dot(x_s_next, self.w)  # shape: (num_of_actions,)
            best_next_q = np.max(q_next_all)

        td_target = reward + self.discount_factor * best_next_q

        # TD error
        td_error = td_target - current_q

        # Gradient step to update w for that action dimension
        # w[:, a] ← w[:, a] + alpha * TD_error * x_s
        self.w[:, action] += self.lr * td_error * x_s
        # ====================================== #

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if np.random.rand() < self.epsilon:
            # Explore: pick random action
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            # Exploit: pick action with max Q(s,a)
            x_s = state["policy"].cpu().numpy().flatten()  # shape (4,) for CartPole
            # Q-values for each action
            q_vals = np.dot(x_s, self.w)  # shape (num_of_action,)
            action_idx = int(np.argmax(q_vals))
        
        return action_idx
        # ====================================== #

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs, _info = env.reset()

        # Track total reward for logging (optional)
        episode_return = 0.0

        for step in range(max_steps):
            # 2) Choose action via epsilon-greedy
            action = self.select_action(obs)

            # 3) Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 4) Update the Q-values (linear approximation)
            # next_action is not used for Q-learning, but it's in the method signature.
            self.update(obs, action, reward, next_obs, terminated=done)

            # 5) Move to next state
            obs = next_obs
            episode_return += reward

            # 6) Break if done
            if done:
                break

        # Decay epsilon after each episode
        self.decay_epsilon()

        # Optionally, return the episode_return if you want to track total reward
        return episode_return
        # ====================================== #
    




    