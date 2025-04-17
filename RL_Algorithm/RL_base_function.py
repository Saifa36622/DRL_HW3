import numpy as np
from collections import defaultdict, namedtuple, deque
import random
from enum import Enum
import os
import json
import torch
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        self.memory.append(Transition(state, action, next_state, reward,done))

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """

        transitions = random.sample(self.memory, k=self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors (assuming all inputs are already torch.Tensor)
        state_batch      = torch.cat(batch.state)
        action_batch     = torch.cat(batch.action)
        reward_batch     = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        # done_batch       = torch.tensor([0.0 if d else 1.0 for d in batch.next_state is not None], dtype=torch.float32)
        done_batch       = torch.cat(batch.done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 1,
        count : int = 0,
        sum_count : int = 0,
        reward_sum : float = 0,
        ):

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.count = 0
        self.sum_count = 0 
        self.reward_sum = 0.0
        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        self.w = np.zeros((4, num_of_action))
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def q(self, obs, a=None):
        """Returns the linearly-estimated Q-value for a given state and action."""
        # a as in action
        # ========= put your code here ========= #

        policy_tensor = obs["policy"]  # e.g., a torch CUDA tensor
        x = policy_tensor.cpu().numpy().flatten()

        
        if a is None:
            # Return Q-values for all actions
            return np.dot(x, self.w)
        else:
            # Return Q-value for a specific action
            return np.dot(x, self.w[:, a])
        # ====================================== #
        
    
    def scale_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n].
            n (int): Number of discrete actions (inclusive range from 0 to n).
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #

        # action_min, action_max = self.action_range
        # num_bins = self.num_of_action - 1
        # scaled = action_min + (action / num_bins) * (action_max - action_min)

        # return torch.tensor([[scaled]], dtype=torch.float32, device=device)

        min_action, max_action = self.action_range 
        num_bins = self.num_of_action - 1 

        continuous_action = min_action + (action / num_bins) * (max_action - min_action)

        if isinstance(action, torch.Tensor):
            device = action.device  
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"  # Default to CUDA if available

        return torch.tensor([[continuous_action]], dtype=torch.float32, device=device)
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        
        # self.w = self.policy_net.fc_out.weight.detach().cpu().numpy()
        w_list = self.w.tolist()
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(w_list, f)

        # np.save(f"{path}/w.npy", self.w)

        # ====================================== #

            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        # self.w = np.load("saved_models/w.npy")
        
        with open(os.path.join(path, filename), 'r') as f:
            w_list = json.load(f)
            self.w = np.array(w_list)
        # ====================================== #


Transition2 = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done', 'log_prob'))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer2:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done,log_prob):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        self.memory.append(Transition2(state, action, next_state, reward, done, log_prob))

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """

        transitions = random.sample(self.memory, k=self.batch_size)
        batch = Transition2(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)
        log_prob_batch = torch.cat(batch.log_prob)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_prob_batch


    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)
    

class BaseAlgorithm2():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 1,
        count : int = 0,
        sum_count : int = 0,
        reward_sum : float = 0,
        ):

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.count = 0
        self.sum_count = 0 
        self.reward_sum = 0.0
        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        self.w = np.zeros((4, num_of_action))
        self.memory = ReplayBuffer2(buffer_size, batch_size)

    def q(self, obs, a=None):
        """Returns the linearly-estimated Q-value for a given state and action."""
        # a as in action
        # ========= put your code here ========= #

        policy_tensor = obs["policy"]  # e.g., a torch CUDA tensor
        x = policy_tensor.cpu().numpy().flatten()

        
        if a is None:
            # Return Q-values for all actions
            return np.dot(x, self.w)
        else:
            # Return Q-value for a specific action
            return np.dot(x, self.w[:, a])
        # ====================================== #
        
    
    def scale_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n].
            n (int): Number of discrete actions (inclusive range from 0 to n).
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #

        # action_min, action_max = self.action_range
        # num_bins = self.num_of_action - 1
        # scaled = action_min + (action / num_bins) * (action_max - action_min)

        # return torch.tensor([[scaled]], dtype=torch.float32, device=device)

        min_action, max_action = self.action_range 
        num_bins = self.num_of_action - 1 

        continuous_action = min_action + (action / num_bins) * (max_action - min_action)

        if isinstance(action, torch.Tensor):
            device = action.device  
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"  # Default to CUDA if available

        return torch.tensor([[continuous_action]], dtype=torch.float32, device=device)
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        
        # self.w = self.policy_net.fc_out.weight.detach().cpu().numpy()
        w_list = self.w.tolist()
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(w_list, f)

        # np.save(f"{path}/w.npy", self.w)

        # ====================================== #

            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        # self.w = np.load("saved_models/w.npy")
        
        with open(os.path.join(path, filename), 'r') as f:
            w_list = json.load(f)
            self.w = np.array(w_list)
        # ====================================== #