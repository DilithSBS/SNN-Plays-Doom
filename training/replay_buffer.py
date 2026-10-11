import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores transitions (state, action, reward, next_state, done) and samples them in random batches.
    """
    def __init__(self, capacity=10000):
        # We use a deque which automatically removes oldest items when full
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """
        Save a transition to the buffer.
        """
        # We store them as a tuple
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays for efficient PyTorch conversion later
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool)
        )
        
    def __len__(self):
        return len(self.buffer)
