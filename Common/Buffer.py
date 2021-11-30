import numpy as np
import torch


class Buffer():

    def __init__(self, state_dim, action_dim, buffer_size,device):
        self.device = device
        self.states = np.zeros([buffer_size, state_dim], dtype=np.float32)
        self.actions = np.zeros([buffer_size, action_dim], dtype=np.float32)
        self.rewards = np.zeros([buffer_size, 1], dtype=np.float32)
        self.next_states = np.zeros([buffer_size, state_dim], dtype=np.float32)
        self.dones = np.zeros([buffer_size, 1], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return torch.FloatTensor(self.states[idxs]).to(self.device), torch.FloatTensor(self.actions[idxs]).to(self.device), torch.FloatTensor(
            self.rewards[idxs]).to(self.device), torch.FloatTensor(self.next_states[idxs]).to(self.device), torch.FloatTensor(self.dones[idxs]).to(self.device)
