import numpy as np
import torch


# Replay buffer
class LAP(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		zsa_dim,
		device,
		args,
		max_size=1e6,
		batch_size=256,
		max_action=1,
		normalize_actions=True,
		prioritized=True
	):

		# Parameters.
		max_size = int(max_size)
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size

		self.action_dim = action_dim
		self.state_dim = state_dim
		self.zsa_dim = zsa_dim

		# Memory
		self.state = torch.zeros((max_size, state_dim)).to(args.device)
		self.action = torch.zeros((max_size, action_dim)).to(args.device)
		self.next_state = torch.zeros((max_size, state_dim)).to(args.device)
		self.reward = torch.zeros((max_size, 1)).to(args.device)
		self.correction = torch.zeros((max_size, 1)).to(args.device)
		self.zsa = torch.zeros((max_size, zsa_dim)).to(args.device)
		self.not_done = torch.zeros((max_size, 1)).to(args.device)

		self.prioritized = prioritized

		if prioritized:
			self.priority = torch.zeros(max_size, device=device)
			self.prioritized = True
			self.max_priority = 1

		self.normalize_actions = max_action if normalize_actions else 1

		self.args = args

	# Add tuple.
	def add(self, state, action, next_state, reward, correction, zsa, done):
		self.state[self.ptr:self.ptr + state.shape[0]] = state
		self.action[self.ptr:self.ptr + state.shape[0]] = action / self.normalize_actions
		self.next_state[self.ptr:self.ptr + state.shape[0]] = next_state
		self.reward[self.ptr:self.ptr + state.shape[0]] = reward
		self.correction[self.ptr:self.ptr + state.shape[0]] = correction
		self.zsa[self.ptr:self.ptr + state.shape[0]] = zsa
		self.not_done[self.ptr:self.ptr + state.shape[0]] = 1. - done
		
		if self.prioritized:
			self.priority[self.ptr:self.ptr + state.shape[0]] = self.max_priority

		self.ptr = (self.ptr + state.shape[0]) % self.max_size
		self.size = min(self.size + state.shape[0], self.max_size)

	# Sample tuple.
	def sample(self, prioritized=False):
		if prioritized:
			csum = torch.cumsum(self.priority[:self.size], 0)
			val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
			self.ind = torch.searchsorted(csum, val).cpu().data.numpy()

		else:
			self.ind = np.random.randint(0, self.size, size=min(self.batch_size, self.size))

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.zsa[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
		)

	def update_priority(self, priority):
		self.priority[self.ind] = priority.reshape(-1).detach()
		self.max_priority = max(float(priority.max()), self.max_priority)

	def reset_max_priority(self):
		self.max_priority = float(self.priority[:self.size].max())

	# Load offline dataset.
	def load_D4RL(self, dataset):
		self.state = torch.tensor(dataset['observations'], dtype=torch.float, device=self.device)
		self.action = torch.tensor(dataset['actions'], dtype=torch.float, device=self.device)
		self.next_state = torch.tensor(dataset['next_observations'], dtype=torch.float, device=self.device)
		self.reward = torch.tensor(dataset['rewards'].reshape(-1, 1), dtype=torch.float, device=self.device)
		self.not_done = torch.tensor(1. - dataset['terminals'].reshape(-1, 1), dtype=torch.float, device=self.device)
		self.size = self.state.shape[0]

		if self.prioritized:
			self.priority = torch.ones(self.size).to(self.device)