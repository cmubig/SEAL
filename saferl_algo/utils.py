import numpy as np
import torch

class ReplayBufferPreserveGRU(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e5)):
		self.max_size = max_size
		self.ptr = 0

		self.size = 0

		# self.state = np.zeros((max_size, state_dim))
		# self.action = np.zeros((max_size, action_dim))
		# self.next_state = np.zeros((max_size, state_dim))
		# self.reward = np.zeros((max_size, 1))
		# self.not_done = np.zeros((max_size, 1))
		self.state = [None] * max_size
		self.action = [None] * max_size
		self.next_state = [None] * max_size
		self.reward = [None] * max_size
		self.not_done = [None] * max_size
		self.curve_state = [None] * max_size
		self.achieved_goal = [None] * max_size
		self.desired_goal = [None] * max_size

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done, curve_state, achieved_goal, desired_goal):
		self.state[self.ptr] = state.cpu().numpy().astype(np.float32)
		self.action[self.ptr] = action.cpu().numpy().astype(np.float32)
		self.next_state[self.ptr] = next_state.cpu().numpy().astype(np.float32)
		self.reward[self.ptr] = reward.cpu().numpy().astype(np.float32)
		self.not_done[self.ptr] = 1. - done.cpu().numpy().astype(np.float32)
		self.curve_state[self.ptr] = curve_state.cpu().numpy().astype(np.float32)
		self.achieved_goal[self.ptr] = achieved_goal.cpu().numpy().astype(np.float32)
		self.desired_goal[self.ptr] = desired_goal.cpu().numpy().astype(np.float32)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
	
	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			[torch.FloatTensor(self.state[i]).to(self.device) for i in ind],
			[torch.FloatTensor(self.action[i]).to(self.device) for i in ind],
			[torch.FloatTensor(self.next_state[i]).to(self.device) for i in ind],
			[torch.FloatTensor(self.reward[i]).to(self.device) for i in ind],
			[torch.FloatTensor(self.not_done[i]).to(self.device) for i in ind],
			[torch.FloatTensor(self.curve_state[i]).to(self.device) for i in ind],
			[torch.FloatTensor(self.achieved_goal[i]).to(self.device) for i in ind],
			[torch.FloatTensor(self.desired_goal[i]).to(self.device) for i in ind],
		)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)