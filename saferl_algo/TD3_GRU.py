import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

HIDDEN_STATE = 256

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, HIDDEN_STATE)
		self.l2 = nn.Linear(HIDDEN_STATE, HIDDEN_STATE)
		self.l3 = nn.Linear(HIDDEN_STATE, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, HIDDEN_STATE)
		self.l2 = nn.Linear(HIDDEN_STATE, HIDDEN_STATE)
		self.l3 = nn.Linear(HIDDEN_STATE, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, HIDDEN_STATE)
		self.l5 = nn.Linear(HIDDEN_STATE, HIDDEN_STATE)
		self.l6 = nn.Linear(HIDDEN_STATE, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3GRU(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		goal_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		# TODO: want a shared feature extractor + GRU to prepend to all networks.
		# Also want forward functions to take such a hidden state? idk yet.

		# 1. Simple MLP extract
		# 2. GRU encode over time
		# 3. ???
		self.shared_gru = nn.GRU(state_dim - (action_dim) - goal_dim * 2, HIDDEN_STATE // 2).to(device)

		self.actor = Actor(HIDDEN_STATE // 2 + (action_dim) + goal_dim * 2, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)

		self.critic = Critic(HIDDEN_STATE // 2 + (action_dim) + goal_dim * 2, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)

		# Combine parameters of the Actor and shared GRU for the optimizers
		self.actor_optimizer = torch.optim.Adam(
			list(self.actor.parameters()) + list(self.shared_gru.parameters()), lr=3e-4
		)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
	
	def preprocess_feature_sequence(self, feature_sequence):
		feature_sequence = torch.FloatTensor(feature_sequence).unsqueeze(1).to(device)
		_, processed = self.shared_gru(feature_sequence)
		processed = processed.squeeze()
		return processed


	def select_action(self, state):
		# state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state_, action, next_state_, reward, not_done, curve_state, achieved_goal, desired_goal = \
			replay_buffer.sample(batch_size)
		# Preprocess both state and next_state with GRU now...
		action = torch.stack(action)
		reward = torch.stack(reward)
		not_done = torch.stack(not_done)
		curve_state = torch.stack(curve_state)
		achieved_goal = torch.stack(achieved_goal)
		desired_goal = torch.stack(desired_goal)

		process = lambda x: self.shared_gru(x.unsqueeze(1))[-1].squeeze() 
		
		state = torch.stack([process(state_i_) for state_i_ in state_])
		next_state = torch.stack([process(next_state_i_) for next_state_i_ in next_state_])
		state = torch.cat([state, curve_state, achieved_goal, desired_goal], dim=-1)
		next_state = torch.cat([next_state, curve_state, achieved_goal, desired_goal], dim=-1)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state.detach()) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state.detach(), next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state.detach(), action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it % self.policy_freq == 0:
			# Delayed policy updates

			# Compute actor losse
			actor_loss = -self.critic.Q1(state.detach(), self.actor(state)).mean()
			
			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		