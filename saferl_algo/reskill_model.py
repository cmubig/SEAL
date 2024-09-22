import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from reskill.utils.general_utils import AttrDict

from saferl_algo import TD3

class ReSkillModel(object):
    def __init__(self, vae_path, prior_path=None, no_residual_agent = False, no_skill_agent = False, 
                 device=torch.device('cpu'), **kwargs):
        self.vae_path = vae_path
        self.prior_path = prior_path
        self.device = device
        self.no_residual_agent = no_residual_agent
        self.no_skill_agent = no_skill_agent

        skill_vae = torch.load(vae_path, map_location=device)
        skill_vae.device = device

        # Load skill prior module
        self.use_skill_prior = prior_path is not None

        if self.use_skill_prior:
            skill_prior = torch.load(prior_path, map_location=device)
            for i in skill_prior.bijectors:
                i.device = device
            self.skill_prior = skill_prior

        self.skill_vae = skill_vae
        self.n_features = self.skill_vae.n_z
        self.seq_len = self.skill_vae.seq_len

        base_mu, base_cov = torch.zeros(self.n_features), torch.eye(self.n_features)
        base_mu = base_mu.to(device)
        base_cov = base_cov.to(device)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

        self.reset_current_skill()

        skill_agent_config = {k: v for k, v in kwargs.items()}
        residual_agent_config = {k: v for k, v in kwargs.items()}

        # TODO: potentially remove policy_noise/noise_clip stuff? Learning rates?
        skill_agent_config['action_dim'] = self.n_features
        residual_agent_config['state_dim'] += self.n_features + residual_agent_config['action_dim']

        self.skill_agent = TD3.TD3(**skill_agent_config)
        self.residual_agent = TD3.TD3(**residual_agent_config)

        # For easy access in ReplayBuffers, etc.
        self.last_obs_z = None
        self.last_a_dec = None
        self.last_obs_res = None
        self.last_a_res = None
        self.last_a = None
        self.last_skill = None
        self.last_skill_index = 0
        self.last_state = None
        self.last_noise_vec = None

    
    def select_action(self, state):
        self.last_state = state[np.newaxis, :]
        obs = torch.from_numpy(state).unsqueeze(dim=0).to(self.device)
        if self.current_skill is None:
            if self.use_skill_prior:
                if self.no_skill_agent:
                    noise_vec = self.base_dist.rsample(sample_shape=(1,)).detach().cpu().numpy()[0]
                else:
                    noise_vec = self.skill_agent.select_action(state)
                self.last_noise_vec = noise_vec[np.newaxis, :]
                noise_vec = torch.from_numpy(noise_vec).to(self.device).unsqueeze(0)
                sample = AttrDict(noise=noise_vec, state=obs)
                z = self.skill_prior.inverse(sample).noise.detach()
            else:
                self.last_noise_vec = np.zeros((1, self.n_features))
                z = torch.normal(0, 1, size=(1, self.n_features)).to(self.device)
            self.current_skill = z

        obs_z = torch.cat((obs, self.current_skill), 1)
        a_dec = self.skill_vae.decoder(obs_z).detach()
        self.last_obs_z = obs_z.detach().cpu().numpy()
        self.last_a_dec = a_dec.cpu().numpy()

        obs_res = torch.cat((obs, self.current_skill, a_dec), 1).cpu().detach().numpy()
        if self.no_residual_agent:
            a_res = np.zeros((self.residual_agent.actor.l3.out_features, )).astype(np.float32)
        else:
            a_res = self.residual_agent.select_action(obs_res)
        self.last_obs_res = obs_res
        self.last_a_res = a_res
        
        a = a_dec.cpu().numpy() + a_res
        self.last_a = a
        a = a[0]

        self.last_skill = self.current_skill.detach().cpu().numpy()
        self.last_skill_index = self.current_skill_index

        self.current_skill_index += 1

        if self.current_skill_index == self.seq_len:
            self.reset_current_skill()
        return a

    
    def reset_current_skill(self):
        self.current_skill = None
        self.current_skill_index = 0

    def train(self, skill_buffer, residual_buffer, batch_size):
        if not self.no_skill_agent:
            self.skill_agent.train(skill_buffer, batch_size)
        if not self.no_residual_agent:
            self.residual_agent.train(residual_buffer, batch_size)
    
    def load(self, filename):
        self.skill_agent.load(filename + '_skill')
        self.residual_agent.load(filename + '_residual')
    
    def save(self, filename):
        self.skill_agent.save(filename + '_skill')
        self.residual_agent.save(filename + '_residual')
        meta_info = {
            'vae_path': self.vae_path,
            'prior_path': self.prior_path,
            'no_residual_agent': self.no_residual_agent,
            'no_skill_agent': self.no_skill_agent
        }
        np.save(filename + '_meta.npy', meta_info)
