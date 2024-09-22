import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from reskill.utils.general_utils import AttrDict

class SkillModel(object):
    def __init__(self, vae_path, prior_path=None, adv_prior_path=None, device=torch.device('cpu')):
        self.vae_path = vae_path
        self.prior_path = prior_path
        self.adv_prior_path = adv_prior_path
        self.adv_prior = adv_prior_path is not None
        self.device = device

        skill_vae = torch.load(vae_path, map_location=device)
        skill_vae.device = device

        # Load skill prior module
        self.use_skill_prior = prior_path is not None or adv_prior_path is not None

        if self.use_skill_prior:
            selected_prior = prior_path if adv_prior_path is None else adv_prior_path
            skill_prior = torch.load(selected_prior, map_location=device)
            for i in skill_prior.bijectors:
                i.device = device
            self.skill_prior = skill_prior

        self.skill_vae = skill_vae
        self.n_features = self.skill_vae.n_z
        self.seq_len = self.skill_vae.seq_len

        base_mu, base_cov = torch.zeros(self.n_features), torch.eye(self.n_features)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

        self.reset_current_skill()
    
    def select_action(self, state):
        obs = torch.from_numpy(state).unsqueeze(dim=0)
        if self.current_skill is None:
            if self.use_skill_prior:
                sample = AttrDict(noise=self.base_dist.rsample(sample_shape=(1,)), state=obs)
                z = self.skill_prior.inverse(sample).noise.detach()
            else:
                z = torch.normal(0, 1, size=(1, self.n_features)).to(self.device)
            self.current_skill = z

        obs_z = torch.cat((obs, self.current_skill), 1)
        a = self.skill_vae.decoder(obs_z).cpu().detach().numpy()[0]
        self.current_skill_index += 1

        if self.current_skill_index == self.seq_len:
            self.reset_current_skill()
        return a

    
    def reset_current_skill(self):
        self.current_skill = None
        self.current_skill_index = 0
