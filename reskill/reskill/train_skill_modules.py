
import torch
import torch.optim as optim
import numpy as np
import argparse
from typing import List
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import pdb
import wandb
from tqdm import tqdm
import os
import time
import yaml

from reskill.models.skill_vae import SkillVAE
from reskill.data.skill_dataloader import SkillsDataset
from reskill.models.rnvp import stacked_NVP
from reskill.utils.general_utils import AttrDict



class ModelTrainer():
    def __init__(self, dataset_name, config_file, extra_tag_name, adv_prior_name):
        self.dataset_name = dataset_name
        self.save_dir = "./results/saved_skill_models/" + dataset_name + adv_prior_name + extra_tag_name + "/"
        self.adv_prior = bool(adv_prior_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.vae_save_path = self.save_dir + "skill_vae.pth"
        self.sp_save_path = self.save_dir + "skill_prior.pth"
        if self.adv_prior:
            self.adv_sp_save_path = self.save_dir + "adv_skill_prior.pth"
        self.vae_best_save_path = self.save_dir + "skill_vae_best.pth"
        self.sp_best_save_path = self.save_dir + "skill_prior_best.pth"
        if self.adv_prior:
            self.adv_sp_best_save_path = self.save_dir + "adv_skill_prior_best.pth"
        config_path = "configs/skill_mdl/" + config_file

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)


        with open(config_path, 'r') as file:
            conf = yaml.safe_load(file)
            conf = AttrDict(conf)
        for key in conf:
            conf[key] = AttrDict(conf[key])        

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])          
        train_data = SkillsDataset(dataset_name, phase="train", subseq_len=conf.skill_vae.subseq_len, transform=transform, adv_prior=self.adv_prior)
        val_data   = SkillsDataset(dataset_name, phase="val", subseq_len=conf.skill_vae.subseq_len, transform=transform, adv_prior=self.adv_prior)

        self.train_loader = DataLoader(
            train_data,
            batch_size = conf.skill_vae.batch_size,
            shuffle = True,
            drop_last=False,
            prefetch_factor=30,
            num_workers=8,
            pin_memory=True)

        self.val_loader = DataLoader(
            val_data,
            batch_size = conf.skill_vae.batch_size,
            shuffle = False,
            drop_last=False,
            prefetch_factor=30,
            num_workers=8,
            pin_memory=True)

        self.skill_vae = SkillVAE(n_actions=conf.skill_vae.n_actions, n_obs=conf.skill_vae.n_obs, n_hidden=conf.skill_vae.n_hidden,
                                  seq_length=conf.skill_vae.subseq_len, n_z=conf.skill_vae.n_z, device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.skill_vae.parameters(), lr=conf.skill_vae.lr)

        self.vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10, cooldown=10, 
                                                                        factor=0.5, verbose=True)

        self.sp_nvp = stacked_NVP(d=conf.skill_prior_nvp.d, k=conf.skill_prior_nvp.k, n_hidden=conf.skill_prior_nvp.n_hidden,
                                  state_size=conf.skill_vae.n_obs, n=conf.skill_prior_nvp.n_coupling_layers, device=self.device).to(self.device)
        
        self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)

        self.sp_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.sp_optimizer, 0.999)

        if self.adv_prior:
            self.adv_sp_nvp = stacked_NVP(d=conf.skill_prior_nvp.d, k=conf.skill_prior_nvp.k, n_hidden=conf.skill_prior_nvp.n_hidden,
                                    state_size=conf.skill_vae.n_obs, n=conf.skill_prior_nvp.n_coupling_layers, device=self.device).to(self.device)
            
            self.adv_sp_optimizer = torch.optim.Adam(self.adv_sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
            self.adv_sp_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.adv_sp_optimizer, 0.999)

        self.n_epochs = conf.skill_vae.epochs


    def fit(self, epoch):
        self.skill_vae.train()
        running_total_loss = 0.0
        running_bc_loss = 0.0
        running_kl_loss = 0.0
        if self.adv_prior:
            running_nvp_loss_normal = 0.0
            running_nvp_loss_adv = 0.0
        else:
            running_nvp_loss = 0.0
        for i, data in enumerate(self.train_loader):

            data["actions"] = data["actions"].to(self.device)
            data["obs"] = data["obs"].to(self.device)

            # Train skills model
            self.skill_vae.init_hidden(data["actions"].size(0))
            self.optimizer.zero_grad()
            output = self.skill_vae(data)
            losses = self.skill_vae.loss(data, output)
            loss = losses.total_loss
            running_total_loss += loss.item()
            running_bc_loss += losses.bc_loss.item()
            running_kl_loss += losses.kld_loss.item()
            loss.backward()
            self.optimizer.step()

            # Train skills prior model
            if self.adv_prior:
                is_adv = data["is_adv"].to(self.device)
                skill_normal = output.z.detach()[~is_adv]
                state_normal = data["obs"][:,0,:][~is_adv]
                skill_adv = output.z.detach()[is_adv]
                state_adv = data["obs"][:,0,:][is_adv]

                if len(skill_normal):
                    self.sp_optimizer.zero_grad()
                    sp_input_normal = AttrDict(skill=skill_normal, state=state_normal)
                    z_normal, log_pz_normal, log_jacob_normal = self.sp_nvp(sp_input_normal)
                    sp_loss_normal = (-log_pz_normal - log_jacob_normal).sum()
                    running_nvp_loss_normal += sp_loss_normal.item()
                    sp_loss_normal.backward()
                    self.sp_optimizer.step()

                if len(skill_adv):
                    self.adv_sp_optimizer.zero_grad()
                    sp_input_adv = AttrDict(skill=skill_adv, state=state_adv)
                    z_adv, log_pz_adv, log_jacob_adv = self.adv_sp_nvp(sp_input_adv)
                    sp_loss_adv = (-log_pz_adv - log_jacob_adv).sum()
                    running_nvp_loss_adv += sp_loss_adv.item()
                    sp_loss_adv.backward()
                    self.adv_sp_optimizer.step()
            else:
                self.sp_optimizer.zero_grad()
                sp_input = AttrDict(skill=output.z.detach(),
                                    state=data["obs"][:,0,:])
                z, log_pz, log_jacob = self.sp_nvp(sp_input)
                sp_loss = (-log_pz - log_jacob).sum()
                running_nvp_loss += sp_loss.item()
                sp_loss.backward()
                self.sp_optimizer.step()

            if i % 250 == 0:
                self.sp_scheduler.step()
                wandb.log({'sp_lr':self.sp_scheduler.get_lr()[0]}, epoch)
                if self.adv_prior:
                    self.adv_sp_scheduler.step()

            
        if self.adv_prior:
            return running_total_loss, running_bc_loss, running_kl_loss, running_nvp_loss_normal, running_nvp_loss_adv
        else:
            return running_total_loss, running_bc_loss, running_kl_loss, running_nvp_loss


    def validate(self):
        self.skill_vae.eval()
        running_total_loss = 0.0
        running_bc_loss = 0.0
        running_kl_loss = 0.0
        if self.adv_prior:
            running_nvp_loss_normal = 0.0
            running_nvp_loss_adv = 0.0
        else:
            running_nvp_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                data["actions"] = data["actions"].to(self.device)
                data["obs"] = data["obs"].to(self.device)
                self.skill_vae.init_hidden(data["actions"].size(0))
                self.optimizer.zero_grad()
                output = self.skill_vae(data)
                losses = self.skill_vae.loss(data, output)

                loss = losses.total_loss
                running_total_loss += loss.item()
                running_bc_loss += losses.bc_loss.item()
                running_kl_loss += losses.kld_loss.item()

                if self.adv_prior:
                    is_adv = data["is_adv"].to(self.device)
                    skill_normal = output.z.detach()[~is_adv]
                    state_normal = data["obs"][:,0,:][~is_adv]
                    skill_adv = output.z.detach()[is_adv]
                    state_adv = data["obs"][:,0,:][is_adv]

                    if len(skill_normal):
                        sp_input_normal = AttrDict(skill=skill_normal, state=state_normal)
                        z_normal, log_pz_normal, log_jacob_normal = self.sp_nvp(sp_input_normal)
                        sp_loss_normal = (-log_pz_normal - log_jacob_normal).sum()
                        running_nvp_loss_normal += sp_loss_normal.item()

                    if len(skill_adv):
                        sp_input_adv = AttrDict(skill=skill_adv, state=state_adv)
                        z_adv, log_pz_adv, log_jacob_adv = self.adv_sp_nvp(sp_input_adv)
                        sp_loss_adv = (-log_pz_adv - log_jacob_adv).sum()
                        running_nvp_loss_adv += sp_loss_adv.item()
                else:
                    sp_input = AttrDict(skill=output.z.detach(),
                                        state=data["obs"][:,0,:])
                    z, log_pz, log_jacob = self.sp_nvp(sp_input)
                    sp_loss = (-log_pz - log_jacob).sum()
                    running_nvp_loss += sp_loss.item()

        if self.adv_prior:
            return running_total_loss, running_bc_loss, running_kl_loss, running_nvp_loss_normal, running_nvp_loss_adv
        else:
            return running_total_loss, running_bc_loss, running_kl_loss, running_nvp_loss


    def train(self):
        print("Training...") 
        best_val_perf = torch.inf
        for epoch in tqdm(range(self.n_epochs)):
            train_losses = self.fit(epoch)
            train_denom = len(self.train_loader.dataset) * self.train_loader.dataset.subseq_len * self.train_loader.dataset.action_dim
            if self.adv_prior:
                is_adv = np.array([self.train_loader.dataset[i].is_adv for i in range(len(self.train_loader.dataset))])
                normal_ratio = len(self.train_loader.dataset) / (~is_adv).sum()
                adv_ratio = len(self.train_loader.dataset) / (is_adv).sum()
                train_total_loss, train_bc_loss, train_kl_loss, train_nvp_loss_normal, train_nvp_loss_adv = [x / train_denom for x in train_losses]
                train_nvp_loss_normal *= normal_ratio
                train_nvp_loss_adv *= adv_ratio
            else:
                train_total_loss, train_bc_loss, train_kl_loss, train_nvp_loss = [x / train_denom for x in train_losses]

            val_losses = self.validate()
            val_denom = len(self.val_loader.dataset) * self.val_loader.dataset.subseq_len * self.val_loader.dataset.action_dim
            if self.adv_prior:
                is_adv = np.array([self.val_loader.dataset[i].is_adv for i in range(len(self.val_loader.dataset))])
                normal_ratio = len(self.val_loader.dataset) / (~is_adv).sum()
                adv_ratio = len(self.val_loader.dataset) / (is_adv).sum()
                val_total_loss, val_bc_loss, val_kl_loss, val_nvp_loss_normal, val_nvp_loss_adv = [x / val_denom for x in val_losses]
                val_nvp_loss_normal *= normal_ratio
                val_nvp_loss_adv *= adv_ratio
            else:
                val_total_loss, val_bc_loss, val_kl_loss, val_nvp_loss  = [x / val_denom for x in val_losses]

            self.vae_scheduler.step(val_bc_loss)

            wandb.log({'vae_lr':self.optimizer.param_groups[0]['lr']}, epoch)
            wandb.log({'train_total_loss':train_total_loss}, epoch)
            # Reconstruction loss
            wandb.log({'train_bc_loss':train_bc_loss}, epoch)
            wandb.log({'train_kl_loss':train_kl_loss}, epoch)
            if self.adv_prior:
                wandb.log({'train_nvp_loss_normal':train_nvp_loss_normal}, epoch)
                wandb.log({'train_nvp_loss_adv':train_nvp_loss_adv}, epoch)
            else:
                wandb.log({'train_nvp_loss':train_nvp_loss}, epoch)

            wandb.log({'val_total_loss':val_total_loss}, epoch)
            # Reconstruction loss
            wandb.log({'val_bc_loss':val_bc_loss}, epoch)
            wandb.log({'val_kl_loss':val_kl_loss}, epoch)
            if self.adv_prior:
                wandb.log({'val_nvp_loss_normal':val_nvp_loss_normal}, epoch)
                wandb.log({'val_nvp_loss_adv':val_nvp_loss_adv}, epoch)
            else:
                wandb.log({'val_nvp_loss':val_nvp_loss}, epoch)



            perf_metric = val_nvp_loss_adv if self.adv_prior else val_bc_loss
            if perf_metric < best_val_perf:
                best_val_perf = perf_metric
                torch.save(self.skill_vae, self.vae_save_path)
                torch.save(self.sp_nvp, self.sp_save_path)
                torch.save(self.skill_vae, self.vae_best_save_path)
                torch.save(self.sp_nvp, self.sp_best_save_path)
                if self.adv_prior:
                    torch.save(self.adv_sp_nvp, self.adv_sp_save_path)
                    torch.save(self.adv_sp_nvp, self.adv_sp_best_save_path)

            if epoch % 50 == 0 or epoch == self.n_epochs - 1:
                torch.save(self.skill_vae, self.vae_save_path)
                torch.save(self.sp_nvp, self.sp_save_path)
                if self.adv_prior:
                    torch.save(self.adv_sp_nvp, self.adv_sp_save_path)
                
   
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="cat/config_all.yaml")
    parser.add_argument('--dataset_name', type=str, default="idm_all")
    parser.add_argument('--extra_tag', type=str, default='')
    parser.add_argument('--adv_prior', action='store_true', help='Whether or not to train a separate adv_prior on crashing sequences')
    args=parser.parse_args()
    
    wandb.init(project="skill_mdl")
    extra_tag_name = '' if not len(args.extra_tag) else f'_{args.extra_tag}'
    adv_prior_name = '' if not args.adv_prior else '_adv_prior'
    wandb.run.name = "skill_mdl_" + args.dataset_name + adv_prior_name + extra_tag_name + '_' + time.asctime()
    wandb.run.save()

    trainer = ModelTrainer(args.dataset_name, args.config_file, extra_tag_name, adv_prior_name)
    trainer.train()