import torch
from torch.utils.data import DataLoader
from reskill.data.skill_dataloader import SkillsDataset
from torchvision import transforms
from reskill.utils.general_utils import AttrDict

device = torch.device('cpu')

if __name__ == '__main__':
    # TODO: remove hard coded paths
    dataset_name_train = 'ego_idm_normal_n1_train-seed0-0'
    dataset_name_eval = dataset_name_train.replace('train', 'eval')

    skill_vae_path = "./results/saved_skill_models/" + dataset_name_train + "/skill_vae_best.pth"
    skill_vae = torch.load(skill_vae_path, map_location=device)
    skill_vae.device = device
    # Load skill prior module
    skill_prior_path = "./results/saved_skill_models/" + dataset_name_train + "/skill_prior_best.pth"
    skill_prior = torch.load(skill_prior_path, map_location=device)
    for i in skill_prior.bijectors:
        i.device = device

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])          
    test_data = SkillsDataset(dataset_name_eval, phase="train", subseq_len=10, transform=transform)
    test_data.start = 0
    test_data.end = test_data.n_seqs


    test_loader = DataLoader(
        test_data,
        batch_size = 128,
        shuffle = False,
        drop_last=False,
        prefetch_factor=30,
        num_workers=8,
        pin_memory=True)

    skill_vae.eval()
    running_total_loss = 0.0
    running_bc_loss = 0.0
    running_kl_loss = 0.0
    running_nvp_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data["actions"] = data["actions"].to(device)
            data["obs"] = data["obs"].to(device)
            skill_vae.init_hidden(data["actions"].size(0))
            output = skill_vae(data)
            losses = skill_vae.loss(data, output)

            loss = losses.total_loss
            running_total_loss += loss.item()
            running_bc_loss += losses.bc_loss.item()
            running_kl_loss += losses.kld_loss.item()

            sp_input = AttrDict(skill=output.z.detach().to('cpu'),
                                state=data["obs"][:,0,:].to('cpu'))
            z, log_pz, log_jacob = skill_prior(sp_input)
            sp_loss = (-log_pz - log_jacob).sum()
            running_nvp_loss += sp_loss.item()

