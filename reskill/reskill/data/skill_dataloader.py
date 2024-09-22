from torch.utils.data import Dataset
from reskill.utils.general_utils import AttrDict
import numpy as np
import os

from metadrive.utils import safe_clip_for_small_array

class SkillsDataset(Dataset):

    SPLIT = AttrDict(train=0.90, val=0.10, test=0.0)

    def __init__(self, dataset_name, phase, subseq_len, transform, adv_prior):        
        self.phase = phase
        self.subseq_len = subseq_len
        self.adv_prior = adv_prior
        curr_dir = os.path.dirname(__file__)
        fname = os.path.join(curr_dir, "../../dataset/" + dataset_name + "/demos.npy")
        self.seqs = np.load(fname, allow_pickle=True)
        self.transform = transform

        self.cat = '40000' not in dataset_name
        if not self.cat:
            raise NotImplementedError('This dataset has been modified for cat; use original dataset for other tasks')
        

        # Things to process
        # 1. Safe clip actions to (-1, 1) range
        # 2. Process sequences to be every possible subsequence, reflected in dataset length
        new_seqs = []
        is_adv = []
        seq_unique = 0
        adv_unique = 0
        for seq in self.seqs:
            actions = seq['actions']
            obs = seq['obs']
            cur_done = None
            if 'done' in seq:
                done = seq.done
                # Failure cases, remove the lead up to the sequences
                if (done == 'crash' and not adv_prior) or done == 'out_of_road':
                    actions = actions[:-self.subseq_len]
                    obs = obs[:-self.subseq_len]
                cur_done = done
            if len(actions) < self.subseq_len:
                continue
            added_set = False
            for seq_i in range(len(actions) - self.subseq_len):
                actions_i = np.clip(actions[seq_i:seq_i + self.subseq_len], -1, 1).astype(np.float32)
                obs_i = np.array(obs[seq_i:seq_i + self.subseq_len])
                new_seqs.append(AttrDict(actions=actions_i, obs=obs_i, is_adv=False))
                if not adv_prior:
                    adv_unique += 1
                    seq_unique += 1
                    is_adv.append(False)
                    continue
                # Keep last (subseq_len * 2), heuristically, as the portion corresponding to adv
                if cur_done == 'crash' and seq_i >= len(actions) - self.subseq_len * 2:
                    is_adv.append(True)
                    if not added_set:
                        added_set = True
                        adv_unique += 1
                else:
                    is_adv.append(False)
                    if not added_set:
                        added_set = True
                        seq_unique += 1
                new_seqs[-1].is_adv = is_adv[-1]
        self.seqs = new_seqs
        self.is_adv = np.array(is_adv)

        self.n_seqs = len(self.seqs)
        print("Dataset size: ", self.n_seqs)
        if adv_prior:
            print(f"Benign subset: {len(self.seqs) - self.is_adv.sum()}, Adv subset: {self.is_adv.sum()}")
        self.action_dim = len(self.seqs[0]['actions'][0])

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        elif self.phase == "test":
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs  



    def __getitem__(self, index):
        return self.seqs[self.start + index]

    def __len__(self):
        return int(self.end-self.start)

    
        
