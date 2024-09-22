import argparse
import os
import numpy as np
import pandas as pd
from natsort import natsorted

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from saferl_plotter.logger import SafeLogger

from safeshift.measure_utils import shift_rotate

def create_model():
    return Model(input_size=3, hidden_size=32, out_size=2)

def clean_traj(x, adv_x):
    x[np.isinf(x)] = 0
    x[np.isnan(x)] = 0
    adv_x[np.isnan(adv_x)] = 0
    adv_x[np.isinf(adv_x)] = 0
    x = np.concatenate([x[:, :2], x[:, -1:]], axis=-1)
    x_rel = []
    x_valid = []
    for xi, adv_xi in zip(x, adv_x):
        adv_xyi = adv_xi[:2]
        adv_headingi = adv_xi[4]
        reli = shift_rotate(xi[:2], -adv_xyi, -adv_headingi)
        x_rel.append(reli)
        x_valid.append(int(adv_xi[-1]) & int(xi[-1]))
    x_rel = np.stack(x_rel, axis=0)

    x = x[:len(x_rel)]
    x[:, :2] = x_rel
    x[:, -1] = x_valid
    return x

class Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, out_size=1):
        super().__init__()

        # Should do layer norm
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.traj_mlp1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.traj_mlp2 = nn.Sequential(
            nn.Linear(int(hidden_size * 2), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.traj_mlp3 = nn.Sequential(
            nn.Linear(int(hidden_size * 2), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.traj_agg = lambda x: torch.max(x, dim=-2)[0]

        # main_mlp_in = int(512 + 2 * hidden_size)
        # self.main_mlp = nn.Sequential(
        #     nn.Linear(main_mlp_in, self.out_size),
        # )
        # main_mlp_in = int(2 * hidden_size)

        main_mlp_in = int(hidden_size)
        # main_mlp_in = int(512 + hidden_size)
        self.main_mlp = nn.Sequential(
            nn.Linear(main_mlp_in, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.out_size)
            # nn.Linear(hidden_size, self.out_size)
        )
    
    def forward(self, last_states, egos, advs):
        # Expected input size: ... x n x 10

        # ego_aggs, adv_aggs = [], []
        ego_aggs = []
        for ego, adv in zip(egos, advs):
            ego_feat1 = self.traj_mlp1(ego)
            ego_agg1 = self.traj_agg(ego_feat1)
            ego_feat1 = torch.cat([ego_feat1, ego_agg1.repeat(len(ego_feat1), 1)], dim=-1)

            ego_feat2 = self.traj_mlp2(ego_feat1)
            ego_agg2 = self.traj_agg(ego_feat2)
            ego_feat2 = torch.cat([ego_feat2, ego_agg2.repeat(len(ego_feat2), 1)], dim=-1)

            ego_feat3 = self.traj_mlp3(ego_feat2)
            ego_agg3 = self.traj_agg(ego_feat3)
            ego_aggs.append(ego_agg3)
        ego_aggs = torch.stack(ego_aggs)

        # main_in = torch.cat([last_states, ego_aggs, adv_aggs], dim=-1)
        # main_in = torch.cat([ego_aggs, adv_aggs], dim=-1)
        main_in = torch.cat([ego_aggs], dim=-1)
        # main_in = torch.cat([last_states, ego_aggs], dim=-1)
        output = torch.sigmoid(self.main_mlp(main_in))
        return output

class DecisionDataset(Dataset):
    def __init__(self, dataset_name=None, all_data =None, split='train', seed=1):
        super().__init__()
        if split not in {'train', 'val', 'test'}:
            raise ValueError()

        assert np.sum([dataset_name is None, all_data is None]) == 1, 'Exactly 1 of name or data must be given'
        all_data = all_data if all_data is not None else \
            np.load(f'dataset/{dataset_name}/data.npy', allow_pickle=True).item()
        all_X = np.array(all_data['last_states'])
        all_y_sc = np.array(all_data['sc_scores'])
        all_y_diff = np.array(all_data['diff_scores'])
        #all_y = np.array(y_sc)
        all_y = np.concatenate([all_y_sc[:, np.newaxis], all_y_diff[:, np.newaxis]], axis=-1)
        all_egos = all_data['original_ego_tracks']
        all_others = all_data['original_other_tracks']
        all_y_sc_others = all_data['other_sc_scores']
        all_y_diff_others = all_data['other_diff_scores']
        all_advs = all_data['adv_tracks']

        unique_scene_ids = natsorted(np.unique(all_data['scene_ids']))
        if split == 'train':
            unique_scene_ids = unique_scene_ids[:400]
            start_range = 0
            end_range = 0.85
        elif split == 'val':
            unique_scene_ids = unique_scene_ids[:400]
            start_range = 0.85
            end_range = 1
        else:
            unique_scene_ids = unique_scene_ids[400:]
            start_range = 0
            end_range = 1

        X = []
        egos = []
        advs = []
        others = []
        scene_ids = []

        y = []
        other_y = []
        random_state = np.random.RandomState(seed)
        # Split by scenes altogether
        # TODO: add in more than just ego-adv relation; also if diff_score is -1, ignore xddd
        for scene_id in unique_scene_ids:
            idxs = np.array(all_data['scene_ids']) == scene_id
            actual_idxs = np.arange(len(idxs))[idxs]
            rand_val = random_state.rand()
            if rand_val < start_range or rand_val >= end_range:
                continue
            scene_ids.extend([scene_id] * len(actual_idxs))
            X.extend(all_X[idxs])
            y.extend(all_y[idxs])
            # TODO: process others here
            d_data = all_egos[actual_idxs[0]].shape[-1]
            d_y = y[-1].shape[-1]
            base_shape_data = np.empty((0, 0, d_data))
            base_shape_y = np.empty((0, d_y))

            for i in actual_idxs:
                other_data = all_others[i]
                other_y_sc = all_y_sc_others[i]
                other_y_diff = all_y_diff_others[i]
                joint_keys = set.intersection(set(other_data.keys()), set(other_y_sc.keys()), set(other_y_diff.keys()))
                joint_keys = natsorted(list(joint_keys))
                joint_keys = [x for x in joint_keys if other_y_diff[x] != -1]
                if not len(joint_keys):
                    others.append(base_shape_data)
                    other_y.append(base_shape_y)
                else:
                    others.append(np.stack([np.array(other_data[k]) for k in joint_keys]))
                    other_y.append(np.stack([np.array([other_y_sc[k], other_y_diff[k]]) for k in joint_keys]))
            egos.extend([np.array(all_egos[i]) for i in actual_idxs])
            advs.extend([np.array(all_advs[i]) for i in actual_idxs])
        for i, (ego, adv, other) in tqdm(enumerate(zip(egos, advs, others)), 'Cleaning trajs', total=len(egos)):
            # Want these to be relative to adv!
            clean_ego = clean_traj(ego, adv)
            clean_adv = clean_traj(adv, adv)
            if len(other):
                clean_other = np.stack([clean_traj(x, adv) for x in other])
            else:
                clean_other = np.empty((0, 0, 3))
            egos[i] = clean_ego
            advs[i] = clean_adv
            others[i] = clean_other
        X = np.array(X)
        y = np.array(y)
        self.X = X
        self.egos = egos
        self.advs = advs
        self.y = y
        self.others = others
        self.other_y = other_y
        self.scene_ids = np.array(scene_ids)
    
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx],
            'ego': self.egos[idx],
            'adv': self.advs[idx],
            'other': self.others[idx],
            'other_y': self.other_y[idx],
            'scene_id': int(self.scene_ids[idx].split('_')[-1])
        }
    
def collate_batch(entries):
    # entries is list of dicts
    combined_X = torch.tensor(np.array([entry['X'] for entry in entries]))
    combined_y = torch.tensor(np.array([entry['y'] for entry in entries]))

    # Sometimes ragged
    ragged_ego = [torch.tensor(entry['ego']) for entry in entries]
    ragged_adv = [torch.tensor(entry['adv']) for entry in entries]
    ragged_other = [torch.tensor(entry['other']) for entry in entries]
    ragged_other_y = [torch.tensor(entry['other_y']) for entry in entries]

    combined_scene_ids = torch.tensor(np.array([entry['scene_id'] for entry in entries]))

    return {
        'X': combined_X,
        'y': combined_y,
        'ragged_egos': ragged_ego,
        'ragged_advs': ragged_adv,
        'ragged_others': ragged_other,
        'ragged_other_y': ragged_other_y,
        'scene_ids': combined_scene_ids
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='idm32_all')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--extra_tag', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    print('Loading data')
    all_data = np.load(f'dataset/{args.dataset_name}/data.npy', allow_pickle=True).item()
    print('Loading train')
    train_data = DecisionDataset(all_data=all_data, split='train', seed=args.seed)
    print('Loading val')
    val_data = DecisionDataset(all_data=all_data, split='val', seed=args.seed)
    print('Loading test')
    test_data = DecisionDataset(all_data=all_data, split='test', seed=args.seed)

    extra_name = f'_{args.extra_tag}' if args.extra_tag else ''
    exp_name = f'{args.dataset_name}{extra_name}'

    logger = SafeLogger(log_dir='./results', exp_name=exp_name, env_name='mlp', seed=args.seed,
                        fieldnames=['train_loss', 'val_loss'], debug=args.debug)
    base_save_dir = logger.log_dir if not args.debug else ''
    if not args.debug:
        os.makedirs(base_save_dir + '/ckpts', exist_ok=True)

    torch.random.manual_seed(args.seed)
    collate_batch([train_data[i] for i in range(16)])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_batch)
    all_test_y = np.array([x['y'] for x in test_data])
    all_val_y = np.array([x['y'] for x in val_data])
    all_train_y = np.array([x['y'] for x in train_data])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = create_model()
    model = model.to(device)

    # TODO: add Dropout?
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction='sum')

    # TODO: add resume capability + store best val in addition to raw state
    best_perf = np.inf
    for epoch in range(args.epochs):
        model.train(True)
        running_loss = 0
        for batch_idx, batch in tqdm(
            enumerate(train_loader),
            f'Training epoch {epoch}',
            total=len(train_loader)
        ):
            X = batch['X'].to(torch.float32).to(device)
            y = batch['y'].to(torch.float32).to(device)
            ragged_egos = [ego.to(torch.float32).to(device) for ego in batch['ragged_egos']]
            ragged_advs = [adv.to(torch.float32).to(device) for adv in batch['ragged_advs']]
            scene_ids = batch['scene_ids'].to(torch.float32).to(device)
            ragged_other_y = [other_y.to(torch.float32).to(device) for other_y in batch['ragged_other_y']]
            ragged_others = [other.to(torch.float32).to(device) for other in batch['ragged_others']]

            optimizer.zero_grad()
            # Ego is ignored, only adv is used
            output = model(X, ragged_egos, ragged_advs)
            loss = loss_fn(output, y)

            other_in = []
            for x_ in ragged_others:
                other_in.extend(x_)
            other_y = []
            for y_ in ragged_other_y:
                other_y.extend(y_)
            other_X = []
            for last_state, y_ in zip(X, ragged_other_y):
                other_X.extend(last_state.repeat(len(y_), 1))
            other_X = torch.stack(other_X)

            # always do the original ego loss
            running_loss += loss.item() 
            if len(other_y):
                other_y = torch.stack(other_y)
                other_output = model(other_X, other_in, other_in)
                other_loss = loss_fn(other_output, other_y)
                # TODO: consider loss weighting? 50% by ego, 50% by others...to keep ego as the more important?
                # TODO: consider weighting loss by GT score too...
                other_loss = other_loss * (len(X) / len(other_y))
                loss = other_loss + loss
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / all_train_y.size
        model.eval()
        with torch.no_grad():
            # Evaluate epoch performance, save results, etc.
            running_val_loss = 0
            for batch_idx, batch in tqdm(
                enumerate(val_loader),
                f'Validating epoch {epoch}',
                total=len(val_loader)
            ):
                X = batch['X'].to(torch.float32).to(device)
                y = batch['y'].to(torch.float32).to(device)
                ragged_egos = [ego.to(torch.float32).to(device) for ego in batch['ragged_egos']]
                ragged_advs = [adv.to(torch.float32).to(device) for adv in batch['ragged_advs']]
                scene_ids = batch['scene_ids'].to(torch.float32).to(device)

                output = model(X, ragged_egos, ragged_advs)
                loss = loss_fn(output, y)
                running_val_loss += loss.item()
            avg_val_loss = running_val_loss / all_val_y.size

        print()
        print('*********')
        print(f'Epoch {epoch} train loss: {avg_loss}, val loss: {avg_val_loss}')
        print('*********')
        logger.update([avg_loss, avg_val_loss], total_steps=epoch)
        if avg_val_loss <= best_perf:
            best_perf = avg_val_loss
            print(f'New best: {best_perf}')
        state = {
            'epoch': epoch,
            'best_val_loss': best_perf,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if not args.debug:
            torch.save(state, base_save_dir + '/ckpts/latest.pth')
            if avg_val_loss <= best_perf:
                torch.save(state, base_save_dir + '/ckpts/best.pth')
        

    # Do test evaluation, load from best.pth
    if args.debug:
        print('Debug mode, so skipping final test')
    else:
        best_model = create_model()
        best_model = best_model.to(device)
        best_path = base_save_dir + '/ckpts/best.pth'
        state = torch.load(best_path)
        best_model.load_state_dict(state['model'])
        best_model.eval()

        torch.random.manual_seed(args.seed)
        # TODO: store results for sc and diff separately
        with torch.no_grad():
            # Evaluate epoch performance, save results, etc.
            running_test_loss = 0
            all_pred_y = []
            for batch_idx, batch in tqdm(
                enumerate(test_loader),
                f'Testing epoch {state["epoch"]}',
                total=len(test_loader)
            ):
                X = batch['X'].to(torch.float32).to(device)
                y = batch['y'].to(torch.float32).to(device)
                ragged_egos = [ego.to(torch.float32).to(device) for ego in batch['ragged_egos']]
                ragged_advs = [adv.to(torch.float32).to(device) for adv in batch['ragged_advs']]
                scene_ids = batch['scene_ids'].to(torch.float32).to(device)

                output = best_model(X, ragged_egos, ragged_advs)
                all_pred_y.append(output.cpu().numpy())
                loss = loss_fn(output, y)
                running_test_loss += loss.item()
            avg_test_loss = running_test_loss / all_test_y.size
        print()
        print(f'Test loss from epoch {state["epoch"]}: {avg_test_loss}')
        # Random
        rand_avg = np.power(all_test_y - np.random.rand(*all_test_y.shape), 2).mean()
        mean_avg = np.power(all_test_y - all_train_y.mean(axis=0), 2).mean()
        print(f'Test loss with rand: {rand_avg}')
        print(f'Test loss with train-mean: {mean_avg}')

        all_pred_y = np.concatenate(all_pred_y, axis=0)

        print()
        sc_score_perf = np.abs(all_test_y - all_pred_y)[:, 0].mean()
        sc_score_rand = np.abs(all_test_y - np.random.rand(*all_test_y.shape))[:, 0].mean()
        sc_score_mean = np.abs(all_test_y - all_train_y.mean(axis=0))[:, 0].mean()

        diff_score_perf = np.abs(all_test_y - all_pred_y)[:, 1].mean()
        diff_score_rand = np.abs(all_test_y - np.random.rand(*all_test_y.shape))[:, 1].mean()
        diff_score_mean = np.abs(all_test_y - all_train_y.mean(axis=0))[:, 1].mean()
        print(f'model SC score abs: {sc_score_perf}, Diff score abs: {diff_score_perf}')
        print(f'rand SC score abs: {sc_score_rand}, Diff score abs: {diff_score_rand}')
        print(f'train-mean SC score abs: {sc_score_mean}, Diff score abs: {diff_score_mean}')

        # Now, we want to look at classification performance!

        # Reset batch size to 32 
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_batch)
        with torch.no_grad():
            # Metrics we care about are:
            # 1. Overall Accuracy and other classification metrics
            # 2. Lost criticality (i.e., actual SC score of selected - optimal SC score available)
            n_correct = 0
            n_seen = 0
            score_lost = []
            rand_score_lost = []

            for batch_idx, batch in tqdm(
                enumerate(test_loader),
                f'Testing classification performance from {state["epoch"]}',
                total=len(test_loader)
            ):
                X = batch['X'].to(torch.float32).to(device)
                y = batch['y'].to(torch.float32).to(device)
                ragged_egos = [ego.to(torch.float32).to(device) for ego in batch['ragged_egos']]
                ragged_advs = [adv.to(torch.float32).to(device) for adv in batch['ragged_advs']]
                scene_ids = batch['scene_ids'].to(torch.float32).to(device)
                assert len(torch.unique(scene_ids)) == 1, 'Mismatch in batch size/scene_id sorting'

                output = best_model(X, ragged_egos, ragged_advs)

                # For now, inspect overall, instead of just sc or just diff...
                overall_pred = output.mean(dim=1)
                # overall_pred = output[:, 0]
                pred_idx = overall_pred.argmax()

                overall_y = y.mean(dim=1)
                # overall_y = y[:, 0]
                optimal_score = overall_y.max()
                optimal_score_idxs = torch.arange(len(overall_y))[overall_y == optimal_score].to(optimal_score.device)

                n_seen += 1
                n_correct += pred_idx in optimal_score_idxs
                # Always positive
                score_lost.append((optimal_score - overall_y[pred_idx]).item())
                rand_score_lost.append(np.mean([(optimal_score - y).item() for y in overall_y]))

        score_lost = np.array(score_lost)
        print()
        print(f'Accuracy (model): {n_correct} / {n_seen}')
        print(f'Avg score lost (model): {np.mean(score_lost)}')
        print(f'Median score lost (model): {np.median(score_lost)}')

        print(f'Avg score lost (rand): {np.mean(rand_score_lost)}')
        print(f'Median score lost (rand): {np.median(rand_score_lost)}')
        pass
