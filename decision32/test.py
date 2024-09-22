import argparse
import os
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from saferl_plotter.logger import SafeLogger

from safeshift.measure_utils import shift_rotate

from train import Model, create_model, DecisionDataset, collate_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='idm32_all')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--extra_tag', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--run_number', type=int, default=0)

    args = parser.parse_args()

    # train_data = DecisionDataset(args.dataset_name, split='train', seed=args.seed)
    # val_data = DecisionDataset(args.dataset_name, split='val', seed=args.seed)
    test_data = DecisionDataset(args.dataset_name, split='test', seed=args.seed)

    extra_name = f'_{args.extra_tag}' if args.extra_tag else ''
    exp_name = f'{args.dataset_name}{extra_name}'

    load_dir = f'./results/{exp_name}_mlp-seed{args.seed}-{args.run_number}'

    torch.random.manual_seed(args.seed)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_batch)
    all_test_y = np.array([x['y'] for x in test_data])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_model = create_model()
    best_model = best_model.to(device)

    loss_fn = nn.MSELoss(reduction='sum')

    best_path = load_dir + '/ckpts/best.pth'
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
            f'Testing performance {state["epoch"]}',
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
    mean_avg = np.power(all_test_y - all_test_y.mean(axis=0), 2).mean()
    print(f'Test loss with rand: {rand_avg}')
    print(f'Test loss with test-mean: {mean_avg}')

    all_pred_y = np.concatenate(all_pred_y, axis=0)
    print()
    sc_score_perf = np.abs(all_test_y - all_pred_y)[:, 0].mean()
    sc_score_rand = np.abs(all_test_y - np.random.rand(*all_test_y.shape))[:, 0].mean()
    sc_score_mean = np.abs(all_test_y - all_test_y.mean(axis=0))[:, 0].mean()

    diff_score_perf = np.abs(all_test_y - all_pred_y)[:, 1].mean()
    diff_score_rand = np.abs(all_test_y - np.random.rand(*all_test_y.shape))[:, 1].mean()
    diff_score_mean = np.abs(all_test_y - all_test_y.mean(axis=0))[:, 1].mean()
    print(f'model SC score abs: {sc_score_perf}, Diff score abs: {diff_score_perf}')
    print(f'rand SC score abs: {sc_score_rand}, Diff score abs: {diff_score_rand}')
    print(f'test-mean SC score abs: {sc_score_mean}, Diff score abs: {diff_score_mean}')

    # Now, we want to look at classification performance!

    # Reset batch size to 32 
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_batch)
    # TODO: compare against ground truth selected labels...
    with torch.no_grad():
        # Metrics we care about are:
        # 1. Overall Accuracy and other classification metrics
        # 2. Lost criticality (i.e., actual SC score of selected - optimal SC score available)
        n_correct = 0
        n_seen = 0
        score_lost = []
        rand_score_lost = []

        sc_idxs = []
        diff_idxs = []
        overall_idxs = []

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
            pred_sc_idx, pred_diff_idx = output[:, 0].argmax(), output[:, 1].argmax()
            overall_idxs.append(overall_pred.argmax().item())
            sc_idxs.append(pred_sc_idx.item())
            diff_idxs.append(pred_diff_idx.item())
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
    sc_idxs = np.array(sc_idxs)
    diff_idxs = np.array(diff_idxs)
    overall_idxs = np.array(overall_idxs)
    print()
    print(f'Accuracy (model): {n_correct} / {n_seen}')
    print(f'Avg score lost (model): {np.mean(score_lost)}')
    print(f'Median score lost (model): {np.median(score_lost)}')

    print(f'Avg score lost (rand): {np.mean(rand_score_lost)}')
    print(f'Median score lost (rand): {np.median(rand_score_lost)}')
    pass
