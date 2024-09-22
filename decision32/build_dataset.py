import argparse
import numpy as np
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

from scipy.stats import wasserstein_distance
from scipy.special import softmax

from natsort import natsorted

from safeshift.measure_utils import get_tracks

# From https://github.com/oliviaguest/gini
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

# From output/ego_idm_adv_n1_save32*, want to build a prediction dataset of how "good" the selection process is

# TODO: Get source from ego_idm_normal_n1_train-seed0-0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Use in this folder, like python cat_convert_output.py --in_dir dir1 dir2 
    parser.add_argument('--in_dir', type=str, action='append', nargs='*', \
                        default=[])
    parser.add_argument('--out_dir', type=str, default='dataset')
    parser.add_argument('--dataset_name', type=str, default='idm32_all')
    args = parser.parse_args()

    in_dirs = args.in_dir
    if not len(in_dirs):
        in_dirs = [['../output/ego_idm_reactive_obs_all_adv_n1_save32_train-seed0-0',
                    '../output/ego_idm_reactive_obs_all_adv_n1_save32_eval-seed0-0']]
    in_dirs = [y for x in in_dirs for y in x]

    output_path = f'{args.out_dir}/{args.dataset_name}'
    os.makedirs(output_path, exist_ok=True)

    # A little preprocessing here:
    # 1. Clip to -1, 1 range
    # 2. Perform filtering on out of road/crash trajectories

    all_data = {}
    for in_dir in in_dirs:
        print('Processing', in_dir)
        full_path = f'{in_dir}/obs'
        all_obs = natsorted(os.listdir(full_path))

        for obs in tqdm(all_obs):
            in_path = f'{full_path}/{obs}' 
            data = np.load(in_path, allow_pickle=True)
            data = data.item()
            observations = data['obs']
            actions = data['actions']
            assert len(observations) == len(actions), 'Mismatch between observations and actions length'
            scene_id = obs.split('_traj')[0]
            all_data.setdefault(scene_id, [])
            all_data[scene_id].append(in_path)
        for v in all_data.values():
            assert len(v) == 32, '32 examples required for each scene'
        
    wds = []
    ginis = []
    max_scores = []
    all_scores = []
    all_sc_scores = []
    all_diff_scores = []
    all_last_states = []
    all_scene_ids = []
    all_gt_tracks = []
    all_ego_tracks = []
    all_adv_tracks = []

    all_gt_other_tracks = []
    all_other_tracks = []
    all_other_sc_scores = []
    all_other_diff_scores = []

    for scene_id, rollout_paths in tqdm(all_data.items(), 'Processing rollouts jointly per scene', total=len(all_data)):
        # TODO: decide on objective function here...linear weight of induced ego motion and/or safety criticality?
        rollouts = [np.load(x, allow_pickle=True).item() for x in rollout_paths]
        base = rollouts[0]
        gt_ego, gt_other_tracks, gt_other_ids = get_tracks({'tracks': base['normal_tracks']})
        assert np.all(gt_ego[:, -1]), 'Ego must be all valid'

        dones = []
        total_diffs = []
        last_states = []

        sc_scores = []
        for traj_idx, rollout in enumerate(rollouts):
            induced_ego, induced_other_tracks, induced_other_ids = get_tracks(rollout)
            assert np.all(induced_ego[:, -1]), 'Ego must be all valid'
            dones.append(rollout['done'])
            # Must be the un-filtered shapes (full)
            all_ego_tracks.append(induced_ego)
            all_gt_tracks.append(gt_ego)

            # TODO: focus on more than just adv_agent
            n_adv_tracks = np.sum(induced_other_ids == rollout['adv_agent'])
            if n_adv_tracks == 1:
                adv_idx = (induced_other_ids == rollout['adv_agent']).argmax()
                adv_data = induced_other_tracks[adv_idx]
                valid_adv_data = adv_data[adv_data[:, -1].astype(bool)]
                adv_track = valid_adv_data[:, :2]
            else:
                adv_track = None
            all_adv_tracks.append(adv_data if adv_track is not None else np.zeros((1, 10)))

            # TODO: store the ground truth selected index
            decoder_last_state = rollout['decoder_last_state']
            last_states.append(decoder_last_state)

            length = min(len(gt_ego), len(induced_ego))
            diff = np.linalg.norm(gt_ego[:length, :2] - induced_ego[:length, :2], axis=-1)
            total_diffs.append(diff.sum())
            # Want to characterize various outcomes:
            # 1. Safety Criticality
            # 2. Induced reactivity
            # 3. etc.
            
            if rollout['done'] == 'crash':
                sc_score = 1
            elif adv_track is None:
                sc_score = 0
            else:
                ego_overlap = induced_ego[adv_data[:, -1].astype(bool)][:, :2]
                # Scaling factor such that 2m away is roughly 0.75
                sc_score = min(1, np.exp(-np.linalg.norm(ego_overlap - adv_track, axis=-1).min() / 8))
            sc_scores.append(sc_score)
            all_scene_ids.append(scene_id)
            
            # Other agents to process
            tmp_gt_info = {}
            tmp_info = {}
            tmp_sc = {}
            tmp_diff = {}
            if len(induced_other_ids) - (adv_track is not None) > 0:
                # Need to access extra_induced_other_info to see if crash occurs
                for induced_other_id, induced_other_data in zip(induced_other_ids, induced_other_tracks):
                    if induced_other_id == rollout['adv_agent']:
                        continue
                    valid_other_data = induced_other_data[induced_other_data[:, -1].astype(bool)]
                    induced_other_track = valid_other_data[:, :2]

                    # gt_other_tracks, gt_other_ids
                    gt_other_idx = gt_other_ids == induced_other_id
                    if np.sum(gt_other_idx) == 1:
                        gt_other_idx = np.argmax(gt_other_idx)
                        gt_other_data = gt_other_tracks[gt_other_idx]
                        overlap_length = min(len(gt_other_data), len(induced_other_data))
                        gt_induced_overlap = induced_other_data[:overlap_length, -1].astype(bool) & \
                                             gt_other_data[:overlap_length, -1].astype(bool)
                        if not np.sum(gt_induced_overlap):
                            other_diff_score = -1
                        else:
                            valid_gt_track = gt_other_data[:overlap_length][gt_induced_overlap][:, :2]
                            tmp_induced_track = induced_other_data[:overlap_length][gt_induced_overlap][:, :2]
                            other_diff = np.linalg.norm(valid_gt_track - tmp_induced_track, axis=-1).sum()
                            other_diff_score = 1 - np.exp(-other_diff / 8)
                    else:
                        other_diff_score = -1
                    
                    ego_overlap = induced_ego[induced_other_data[:, -1].astype(bool)][:, :2]
                    did_crash = any([x[induced_other_id]['crash'] for x in rollout['extra_other_info'] if induced_other_id in x])
                    if did_crash:
                        other_sc_score = 1
                    elif not len(ego_overlap):
                        other_sc_score = 0
                    else:
                        other_sc_score = min(1, np.exp(-np.linalg.norm(ego_overlap - induced_other_track, axis=-1).min() / 8))
                    # Ensure that it is the 
                    tmp_gt_info[induced_other_id] = gt_other_data
                    tmp_info[induced_other_id] = induced_other_data
                    tmp_sc[induced_other_id] = other_sc_score
                    tmp_diff[induced_other_id] = other_diff_score
                    
            all_gt_other_tracks.append(tmp_gt_info)
            all_other_tracks.append(tmp_info)
            all_other_sc_scores.append(tmp_sc)
            all_other_diff_scores.append(tmp_diff)

        # if len(np.unique(dones)) < 2:
        #     continue
        
        # Such that an induced ego total deviation of 10m is roughly 0.75
        diff_scores = 1 - np.exp(-np.array(total_diffs) / 8)

        # Equally weighting induced ego behavior change
        scores = 0.5 * np.array(sc_scores) + 0.5 * np.array(diff_scores)

        max_scores.append(scores.max())
        all_scores.extend(list(scores))
        all_sc_scores.extend(list(sc_scores))
        all_diff_scores.extend(list(diff_scores))
        all_last_states.extend(last_states)
        plt.plot(np.arange(len(scores)), natsorted(scores), color='r', alpha=0.02)

        # Filter away boring scenes
        if scores.max() < 0.2:
            continue

        # WD vs. a uniform dist
        uniform_dist = np.array([1.0] * len(scores))
        ideal_dist = np.array([-1e100] * (len(scores - 1)) + [1e100])
        scores = softmax(diff_scores)
        uniform_dist = softmax(uniform_dist)
        ideal_dist = softmax(ideal_dist)

        wd_scale_factor = wasserstein_distance(ideal_dist, uniform_dist)
        wds.append(wasserstein_distance(scores, uniform_dist) / wd_scale_factor)

        gini_scale_factor = gini(ideal_dist)
        ginis.append(gini(scores) / gini_scale_factor)
        

    # plt.savefig('scores.png')
    # plt.clf()
    # plt.hist(wds, bins=20)
    # plt.savefig('wd.png')
    # plt.clf()
    # plt.hist(ginis, bins=20)
    # plt.savefig('gini.png')
    # plt.clf()
    # plt.hist(all_scores, bins=20)
    # plt.savefig('all_scores_hist.png')
    # plt.clf()
    # plt.hist(all_sc_scores, bins=20)
    # plt.savefig('all_sc_scores_hist.png')
    # plt.clf()
    # plt.hist(all_diff_scores, bins=20)
    # plt.savefig('all_diff_scores_hist.png')
    # plt.clf()
    # plt.hist(max_scores, bins=20)
    # plt.savefig('max_scores_hist.png')

    out_dict = {
        'last_states': all_last_states,
        'sc_scores': all_sc_scores,
        'diff_scores': all_diff_scores,
        'other_sc_scores': all_other_sc_scores,
        'other_diff_scores': all_other_diff_scores,
        'scene_ids': all_scene_ids,
        'original_ego_tracks': all_gt_tracks,
        'induced_ego_tracks': all_ego_tracks,
        'original_other_tracks': all_gt_other_tracks,
        'induced_other_tracks': all_other_tracks,
        'adv_tracks': all_adv_tracks,
    }
    output_file = f'{output_path}/data.npy'
    print('Saving data now')
    np.save(output_file, out_dict)


