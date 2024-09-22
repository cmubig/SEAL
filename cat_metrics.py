import argparse
import numpy as np
import os
import glob
from natsort import natsorted
from matplotlib import pyplot as plt
import scipy.stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import pandas as pd
from tqdm import tqdm
import warnings
import re
import json
import seaborn as sns

from safeshift.measure_utils import get_tracks, interaction_measures, possible_leading

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 1000)

REPLAY = 'None (GT)'
CAT_NO_ADV = 'TD3'
CAT_NO_ADV_RESKILL = 'No Adv'
CAT_HEURISTIC = 'CAT Rule-Based Adv (TD3)'
CAT_OPEN = 'CAT Open-Loop Adv (TD3)'
CAT_CLOSED = 'CAT (TD3)'
CAT_CLOSED_RESKILL = 'CAT'
GOOSE = 'GOOSE (TD3)'
GOOSE_RESKILL = 'GOOSE'
OURS = 'SEAL'

OUTPUT_ORDER = [
    '_normal_n1_eval', #SIMPLE
    '_normal_n1_hard_rand_eval', #HARD
    '_adv_n5_goose_eval', #GOOSE
    '_adv_n5_eval', #CAT
    '_adv_n5_skill_idm_all_adv_prior_learned_obj_eval', #SEAL
]

# Establishes the order for table creation too; order matters to
ABLATIONS = {
    'model_cat_reskill_skill_idm_all_adv_prior_learned_obj_initial': '$\\mathbf{' + OURS + '}$: Adv Skill Prior + Learned Obj',
    'model_cat_reskill_skill_idm_all_learned_obj_initial': 'Benign Skill Prior + Learned Obj',
    'model_cat_reskill_idm_learned_obj_initial': 'IDM Adv + Learned Obj',
    'model_cat_reskill_learned_obj_initial': 'TrajPred Adv + Learned Obj',
    'model_cat_reskill_skill_idm_all_adv_prior_initial': 'Adv Skill Prior + Heuristic Obj',
    'model_cat_reskill_skill_idm_all_initial': 'Benign Skill Prior + Heuristic Obj',
    'model_cat_reskill_idm_shared_initial': 'IDM Adv + Heuristic Obj',
    'model_cat_reskill_shared_initial': 'CAT: TrajPred Adv + Heuristic Obj',
}

CAPTION = \
"""Results on real and adversarially-perturbed scenes. ``Normal'' are WOMD
scenes with basic interactive agents labeled by Waymo; ``Hard'' refers to
SafeShift-mined real scenes in WOMD. Adversarially-perturbed scenes use
``Normal'' as base scenarios, for both training and eval settings. All trained
ego models utilize ReSkill."""

ABLATION_CAPTION = \
"""Ablations on our scenario perturbation method. All models utilize ReSkill."""


QUALITY_CAPTION = \
"""Scenario generation quality; results are averaged over all tested ego
models. WD measures are Wasserstein distances over adversary behavior; a lower
value indicates greater realism. A lower Ego Success is better, as this table
assesses the effectiveness of safety critical scenario generation."""

def pad_percent(value, n=5):
    return f'{value:.1%}'.rjust(n, '0')
def pad_nonpercent(value, n=5):
    return f'{value:.2f}'.rjust(n, '0')

# https://stackoverflow.com/a/70396916 for more info
def mean_confidence_interval(data, confidence=0.95, use_percent=True, std_only=True, agg_metric='mean'):
    # Check for string data types; return unmodified
    try:
        a = 1.0 * np.array(data)
    except Exception:
        return None, None, None, str(data[0])

    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    if agg_metric == 'median':
        m = np.median(a)
    elif agg_metric == 'iqm' and len(a) >= 4:
        if len(a) >= 4:
            a = np.array(natsorted(a))
            off = len(a) // 4
            a = a[off:-off]
            m = np.mean(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    if np.isnan(h):
        h = 0
    plus_minus = '\u00B1'
    if std_only:
        if agg_metric == 'mean':
            h = np.std(a)
        elif agg_metric == 'median':
            low, high = np.percentile(a, [25, 75])
            h = high - low
        elif agg_metric == 'iqm':
            # A is already the middle 50%
            h = a[-1] - a[0]
        if use_percent:
            return m, m-h, m+h, f'{pad_percent(m, n=5)} ({pad_percent(h, n=4)})'
        else:
            return m, m-h, m+h, f'{pad_nonpercent(m, n=4)} ({pad_nonpercent(h, n=4)})'
    else:
        if use_percent:
            return m, m-h, m+h, f'{pad_percent(m, n=5)} {plus_minus} {pad_percent(h, n=4)}'
        else:
            return m, m-h, m+h, f'{pad_nonpercent(m, n=4)} {plus_minus} {pad_nonpercent(h, n=4)}'


def get_normal_path(path):
    inner_path = path.split('/')[1]
    if '_normal_n1_' in inner_path:
        return path
    # Consume until either eval, train, or custom
    # Either after n1 or after n5
    env = inner_path.split('_')[-1]
    assert not ('_adv_n1_' in inner_path and '_adv_n5_' in inner_path), 'Invalid path'

    if '_adv_n1_' in inner_path:
        ego_name = inner_path.split('_adv_n1_')[0]
    elif '_adv_n5_' in inner_path:
        ego_name = inner_path.split('_adv_n5_')[0]
    else:
        raise ValueError('Missing adv_n1 and adv_n5 and normal_n1')
    
    new_inner_path = f'{ego_name}_normal_n1_{env}'
    return path.replace(inner_path, new_inner_path)

def get_attributes(name, args):
    if '_normal_n1' in name:
        method = name.split('_normal_n1')[0]
    elif '_adv_n1' in name:
        method = name.split('_adv_n1')[0]
    elif '_adv_n5' in name:
        method = name.split('_adv_n5')[0]
    
    ret_attrs = {}

    if method == 'ego_idm':
        ret_attrs['ego_method'] = 'IDM'
    elif method == 'ego_replay':
        ret_attrs['ego_method'] = REPLAY
    elif method.startswith('model_cat_initial') or method.startswith('model_cat_5e6_initial'):
        ret_attrs['ego_method'] = CAT_CLOSED
    elif method.startswith('model_replay_initial') or method.startswith('model_replay_5e6_initial'):
        ret_attrs['ego_method'] = CAT_NO_ADV
    elif method.startswith('model_cat_heuristic_initial') or method.startswith('model_cat_heuristic_5e6_initial'):
        ret_attrs['ego_method'] = CAT_HEURISTIC
    elif method.startswith('model_cat_open_initial') or method.startswith('model_cat_open_5e6_initial'):
        ret_attrs['ego_method'] = CAT_OPEN
    elif method.startswith('model_replay_reskill_initial') or \
         method.startswith('model_replay_reskill_5e6_initial'):
        ret_attrs['ego_method'] = CAT_NO_ADV_RESKILL
    elif method.startswith('model_cat_reskill_shared_initial') or \
         method.startswith('model_cat_reskill_shared_5e6_initial'):
        ret_attrs['ego_method'] = CAT_CLOSED_RESKILL
    elif method.startswith('model_cat_goose_initial'):
        ret_attrs['ego_method'] = GOOSE
    elif method.startswith('model_cat_reskill_goose_initial'):
        ret_attrs['ego_method'] = GOOSE_RESKILL
    elif method.startswith('model_cat_reskill_skill_idm_all_adv_prior_learned_obj_initial') or \
         method.startswith('model_cat_reskill_skill_idm_all_adv_prior_learned_obj_5e6_initial'):
        ret_attrs['ego_method'] = OURS
    else:
        ret_attrs['ego_method'] = 'Ablation'
    
    if 'ablation' in args.in_file:
        for ablation_k, ablation_v in ABLATIONS.items():
            if method.startswith(ablation_k):
                ret_attrs['ego_method'] = ablation_v
                break

    # Guaranteed attributes
    if method.startswith('ego_replay'):
        ret_attrs['method_type'] = 'replay'
        ret_attrs['train_gen_type'] = '-'
    elif method.startswith('ego_idm'):
        ret_attrs['method_type'] = 'idm'
        ret_attrs['train_gen_type'] = '-'
    elif method.startswith('model_skill'):
        ret_attrs['method_type'] = 'skill'
        ret_attrs['train_gen_type'] = '-'
    elif method.startswith('model_replay_reskill'):
        ret_attrs['method_type'] = 'reskill'
        ret_attrs['train_gen_type'] = 'normal'
    elif method.startswith('model_replay'):
        ret_attrs['method_type'] = 'td3'
        ret_attrs['train_gen_type'] = 'normal'
    elif method.startswith('model_cat_reskill'):
        ret_attrs['method_type'] = 'reskill'
        ret_attrs['train_gen_type'] = 'adv'
    elif method.startswith('model_cat'):
        ret_attrs['method_type'] = 'td3'
        ret_attrs['train_gen_type'] = 'adv'
    
    # Optional attributes
    if method.startswith('model') and 'initial' in method:
        train_tag = None
        if '5e6_shared_initial' in method:
            train_tag = '5e6_shared_initial'
            ret_attrs['train_time'] = '5e6'
        elif '5e6_initial' in method:
            train_tag = '5e6_initial'
            ret_attrs['train_time'] = '5e6'
        elif 'shared_initial' in method:
            train_tag = 'shared_initial'
            ret_attrs['train_time'] = '1e6'
        else:
            train_tag = 'initial'
            ret_attrs['train_time'] = '1e6'

        if 'heuristic' in method:
            ret_attrs['adv_mode'] = 'heuristic'
        elif 'goose' in method:
            ret_attrs['adv_mode'] = 'goose'
        elif 'open' in method:
            ret_attrs['adv_mode'] = 'open'
        elif 'replay' not in method:
            ret_attrs['adv_mode'] = 'densetnt'
        
        if 'guided' in method:
            ret_attrs['adv_mode'] += '-guided'

        if 'current_adv_prior' in method:
            ret_attrs['hybrid_adv_mode'] = 'current-advprior'
        elif 'current' in method:
            ret_attrs['hybrid_adv_mode'] = 'current'
        elif 'skill_idm_all_adv_prior' in method:
            ret_attrs['hybrid_adv_mode'] = 'skill-advprior'
        elif 'skill_idm_all' in method:
            ret_attrs['hybrid_adv_mode'] = 'skill-benign'
        elif f'idm_{train_tag}' in method or f'idm_learned_obj_{train_tag}' in method or \
             f'idm_learned_obj_sc_{train_tag}' in method or f'idm_learned_obj_diff_{train_tag}' in method:
            ret_attrs['hybrid_adv_mode'] = 'idm'
        
        if 'co_var' in method:
            ret_attrs['hybrid_co'] = 'var'
        elif 'co_inf' in method:
            ret_attrs['hybrid_co'] = 'inf'
        elif 'co_25' in method:
            ret_attrs['hybrid_co'] = '25'
        elif 'hybrid_adv_mode' in ret_attrs:
            ret_attrs['hybrid_co'] = '10'

        if 'learned_obj_sc' in method:
            ret_attrs['adv_objective'] = 'learned-sc'
        elif 'learned_obj_diff' in method:
            ret_attrs['adv_objective'] = 'learned-diff'
        elif 'learned_obj' in method:
            ret_attrs['adv_objective'] = 'learned'
        elif ('goose' not in method and 'heuristic' not in method and 'replay' not in method and 'open' not in method):
            ret_attrs['adv_objective'] = 'overlap'
    
    return ret_attrs
    

def get_data(files, args):

    all_arrive = 0
    all_crash = 0
    all_out_of_road = 0
    all_route_completion = 0
    all_adv_out_of_road = 0
    all_adv_other_crash = 0
    all_adv_t = 0

    # Just looking at default_agent for now
    all_speed = []
    all_acc = []
    all_dists = []
    all_yaw_rate = []
    all_sc_score = []
    all_diff_score = []
    all_mean_score = []
    all_gen_times = 0

    # Now, looking at the adversarial vehicle
    all_adv_speed = []
    all_adv_acc = []
    all_adv_dists = []
    all_adv_yaw_rate = []

    # ttc_into_ego, thw_into_ego, drac_into_ego, ttc_into_other, thw_into_other, drac_into_other, relative_mttcp, interaction_score = measures
    n_int = 8
    keep_max = args.keep_max
    interaction_defaults = [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, 0] if keep_max else [[] for _ in range(n_int)]
    interaction_funcs = [np.min, np.min, np.max, np.min, np.min, np.max, np.min, np.sum]
    all_interaction_measures = [[] for _ in range(n_int)]

    for f in tqdm(files, 'Processing files', leave=False):
        data = np.load(f, allow_pickle=True).item()
        sc_score = 0
        diff_score = 0

        if data['done'] == 'crash':
            all_crash += 1
            sc_score = 1
        elif data['done'] == 'out_of_road':
            all_out_of_road += 1
        elif data['done'] == 'arrive':
            all_arrive += 1
        else:
            #raise ValueError('Invalid done reason')
            pass
        if f == files[-1]:
            all_gen_times += data['gen_time']

        all_route_completion += data['route_completion']
        tracks = data['tracks']
        key = 'default_agent'
        # N x 8 shape (pos_x, pos_y, vel_x, vel_y, heading_theta, length, width, crash)
        ego_data = np.array([x[key] for x in tracks])
        acc = np.zeros((ego_data.shape[0] - 1, 2))
        acc = (ego_data[1:, 2:4] - ego_data[:-1, 2:4]) * 10

        all_speed.extend(np.linalg.norm(ego_data[:, 2:4], axis=-1))
        all_acc.extend(np.linalg.norm(acc, axis=-1))

        # In radians/second, so multiply by 10 since measurements at 10 Hz
        yaw_rate = (ego_data[1:, 4] - ego_data[:-1, 4])
        yaw_rate = (yaw_rate + np.pi) % (2*np.pi) - np.pi
        yaw_rate *= 10
        # TODO: save yaw_rate stuff
        all_yaw_rate.extend(yaw_rate)

        # Format = (x, y, vx, vy, heading_theta, length, width, crash, new_id, valid)
        ego_track, other_tracks, other_ids = get_tracks(data)

        normal_path = get_normal_path(f)
        normal_ego_track, _, _ = get_tracks(np.load(normal_path, allow_pickle=True).item())
        length = min(len(normal_ego_track), len(ego_track))
        # Mean instead of sum, to allow for early termination fairness, etc.
        diff = np.linalg.norm(normal_ego_track[:length, :2] - ego_track[:length, :2], axis=-1).mean() * 100
        # Such that an induced ego total deviation of 10m is roughly 0.75
        diff_score = 1 - np.exp(-diff / 8)
        all_diff_score.append(diff_score)

        # Can be either 0 or 1
        # TODO: do adversary specific stuff, now that we have the track. Interaction realism, etc.
        n_adv_tracks = np.sum(other_ids == data['adv_agent'])
        if n_adv_tracks == 1:
            adv_idx = (other_ids == data['adv_agent']).argmax()
            adv_data = other_tracks[adv_idx]
            if sc_score != 1:
                ego_overlap = ego_track[adv_data[:, -1].astype(bool)][:, :2]
                # Scaling factor such that 2m away is roughly 0.75
                sc_score = min(1, np.exp(-np.linalg.norm(ego_overlap - adv_data[adv_data[:, -1].astype(bool)][:, :2],
                                                         axis=-1).min() / 8))
            valid_adv_data = adv_data[adv_data[:, -1].astype(bool)]
            if len(valid_adv_data) >= 2:
                adv_acc = np.zeros((valid_adv_data.shape[0] - 1, 2))
                adv_acc = (valid_adv_data[1:, 2:4] - valid_adv_data[:-1, 2:4]) * 10

                all_adv_speed.extend(np.linalg.norm(valid_adv_data[:, 2:4], axis=-1))
                all_adv_acc.extend(np.linalg.norm(adv_acc, axis=-1))

                # In radians/second, so multiply by 10 since measurements at 10 Hz
                adv_yaw_rate = (valid_adv_data[1:, 4] - valid_adv_data[:-1, 4])
                adv_yaw_rate = (adv_yaw_rate + np.pi) % (2*np.pi) - np.pi
                adv_yaw_rate *= 10
                all_adv_yaw_rate.extend(yaw_rate)

                # TODO: remove if, replace with assertion
                assert 'extra_other_info' in data, 'extra_other_info must be provided'
                adv_out_of_road = [x[data['adv_agent']]['out_of_road'] for x in data['extra_other_info'] if data['adv_agent'] in x]
                adv_other_crash = [x[data['adv_agent']]['crash'] for x in data['extra_other_info'] if data['adv_agent'] in x]
                all_adv_t += len(adv_other_crash)
                if np.any(adv_out_of_road):
                    all_adv_out_of_road += np.sum(adv_out_of_road)
                    # all_adv_out_of_road += 1
                if np.any(adv_other_crash):
                    if not data['done'] == 'crash':
                        all_adv_other_crash += np.sum(adv_other_crash)
                    else:
                        all_adv_other_crash += np.sum(adv_other_crash[:-1])
                    # excess_crash = (not data['done'] == 'crash') or np.sum(adv_other_crash) > 1
                    # if excess_crash:
                    #   all_adv_other_crash += excess_crash

                # TODO: calculate adv_dist differently? 
                adv_dists = np.linalg.norm(valid_adv_data[:, :2] - other_tracks[:, adv_data[:, -1].astype(bool), :2], axis=-1)
                adv_dist_valid = other_tracks[:, adv_data[:, -1].astype(bool), -1]
                adv_dists[~adv_dist_valid.astype(bool)] = np.inf
                adv_dists[adv_dists == 0] = np.inf
                adv_dists = adv_dists.min(axis=0)
                adv_dists = adv_dists[~np.isinf(adv_dists)]
                if adv_dists.size:
                    all_adv_dists.extend(adv_dists)
                # adv_dists = np.linalg.norm(valid_adv_data[:, :2] - ego_data[adv_data[:, -1].astype(bool), :2], axis=-1)
                # all_adv_dists.extend(adv_dists)
        all_sc_score.append(sc_score)
        all_mean_score.append(0.5*sc_score + 0.5*diff_score)

        scene_measures = np.copy(interaction_defaults).tolist()
        scene_measure_set = [False] * len(scene_measures)
        if args.interaction:
            possible_ego_lf = possible_leading(ego_track, other_tracks, heading_threshold=args.heading_threshold, voronoi=args.voronoi) 
            possible_other_lfs = []
            for track_idx, other_track in enumerate(other_tracks):
                tmp_other_tracks = np.copy(other_tracks)
                tmp_other_tracks[track_idx] = ego_track
                possible_other_lf = possible_leading(other_track, tmp_other_tracks, heading_threshold=args.heading_threshold, voronoi=args.voronoi)
                possible_other_lfs.append(possible_other_lf)
            

            for other_track, possible_other_lf in zip(other_tracks, possible_other_lfs):
                # TODO: add in collisions, then finally aggregate into score like in SafeShift!
                measures = interaction_measures(ego_track, other_track, possible_ego_lf, possible_other_lf, 
                                                mttcp_threshold=args.mttcp_threshold)
                for i, measure in enumerate(measures):
                    if len(measure):
                        traj_val = [interaction_funcs[i](measure)]
                        if keep_max:
                            scene_measures[i] = interaction_funcs[i]([scene_measures[i], *traj_val])
                            if interaction_funcs[i] == np.sum:
                                # TODO: regularize?
                                # denom = np.sqrt(len(other_tracks + 1))
                                denom = 1
                                scene_measures[i] /= denom
                        else:
                            scene_measures[i].extend(measure)
                        scene_measure_set[i] = True

        for (i, measure), measure_set in zip(enumerate(scene_measures), scene_measure_set):
            if measure_set:
                func = all_interaction_measures[i].append if keep_max else all_interaction_measures[i].extend
                func(measure)
        ego_positions = np.array([x[key] for x in tracks])[:, :2]

        all_keys = set()
        [all_keys.update(x.keys()) for x in tracks]
        default_val = np.array([np.inf] * tracks[0][key].shape[-1])
        other_positions = np.array([[x[k] if k in x else default_val for x in tracks] for k in all_keys if k != key])[..., :2]

        # Account for no nearby other vehicle the whole time
        if other_positions.size:
            other_dists = ego_positions[np.newaxis, :, :] - other_positions
            other_dists = np.linalg.norm(other_dists, axis=-1)
            nearest_dists = np.min(other_dists, axis=0)
            # Account for no nearby other vehicle at a given timestep
            nearest_dists[np.isinf(nearest_dists)] = 0
        else:
            nearest_dists = np.zeros((ego_positions.shape[0],))
        all_dists.extend(nearest_dists)

    
    all_speed = np.array(all_speed)
    all_acc = np.array(all_acc)
    all_dists = np.array(all_dists)
    all_yaw_rate = np.array(all_yaw_rate)

    all_adv_speed = np.array(all_adv_speed)
    all_adv_acc = np.array(all_adv_acc)
    all_adv_dists = np.array(all_adv_dists)
    all_adv_yaw_rate = np.array(all_adv_yaw_rate)

    all_crash = all_crash/len(files)
    all_out_of_road = all_out_of_road/len(files)
    all_arrive = all_arrive/len(files)
    all_route_completion = all_route_completion/len(files)
    all_gen_times = all_gen_times/len(files)
    all_adv_other_crash = all_adv_other_crash/all_adv_t
    all_adv_out_of_road = all_adv_out_of_road/all_adv_t

    all_interaction_measures = [np.mean(x) for x in all_interaction_measures]

    all_sc_score = np.array(all_sc_score).mean()
    all_diff_score = np.array(all_diff_score).mean()
    all_mean_score = np.array(all_mean_score).mean()

    return {'distribution': [all_speed, all_acc, all_dists, all_yaw_rate],
            'adv_distribution': [all_adv_speed, all_adv_acc, all_adv_dists, all_adv_yaw_rate], 
            'basic': [all_crash, all_out_of_road, all_arrive, all_adv_other_crash, all_adv_out_of_road, all_route_completion, all_gen_times, all_sc_score, all_diff_score, all_mean_score, *all_interaction_measures]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='output/ego_replay_normal_n1_eval-seed0-0')
    parser.add_argument('--other_paths', type=str, default='output/*_eval-seed*-0')
    parser.add_argument('--in_file', type=str, default='metrics/in_main.txt')

    parser.add_argument('--metric', type=str, default='wd', choices=['wd', 'jsd', 'rl'], help='Distribution metric choice: wasserstein, jensen-shannon, relative likelihood')

    parser.add_argument('--keep_max', action='store_true', help='Whether to keep max or all interaction measures')
    parser.add_argument('--interaction', action='store_true', help='Compute interaction in the first place')
    parser.add_argument('--heading_threshold', type=float, default=30, help='Heading diff limit for leader-follower, 0-180')
    parser.add_argument('--mttcp_threshold', type=float, default=1.0, help='Minimum distance for a potential conflict point')
    parser.add_argument('--voronoi', action='store_true', help='Utilize voronoi diagram for leader-follower limit')
    parser.add_argument('--median', action='store_true', help='Use median instead of mean for groups')
    parser.add_argument('--iqm', action='store_true', help='Use interquartile mean instead of mean/median for groups')

    parser.add_argument('--gen_types', default='metrics/gen_types_main.json', help='JSON of gen_types map')
    
    args = parser.parse_args()
    assert not args.iqm or not args.median, 'At most one of iqm or median is allowed'
    agg_metric = 'median' if args.median else 'iqm' if args.iqm else 'mean'

    assert os.path.exists(args.gt_path), 'gt source must exist'
    use_median = args.median

    # Use ggplot style to get its color cycle
    plt.style.use('ggplot')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Revert to the default style
    plt.style.use('default')

    # Apply ggplot colors to default style
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    def get_files(dir):
        assert os.path.exists(os.path.join(dir, 'obs')), 'observations must be stored from running cat_advgen'
        files = glob.glob(os.path.join(dir, 'obs', 'adv_*.npy'))
        assert len(files) >= 100, 'At least 100 obs must be stored'
        files = natsorted(files)
        return files

    # TODO: improve speed by caching leader/follower relations

    # First, let's compute a few basic metrics as in MixSim: velocity and acceleration
    # More complicated distributional metrics, like agent-agent things, and agent-map things, can be computed later.
    print('Processing base model')
    gt_data = get_data(get_files(args.gt_path), args)

    #other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x]
    #other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x and x != args.gt_path]
    #other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x and '_adv' in x]
    # other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x and '_n5' in x and 'full' not in x]
    #other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x and '_n5' in x]
    #other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x and 'full' not in x]
    # other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x and 'full' not in x \
    #               and 'reactive' not in x and 'no_prior' not in x and 'idm_small' not in x]

    # other_dirs = [x for x in natsorted(glob.glob(args.other_paths)) if '_eval' in x and \
    #               'reactive' not in x and 'no_prior' not in x and 'idm_small' not in x and 
    #               ('ego_replay' in x or 'ego_idm' in x or 'cat_initial' in x or 'cat_reskill_initial' in x)]
    other_dirs = [x for x in natsorted(glob.glob(args.other_paths))]
    
    if args.in_file is not None:
        filtered_dirs = []
        patterns = []
        with open(args.in_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not len(line):
                    continue
                try:
                    pattern = re.compile(line)
                    patterns.append(pattern)
                except Exception as e:
                    continue
        for other_dir in other_dirs:
            for pattern in patterns:
                if pattern.search(other_dir):
                    filtered_dirs.append(other_dir)
        other_dirs = filtered_dirs

    # Key order matters
    with open(args.gen_types, 'r') as f:
        gen_type_map = json.load(f)
    filtered_dirs = []
    gen_types = []
    for other_dir in other_dirs:
        for k, v in gen_type_map.items():
            if k in other_dir:
                gen_types.append(v)
                filtered_dirs.append(other_dir)
                break
    other_dirs = filtered_dirs

    dir_keys = {k: k.split('/')[-1].split('_eval')[0] for k in other_dirs}
    metrics = {dir_keys[k]: [] for k in other_dirs}
    for other_dir, gen_type in tqdm(zip(other_dirs, gen_types), 'Processing models', total=len(gen_types)):
        other_files = get_files(other_dir)
        other_data = get_data(other_files, args)
        for gt, adv_gt, other, adv_other in zip(gt_data['distribution'], gt_data['adv_distribution'], other_data['distribution'], other_data['adv_distribution']):
            # Same epsilon as Waymo Sim Agents
            eps = 0.1
            gt_histogram, gt_bins = np.histogram(gt, bins=int(np.ceil(np.sqrt(len(gt)))))
            gt_histogram = gt_histogram + eps
            gt_histogram /= np.sum(gt_histogram)

            other_histogram, _ = np.histogram(other, bins=gt_bins)
            other_histogram = other_histogram + eps
            other_histogram /= np.sum(other_histogram)

            adv_gt_histogram, _ = np.histogram(adv_gt, bins=gt_bins)
            adv_gt_histogram = adv_gt_histogram + eps
            adv_gt_histogram /= np.sum(adv_gt_histogram)

            adv_other_histogram, _ = np.histogram(adv_other, bins=gt_bins)
            adv_other_histogram = adv_other_histogram + eps
            adv_other_histogram /= np.sum(adv_other_histogram)

            if args.metric == 'jsd':
                metrics[dir_keys[other_dir]].append(jensenshannon(gt_histogram, other_histogram))
                metrics[dir_keys[other_dir]].append(jensenshannon(adv_gt_histogram, adv_other_histogram))
            elif args.metric == 'wd':
                wd = wasserstein_distance(u_values=gt_bins[:-1], v_values=gt_bins[:-1],
                                        u_weights=gt_histogram, v_weights=other_histogram)
                metrics[dir_keys[other_dir]].append(wd)
                if np.isnan(adv_other_histogram).any():
                    adv_wd = 0
                else:
                    adv_wd = wasserstein_distance(u_values=gt_bins[:-1], v_values=gt_bins[:-1],
                                            u_weights=adv_gt_histogram, v_weights=adv_other_histogram)
                metrics[dir_keys[other_dir]].append(adv_wd)
            else:
                ll_other = np.sum(gt_histogram * np.log(other_histogram))
                ll_max = np.sum(gt_histogram * np.log(gt_histogram))
                likelihood = np.exp(ll_other - ll_max)
                metrics[dir_keys[other_dir]].append(likelihood)
                if np.isnan(adv_other_histogram).any():
                    adv_likelihood = 0
                else:
                    adv_ll = np.sum(adv_gt_histogram * np.log(adv_other_histogram))
                    adv_ll_max = np.sum(adv_gt_histogram * np.log(adv_gt_histogram))
                    adv_likelihood = np.exp(adv_ll - adv_ll_max)
                metrics[dir_keys[other_dir]].append(adv_likelihood)
        for other in other_data['basic']:
            metrics[dir_keys[other_dir]].append(other)

        # crash, out_of_road, arrive
        for gt, other in zip(gt_data['basic'][:5], other_data['basic'][:5]):
            bins = [-0.5, 1.5]
            n_gt, n_other = int(gt * len(other_files)), int(other * len(other_files))
            gt_binary = np.array([0] * len(other_files))
            other_binary = np.array([0] * len(other_files))
            gt_binary[:n_gt] = 1
            other_binary[:n_other] = 1
            bins = [-0.5, 0.5, 1.5]
            
            eps = 0.1
            gt_histogram, _ = np.histogram(gt_binary, bins)
            gt_histogram = gt_histogram + eps
            gt_histogram /= np.sum(gt_histogram)

            other_histogram, _ = np.histogram(other_binary, bins)
            other_histogram = other_histogram + eps
            other_histogram /= np.sum(other_histogram)

            if args.metric == 'jsd':
                metrics[dir_keys[other_dir]].append(jensenshannon(gt_histogram, other_histogram))
            elif args.metric == 'wd':
                wd = wasserstein_distance(u_values=bins[:-1], v_values=bins[:-1],
                                        u_weights=gt_histogram, v_weights=other_histogram)
                metrics[dir_keys[other_dir]].append(wd)
            else:
                ll_other = np.sum(gt_histogram * np.log(other_histogram))
                ll_max = np.sum(gt_histogram * np.log(gt_histogram))
                likelihood = np.exp(ll_other - ll_max)
                metrics[dir_keys[other_dir]].append(likelihood)
        metrics[dir_keys[other_dir]].append(gen_type)
    
    df = pd.DataFrame(metrics).transpose()
    wd_metrics = ['speed', 'adv_speed', 'acc', 'adv_acc', 'dist', 'adv_dist', 'yaw', 'adv_yaw']
    wd_suff_metrics = [f'{x}_{args.metric}' for x in wd_metrics]
    interaction_names = ['ttc_into_ego', 'thw_into_ego', 'drac_into_ego', 'ttc_into_other', 'thw_into_other', 'drac_into_other', 'delta_mttcp', 'interaction_score']
    basic_metrics = ['crash', 'out_of_road', 'arrive', 'adv_other_crash', 'adv_out_of_road', 'route_completion', 'gen_time',
                     'sc_score', 'diff_score', 'mean_score', *interaction_names, 
                     f'crash_{args.metric}', f'out_of_road_{args.metric}', f'arrive_{args.metric}',
                     f'adv_other_crash_{args.metric}', f'adv_out_of_road_{args.metric}',
                     'gen_type']

    df.columns = wd_suff_metrics + basic_metrics
    basic_metrics.append('adjusted_crash')
    basic_metrics.append('adv_meta_' + args.metric)
    
    # Idea: high arrival rate #1, but crashes are worse than out of road
    df['adjusted_crash'] = (df['crash'] / (1 - df['out_of_road']))

    # kinematic_realism = (df['acc_' + args.metric] + df['speed_' + args.metric] + df['yaw_' + args.metric]) / 3
    # df['adjusted_crash'] = kinematic_realism
    # road_realism = df['out_of_road_' + args.metric]
    # safety_realism = (df['crash_' + args.metric] * 0.25 + df['dist_' + args.metric] * 0.1) / 0.35

    # TODO: find ideal metric here...
    # Justification of weighted: in Waymo Sim Agents, safety is penalized more than out_of_road in realism metric

    # Waymo Sim Agents ratios: 20% kinematic realism, 35% road adherence, 45% collision safety
    # TODO: get distance to nearest edge dist + TTC dist
    # df['meta_metric'] = (0.2 * kinematic_realism + 0.35 * road_realism + 0.45 * safety_realism) * 100


    # main_col = 'crash'
    # main_col = 'adjusted_crash'
    # main_reversed = True

    main_col = 'arrive'
    # main_col = 'meta_metric'
    main_reversed = False
    adv_meta_metrics = ['adv_yaw_' + args.metric, 'adv_acc_' + args.metric, 'adv_out_of_road_' + args.metric, 'adv_other_crash_' + args.metric]
    df['adv_meta_' + args.metric] = 1/len(adv_meta_metrics) * df[adv_meta_metrics].sum(axis=1)


    reordered_cols = [main_col, 'adjusted_crash'] if main_col != 'adjusted_crash' else [main_col]
    if main_col == 'adjusted_crash':
        reordered_cols = [main_col, 'adv_meta_' + args.metric]
    elif main_col == 'adv_meta_' + args.metric:
        reordered_cols = [main_col, 'adjusted_crash']
    else:
        reordered_cols = [main_col, 'adjusted_crash', 'adv_meta_' + args.metric]
    df = df[reordered_cols + [x for x in basic_metrics if x not in reordered_cols] + wd_suff_metrics]
    df = df.sort_values(['gen_type', main_col])
    #df = df.drop(columns=['gen_type'])

    # df = df.drop(columns=[x for x in wd_suff_metrics if 'yaw' not in x])
    # df = df.drop(columns=['ttc_into_ego', 'drac_into_ego', 'thw_into_ego'])
    # df = df.drop(columns=['ttc_into_other', 'drac_into_other', 'thw_into_other'])
    # df = df.drop(columns=['thw_into_ego', 'thw_into_other', 'drac_into_ego', 'drac_into_other', 'delta_mttcp'])
    # df = df.drop(columns=['ttc_into_ego', 'ttc_into_other', 'drac_into_ego', 'drac_into_other', 'delta_mttcp'])
    # df = df.drop(columns=['ttc_into_ego', 'ttc_into_other', 'thw_into_ego', 'thw_into_other', 'delta_mttcp'])
    # df = df.drop(columns=['ttc_into_ego', 'ttc_into_other', 'thw_into_ego', 'thw_into_other', 'drac_into_ego', 'drac_into_other'])
    if args.interaction:
        df = df.drop(columns=['ttc_into_ego', 'ttc_into_other', 'thw_into_ego', 'thw_into_other', 'drac_into_ego', 'drac_into_other', 'delta_mttcp'])
    else:

        df = df.drop(columns=['ttc_into_ego', 'ttc_into_other', 'thw_into_ego', 'thw_into_other', 'drac_into_ego', 'drac_into_other', 'delta_mttcp', 'interaction_score'])
    percent_cols = set(['arrive', 'adjusted_crash', 'crash', 'out_of_road', 'adv_other_crash', 'adv_out_of_road', 'route_completion', 'weighted'])

    # Only include attributes that have more than one value
    all_attributes = []
    for idx, row in df.iterrows():
        attributes = get_attributes(idx, args)
        all_attributes.append(attributes)
    all_attribute_keys = set()
    for x in all_attributes:
        all_attribute_keys = set.union(all_attribute_keys, set(x.keys()))

    attribute_vals = {}
    for x in all_attributes:
        for k in all_attribute_keys:
            if k not in x:
                x[k] = '-'
            attribute_vals.setdefault(k, [])
            attribute_vals[k].append(x[k])
    
    optional_attributes = {'hybrid_co', 'adv_mode', 'train_time', 'adv_objective'}

    filtered_attributes = [k for k in all_attribute_keys if \
                           (len(vals := np.unique(attribute_vals[k])) > 1 and k not in optional_attributes) or \
                           (len(vals) > 1 and '-' not in vals) or \
                           (len(vals) > 2) ]
    if 'ego_method' not in filtered_attributes:
        filtered_attributes.append('ego_method')
    filtered_attribute_vals = {k: attribute_vals[k] for k in filtered_attributes}
    for i, (k, v) in enumerate(filtered_attribute_vals.items()):
        df.insert(i, k, v)

    eval_setting_dfs = {}
    eval_raw_m = {}
    eval_raw_h = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen_types = gen_type_map.values()
        seen_gen_types = set()
        for gen_type in gen_types:
            if gen_type in seen_gen_types:
                continue
            seen_gen_types.add(gen_type)
            gen_df = df[df.gen_type == gen_type].drop(columns=['gen_type'])
            if not len(gen_df):
                continue
            print(gen_type)
            to_avg = {}
            for index in gen_df.index:
                if re.search('_full[\\d]+', index):
                    new_name = index.split('_full')[0] + '_full*_' + '_'.join(index.split('_full')[1].split('_')[1:])
                    to_avg.setdefault(new_name, [])
                    to_avg[new_name].append(index)
                elif re.search('_initial[\\d]+', index):
                    new_name = index.split('_initial')[0] + '_initial*_' + '_'.join(index.split('_initial')[1].split('_')[1:])
                    to_avg.setdefault(new_name, [])
                    to_avg[new_name].append(index)
                else:
                    to_avg[index] = [index]
            raw_df_m = gen_df.copy()
            raw_df_h = gen_df.copy()
            for new_name, to_avg_idx in to_avg.items():
                to_avg_df = gen_df.loc[to_avg_idx]
                gen_df = gen_df.drop(index=to_avg_idx)
                col_vals = []
                raw_vals_m = []
                raw_vals_h = []
                for col in to_avg_df.columns:
                    m, _, hi, mci_str = mean_confidence_interval(to_avg_df[col], use_percent=(col in percent_cols), agg_metric=agg_metric)
                    col_vals.append(mci_str)
                    if m is not None:
                        raw_vals_m.append(m)
                        raw_vals_h.append(hi - m)
                    else:
                        raw_vals_m.append(mci_str)
                        raw_vals_h.append(mci_str)
                gen_df.loc[new_name] = col_vals
                raw_df_m.loc[new_name] = raw_vals_m
                raw_df_h.loc[new_name] = raw_vals_h
            arrive_dict = {index: gen_df.loc[index, main_col] for index in gen_df.index}
            arrive_dict = {k: v for k, v in natsorted(arrive_dict.items(), key=lambda item: item[1], reverse=main_reversed)}
            gen_df = gen_df.loc[arrive_dict.keys()]
            raw_df_m = raw_df_m.loc[arrive_dict.keys()]
            raw_df_h = raw_df_h.loc[arrive_dict.keys()]
            gen_df['full_name'] = gen_df.index
            gen_df = gen_df.reset_index(drop=True)
            gen_df = gen_df.set_index('ego_method')
            raw_df_m['full_name'] = raw_df_m.index
            raw_df_m = raw_df_m.reset_index(drop=True)
            raw_df_m = raw_df_m.set_index('ego_method')
            raw_df_h['full_name'] = raw_df_h.index
            raw_df_h = raw_df_h.reset_index(drop=True)
            raw_df_h = raw_df_h.set_index('ego_method')
            print(gen_df)
            print()
            eval_setting_dfs[gen_type] = gen_df
            eval_raw_m[gen_type] = raw_df_m
            eval_raw_h[gen_type] = raw_df_h
    
    agg_methods = {}
    agg_evals = {}
    for k in eval_setting_dfs.keys():
        df_m, df_h = eval_raw_m[k], eval_raw_h[k]
        for ((idx_m, row_m), (idx_h, row_h)) in zip(df_m.iterrows(), df_h.iterrows()):
            full_name = row_m.full_name
            assert full_name == row_h.full_name, 'm and h mismatch'
            if '_normal_n1' in full_name:
                method = full_name.split('_normal_n1')[0]
                eval_setting = 'normal_n1' + full_name.split('_normal_n1')[1]
            elif '_adv_n1' in full_name:
                method = full_name.split('_adv_n1')[0]
                eval_setting = 'adv_n1' + full_name.split('_adv_n1')[1]
            elif '_adv_n5' in full_name:
                method = full_name.split('_adv_n5')[0]
                eval_setting = 'adv_n5' + full_name.split('_adv_n5')[1]
            agg_methods.setdefault(method, {})
            name_info = agg_methods[method]
            name_info['index'] = idx_m
            agg_evals.setdefault(eval_setting, {})
            eval_info = agg_evals[eval_setting]
            eval_info['index'] = idx_m
            for col in row_m.index:
                if type(row_m[col]) == str:
                    name_info[col] = row_m[col]
                    eval_info[col] = row_m[col]
                else:
                    if col not in name_info:
                        name_info[col] = [row_m[col]]
                    else:
                        name_info[col].append(row_m[col])
                    if col not in eval_info:
                        eval_info[col] = [row_m[col]]
                    else:
                        eval_info[col].append(row_m[col])
    for name, name_info in agg_methods.items():
        for k, v in name_info.items():
            if type(v) != str:
                name_info[k] = np.mean(v)
    for eval_setting, eval_setting_info in agg_evals.items():
        for k, v in eval_setting_info.items():
            if type(v) != str:
                eval_setting_info[k] = np.mean(v)



    metric_cols = ['arrive', 'crash', 'out_of_road']
    # metric_cols = ['arrive', 'crash', 'out_of_road', 'adv_meta_' + args.metric]

    # TODO: create avg_df to print and add to latex too?
    agg_method_df = pd.DataFrame(agg_methods).transpose()
    agg_method_df = agg_method_df.sort_values('arrive')
    print(agg_method_df[metric_cols])

    agg_eval_df = pd.DataFrame(agg_evals).transpose()
    agg_eval_df.index = [gen_type_map['_' + k + '_eval'] for k in agg_eval_df.index]

    # adv_kinematic = ['adv_acc_' + args.metric, 'adv_yaw_' + args.metric]
    # adv_road = ['adv_out_of_road_' + args.metric]
    # adv_safety = ['adv_other_crash_' + args.metric]
    # agg_eval_df['adv_kinematic_' + args.metric] = 1/len(adv_kinematic) * agg_eval_df[adv_kinematic].sum(axis=1)
    # agg_eval_df['adv_road_' + args.metric] = 1/len(adv_road) * agg_eval_df[adv_road].sum(axis=1)
    # agg_eval_df['adv_safety_' + args.metric] = 1/len(adv_safety) * agg_eval_df[adv_safety].sum(axis=1)

    # meta_metrics = ['adv_yaw_' + args.metric, 'adv_acc_' + args.metric, 'adv_out_of_road_' + args.metric, 'adv_other_crash_' + args.metric]
    # agg_eval_df['meta_' + args.metric] = 1/len(meta_metrics) * agg_eval_df[meta_metrics].sum(axis=1)
    # meta_metrics = ['adv_kinematic_' + args.metric, 'adv_road_' + args.metric, 'adv_safety_' + args.metric]
    # agg_eval_df['meta_' + args.metric] = 0.2 * agg_eval_df[meta_metrics[0]] + 0.35 * agg_eval_df[meta_metrics[1]] + 0.45 * agg_eval_df[meta_metrics[2]]
    # agg_eval_df['meta_' + args.metric] = 1/3 * agg_eval_df[meta_metrics[0]] + 1/3 * agg_eval_df[meta_metrics[1]] + 1/3 * agg_eval_df[meta_metrics[2]]
    # agg_eval_df['adv_in_road'] = 1 - agg_eval_df['adv_out_of_road']
    # agg_eval_df['adv_no_other_crash'] = 1 - agg_eval_df['adv_other_crash']
    # meta_metrics = ['adv_yaw_' + args.metric, 'adv_acc_' + args.metric, 'adv_in_road', 'adv_no_other_crash'] if args.metric == 'rl' else \
    #                ['adv_yaw_' + args.metric, 'adv_acc_' + args.metric, 'adv_out_of_road', 'adv_other_crash'] 
    # agg_eval_df['meta_' + args.metric] = 1/len(meta_metrics) * agg_eval_df[meta_metrics].sum(axis=1)

    OUTPUT_ORDER_VALS = [gen_type_map[k] for k in OUTPUT_ORDER]
    eval_setting_dfs = {k: eval_setting_dfs[k] for k in OUTPUT_ORDER_VALS if k in eval_setting_dfs}
    keys_ordered = list(eval_setting_dfs.keys())
    agg_eval_df['sort_val'] = 0
    for i, k in enumerate(keys_ordered):
        agg_eval_df.loc[k, 'sort_val'] = i
    agg_eval_df = agg_eval_df.sort_values('sort_val')
    agg_eval_df = agg_eval_df.drop(columns=['sort_val'])


    # agg_eval_df = agg_eval_df.sort_values('adv_meta_' + args.metric, ascending=(args.metric == 'rl'))
    print()
    agg_eval_cols = ['arrive', 'adv_meta_' + args.metric, *adv_meta_metrics]
    agg_eval_df['eval_setting'] = agg_eval_df.index
    agg_eval_df = agg_eval_df[['eval_setting', *agg_eval_cols]]
    print(agg_eval_df[agg_eval_cols])

    print('\ngen_quality\n')
    for col in agg_eval_df.columns[1:]:
        agg_eval_df[col] = [f'{pad_percent(float(x), n=5)}' if col in percent_cols else f'{pad_nonpercent(float(x), n=4)}' for x in agg_eval_df[col]]
    agg_latex = agg_eval_df.to_latex( multirow=True, multicolumn=True, header=True, index=False, escape=True)
    agg_latex = agg_latex.replace('ego\\_method', 'Ego Method')
    agg_latex = agg_latex.replace('eval\\_setting', 'Eval Scenario Type')
    agg_latex = agg_latex.replace('adv\\_meta\\_wd', 'Realism WD ($\\downarrow$)')
    agg_latex = agg_latex.replace('adv\\_yaw\\_wd', 'Yaw WD ($\\downarrow$)')
    agg_latex = agg_latex.replace('adv\\_acc\\_wd', 'Acc WD ($\\downarrow$)')
    agg_latex = agg_latex.replace('adv\\_out\\_of\\_road\\_wd', 'Road WD ($\\downarrow$)')
    agg_latex = agg_latex.replace('adv\\_other\\_crash\\_wd', 'Crash WD ($\\downarrow$)')
    agg_latex = agg_latex.replace('crash', 'Ego Crash ($\\uparrow$)')
    agg_latex = agg_latex.replace('out\\_of\\_road', 'Ego Out of Road ($\\uparrow$)')
    agg_latex = agg_latex.replace('arrive', 'Ego Success ($\\downarrow$)')
    lines = agg_latex.split('\n')
    caption_line = '\\caption{' + QUALITY_CAPTION + '}'
    label_line = '\\label{' + f'tab:gen_quality' + '}'
    lines = ['\\begin{table*}[hbtp]',
                '\\centering',
                caption_line,
                label_line, 
                '\\resizebox{1.0\\textwidth}{!}{' + lines[0], 
                *lines[1:-1],
                lines[-1] + '}', 
                '\\end{table*}']
    agg_latex = '\n'.join(lines)
    print(agg_latex)
    print('\n\n\n\n')

    # print(agg_eval_df[['meta_' + args.metric, *metric_cols]])

    # real_keys = [x for x in eval_setting_dfs.keys() if 'Adv' not in x]
    # adv_gen_keys = [x for x in eval_setting_dfs.keys() if 'Adv' in x]
    # all_keys = [real_keys, adv_gen_keys]
    real_keys = [x for x in eval_setting_dfs.keys()]
    all_keys = [real_keys]

    # names = ['real', 'adv_gen']
    # captions = ['Results on real scenes.', 'Results on adversarially-perturbed interactive scenes.']
    print()
    names = ['main'] if 'ablation' not in args.in_file else ['ablation']
    captions = [CAPTION] if 'ablation' not in args.in_file else [ABLATION_CAPTION]

    for keys, name, caption in zip(all_keys, names, captions):
        to_concat = []
        for x in keys:
            df = eval_setting_dfs[x]
            df['eval_setting'] = x
            df = df[['eval_setting', *metric_cols]]
            replay = df[df.index == REPLAY]
            goose_adv = df[df.index == GOOSE]
            goose_adv_reskill = df[df.index == GOOSE_RESKILL]
            cat_no_adv = df[df.index == CAT_NO_ADV]
            cat_heuristic = df[df.index == CAT_HEURISTIC]
            cat_open = df[df.index == CAT_OPEN]
            cat_adv = df[df.index == CAT_CLOSED]
            cat_adv_reskill = df[df.index == CAT_CLOSED_RESKILL]
            ours = df[df.index == OURS]
            others = df[~df.index.isin(np.concatenate([np.array(replay.index),
                                                       np.array(goose_adv.index),
                                                       np.array(goose_adv_reskill.index),
                                                       np.array(cat_no_adv.index),
                                                       np.array(cat_heuristic.index),
                                                       np.array(cat_open.index),
                                                       np.array(cat_adv.index),
                                                       np.array(cat_adv_reskill.index),
                                                       np.array(ours.index)]))]
            others = others.sort_index()
            df = pd.concat([replay,
                            others,
                            goose_adv, 
                            goose_adv_reskill, 
                            cat_no_adv,
                            cat_heuristic,
                            cat_open,
                            cat_adv, 
                            cat_adv_reskill, 
                            ours])
            ablation_dfs = []
            if 'ablation' in args.in_file:
                for ablation_k, ablation_v in ABLATIONS.items():
                    ablation_df = df[df.index == ablation_v]
                    if not len(ablation_df):
                        continue
                    ablation_dfs.append(ablation_df)
                assert len(ablation_dfs), 'At least one ablation match needed for ablation mode'
                df = pd.concat(ablation_dfs)
            to_concat.append(df)
        df = pd.concat(to_concat)
        df.set_index('eval_setting', append=True, inplace=True)
        df = df.reorder_levels(['eval_setting', df.index.names[0]])
        latex_code = df.to_latex( multirow=True, multicolumn=True, header=True, index=True, escape=True)
        # No idea why some of these have an actual _ in them...
        latex_code = latex_code.replace('ego_method', 'Training Setting')
        latex_code = latex_code.replace('eval_setting', 'Eval Scenario Type')
        latex_code = latex_code.replace('crash', 'Crash ($\\downarrow$)')
        latex_code = latex_code.replace('out\\_of\\_road', 'Out of Road ($\\downarrow$)')
        latex_code = latex_code.replace('arrive', 'Success ($\\uparrow$)')
        latex_code = latex_code.replace('adv\\_meta\\_wd', 'Adv Realism WD ($\\downarrow$)')
        lines = latex_code.split('\n')
        tmp1 = lines[2].split('&')[2:]
        tmp2 = lines[3].split('&')[:2]
        tmp = '&'.join([*tmp2, *tmp1])
        caption_line = '\\caption{' + caption + '}'
        label_line = '\\label{' + f'tab:results_{name}' + '}'
        lines = ['\\begin{table*}[hbtp]',
                 '\\centering',
                 caption_line,
                 label_line, 
                 '\\resizebox{0.8\\textwidth}{!}{' + lines[0], 
                 lines[1], tmp, *lines[4:-1], 
                 lines[-1] + '}', 
                 '\\end{table*}']
        for line_idx, line in enumerate(lines):
            if OURS in line:
                line = line.replace(OURS, '\\textbf{' + OURS + '}')
            # Add an extra cline after each GT row
            if 'multirow' in line and 'ablation' not in args.in_file:
                line = line + '\n' + '\\cline{2-5}'
            if 'ablation' in args.in_file and ABLATIONS['model_cat_reskill_skill_idm_all_adv_prior_initial'] in line:
                line = line + '\n' + '\\cline{2-5}'
            if 'ablation' in args.in_file and ABLATIONS['model_cat_reskill_idm_learned_obj_initial'] in line:
                line = line + '\n' + '\\cline{2-5}'
            lines[line_idx] = line
        # Remove last cline 1-5 
        lines.pop(-5)
        latex_code = '\n'.join(lines)
        # TODO: make prettier, etc.
        print(name)
        print()
        print(latex_code)
        print('\n\n\n\n')

    for indiv_metric, name, y_label in zip(
        # ['arrive', 'adv_meta_' + args.metric],
        # ['success', 'realism'],
        # ['Ego Success', 'Adv Realism WD']):
        ['arrive'],
        ['success'],
        ['            Ego Success']):
        arrive_info = {k: v[indiv_metric] for k, v in eval_setting_dfs.items()}
        if indiv_metric in percent_cols:
            arrive_info = { k: v.apply(lambda x: float(x.split('%')[0])) for k, v in arrive_info.items() }
        else:
            arrive_info = { k: v.apply(lambda x: float(x.split(' ')[0])) for k, v in arrive_info.items() }
        arrive_df = pd.DataFrame(arrive_info).T
        arrive_df = arrive_df[[x for x in ABLATIONS.values() if x in arrive_df.columns]]
        ax = arrive_df.plot(kind='bar', figsize=(12, 6))
        ax.set_ylabel(y_label)
        plt.xticks(rotation=0, ha='center')
        #plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend(title='Training Settings')

        out_name = args.in_file.split('in_')[-1].split('.txt')[0]
        plt.savefig(f'metrics/{out_name}_{name}.png', dpi=400, bbox_inches='tight')

        plt.clf()
        ax = sns.heatmap(arrive_df.T, annot=True, cmap="viridis", cbar_kws={'label': y_label})
        # ax = sns.heatmap(arrive_df.T, annot=True, cmap="viridis")

        # Labeling the axes
        ax.set_ylabel('Training Settings')
        # ax.set_xlabel('Evaluation Settings')

        # Save the figure
        plt.savefig(f'metrics/{out_name}_{name}_heatmap.png', dpi=400, bbox_inches='tight')