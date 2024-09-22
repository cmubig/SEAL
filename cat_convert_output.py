import argparse
import numpy as np
import os
import glob
from tqdm import tqdm

from natsort import natsorted

from reskill.utils.general_utils import AttrDict

# Adapted version of reskill/**/collect_demos.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subseq_len', type=int, default=10)
    # TODO: union of (ego_idm_normal and ego_idm_reactive_obs_all_normal)
    
    # Use like python cat_convert_output.py --in_dir dir1 dir2 
    parser.add_argument('--in_dir', type=str, action='append', nargs='*', \
                        default=[])
    parser.add_argument('--out_dir', type=str, default='reskill/dataset')
    parser.add_argument('--dataset_name', type=str, default='idm_all')
    args = parser.parse_args()

    in_dirs = args.in_dir
    if not len(in_dirs):
        in_dirs = [['output/ego_idm_normal_n1_train-seed0-0', 'output/ego_idm_reactive_obs_all_normal_n1_train-seed0-0']]
    in_dirs = [y for x in in_dirs for y in x]


    #output_path = f'{args.out_dir}/{args.in_dir.split("/")[-1]}'
    output_path = f'{args.out_dir}/{args.dataset_name}'
    os.makedirs(output_path, exist_ok=True)

    # A little preprocessing here:
    # 1. Clip to -1, 1 range
    # 2. Perform filtering on out of road/crash trajectories

    seqs = []
    for in_dir in in_dirs:
        print('Processing', in_dir)
        full_path = f'{in_dir}/obs'
        all_obs = natsorted(os.listdir(full_path))
        for obs in tqdm(all_obs):
            data = np.load(f'{full_path}/{obs}', allow_pickle=True)
            data = data.item()
            observations = data['obs']
            actions = data['actions']
            assert len(observations) == len(actions), 'Mismatch between observations and actions length'

            # TODO: Further filter out actions immediately preceding a fail? Beyond just the immediate step..
            # Save reason to allow for that later
            if len(actions) >= args.subseq_len:
                ego_track = [x['default_agent'] for x in data['tracks']]
                ego_crash = np.array([x['crash'] for x in data['extra_ego_info']])
                ego_out_of_road = np.array([x['out_of_road'] for x in data['extra_ego_info']])
                if ego_crash.any():
                    assert np.sum(ego_crash) == 1, 'Too many terminations'
                    assert ego_crash[-1] == True, 'Unexpected location of crash'
                if ego_out_of_road.any():
                    assert np.sum(ego_out_of_road) == 1, 'Too many terminations'
                    assert ego_out_of_road[-1] == True, 'Unexpected location of out_of_road'
                if ego_crash.any() or ego_out_of_road.any():
                    observations = observations[:-1]
                    actions = actions[:-1]
                    ego_track = ego_track[:-1]
                # One of: crash, out_of_road, arrive
                ego_reason = 'crash' if ego_crash.any() else 'out_of_road' if ego_out_of_road.any() else 'arrive'
                ego_dynamics = data['extra_ego_info'][0]['dynamics']
                assert (np.array([x['dynamics'] for x in data['extra_ego_info']]) == ego_dynamics).all(), 'Dynamics changed'
                if len(actions) >= args.subseq_len:
                    seqs.append(AttrDict(obs=observations, actions=actions, track=ego_track, done=ego_reason, dynamics=ego_dynamics))

            # For other obs, need to split sequences at discontinuities
            if 'other_obs' in data and 'other_actions' in data:
                all_keys = set()
                for x, y in zip(data['other_obs'], data['other_actions']):
                    # Sometimes a mismatch, if an agent ends early
                    all_keys.update(x.keys())
                    all_keys.update(y.keys())
                for name in all_keys:
                    cur_obs = []
                    cur_actions = []
                    cur_track = []
                    cur_dynamics = None
                    for other_obs, other_action, other_info, tracks in \
                            zip(data['other_obs'], data['other_actions'], data['extra_other_info'], data['tracks']):
                        valid = name in other_obs and name in other_action and name in other_info and name in tracks
                        reason = 'arrive'
                        if not valid:
                            reason = 'missing_obs'
                        else:
                            info = other_info[name]
                            if info['out_of_road'] or info['crash']:
                                reason = 'crash' if info['crash'] else 'out_of_road'
                        # A discontinuity to deal with
                        if reason != 'arrive':
                            if len(cur_obs) < args.subseq_len:
                                cur_obs = []
                                cur_actions = []
                                cur_track = []
                                cur_dynamics = None
                                continue
                            seqs.append(AttrDict(obs=cur_obs, actions=cur_actions, track=cur_track, done=reason, dynamics=cur_dynamics))
                            cur_obs = []
                            cur_actions = []
                            cur_track = []
                            cur_dynamics = None
                            continue

                        cur_obs.append(other_obs[name])
                        cur_actions.append(other_action[name])
                        cur_track.append(tracks[name])
                        if cur_dynamics is not None:
                            assert (cur_dynamics == info['dynamics']).all(), 'Dynamics mismatch'
                        cur_dynamics = info['dynamics']
                    if reason == 'arrive' and len(cur_obs) >= args.subseq_len:
                        seqs.append(AttrDict(obs=cur_obs, actions=cur_actions, track=cur_track, done=reason, dynamics=cur_dynamics))

    
    reasons = [x.done for x in seqs]
    print(np.unique(reasons, return_counts=True))

    np_seq = np.array(seqs)
    output_file = f'{output_path}/demos.npy'
    np.save(output_file, np_seq)


