import pickle
import argparse
import shutil
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_summary_path", required=True, help="dataset_summary file with extra info saved")
    parser.add_argument("--output", default="raw_scenes_hard", help="Where to copy the filtered scenarios")
    parser.add_argument('--filter_path', type=str, default="safeshift/sorted_meta.pkl", help="Additional filter to apply")
    parser.add_argument('--n', type=int, default=100, help="How many scenes to save")
    args = parser.parse_args()

    filter_path = args.filter_path
    with open(filter_path, 'rb') as f:
        filter_info = pickle.load(f)
    

    # For original safeshift-hard set, just take the strict top-N highest scoring scenarios
    # threshold = sorted([x['sdc_score'] for x in filter_info], reverse=True)[min(len(filter_info) - 1, args.n - 1)]
    
    # Top 20% reduces 1465 scenes to 293 scenes
    # Instead of just taking the strict 100 highest sdc score, let's examine the top 20% scoring scenes, get more diversity...
    # Possibilities:
    # 1. Just take a random subset of 100 scenes of them
    # 2. Take the 100 where the ego and adv get closest?

    threshold = np.percentile([x['sdc_score'] for x in filter_info], 80) 
    filtered = [x for x in filter_info if x['sdc_score'] >= threshold]
    filter_ids = set([x['scenario_id'] for x in filtered])

    with open(args.input_summary_path, "rb") as f:
        summary_dict = pickle.load(f)
    
    new_summary = {}
    min_dists = []
    to_copy_paths = []
    adv_vals = []
    sdc_vals = []
    for obj_id, summary in tqdm(summary_dict.items(), 'Processing filtered scenes'):
        if summary['scenario_id'] not in filter_ids:
            continue

        if summary['track_length'] != 91:
            raise ValueError('Unexpected track length not 91')

        sdc_id = summary['sdc_id']
        objects_of_interest = summary['objects_of_interest']

        if sdc_id not in objects_of_interest or len(objects_of_interest) != 2:
            raise ValueError('Unexpected objects_of_interest vals')
        
        to_copy = args.input_summary_path.replace('dataset_summary.pkl', obj_id)
        adv_id = [x for x in objects_of_interest if x != sdc_id][0]

        full_scene = np.load(to_copy, allow_pickle=True)
        def get_track(x_id):
            """Return track of [x, y, val] over time."""
            x = full_scene['tracks'][x_id]['state']['position']
            x_val = full_scene['tracks'][x_id]['state']['valid']
            x[:, -1] = x_val
            return x
        
        adv_track = get_track(adv_id)
        sdc_track = get_track(sdc_id)
        shared_val = adv_track[:, -1].astype(bool) & sdc_track[:, -1].astype(bool)
        adv_val = adv_track[shared_val][:, :2]
        sdc_val = sdc_track[shared_val][:, :2]
        min_dist = np.linalg.norm(adv_val - sdc_val, axis=-1).min()
        min_dists.append(min_dist)
        sdc_vals.append(sdc_val)
        adv_vals.append(adv_val)
        to_copy_paths.append(to_copy)

    count = 0

    scene_idxs = np.arange(len(min_dists))
    select_state = np.random.RandomState(42)
    scene_idxs = select_state.choice(scene_idxs, size=args.n, replace=False)
    # dist_thresh = sorted(min_dists)[min(len(min_dists) - 1, args.n - 1)]

    for scene_idx, (min_dist, to_copy_path, adv_val, sdc_val) in tqdm(enumerate(zip(min_dists, to_copy_paths, adv_vals, sdc_vals)),
                                                         'Rendering and copying scenes', total=len(min_dists)):
        if scene_idx not in scene_idxs:
            continue
        # if min_dist > dist_thresh:
        #     continue
        origin = sdc_val[0]
        adv_val -= origin
        sdc_val -= origin
        plt.clf()
        plt.plot(adv_val[:, 0], adv_val[:, 1], marker = '.', color='b')
        plt.plot(sdc_val[:, 0], sdc_val[:, 1], marker = '.', color='r')
        for adv_xy, sdc_xy in zip(adv_val, sdc_val):
            plt.plot([adv_xy[0], sdc_xy[0]], [adv_xy[1], sdc_xy[1]], color='k', alpha=0.2)
        plt.savefig(f'safeshift/hard_viz_rand/{count}.png')
        shutil.copyfile(to_copy_path, f'{args.output}/{count}.pkl')
        count += 1