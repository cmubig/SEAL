import argparse
import numpy as np
import pickle as pkl

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/av_shared_ssd/mtr_process_ssd', help='Where to search for input scenarios')
    parser.add_argument('--input_test', type=str, default='score_asym_combined_80_extra_processed_scenarios_test_infos.pkl', help='Which scores to use')
    parser.add_argument('--output_path', type=str, default='sorted_meta.pkl', help='Output path to save filtered and sorted scenarios')
    args = parser.parse_args()


    input_dir = Path(args.input_dir)
    meta_files = input_dir / args.input_test
    test_infos = np.load(meta_files, allow_pickle=True)
    val_infos = np.load(Path(str(meta_files).replace('test', 'training')), allow_pickle=True)
    train_infos = np.load(Path(str(meta_files).replace('test', 'val')), allow_pickle=True)

    all_infos = train_infos + val_infos + test_infos
    # 1492 scenes
    all_infos = [x for x in all_infos if len(x['objects_of_interest']) == 2 and \
                              x['objects_of_interest'][0] in x['tracks_to_predict']['track_index'] and \
                                x['objects_of_interest'][1] in x['tracks_to_predict']['track_index']]
    filtered_infos = []
    for x in tqdm(all_infos, 'Finding relevant cases', total=len(all_infos)):
        scenario_id = x['scenario_id']
        input_path = input_dir / 'joint_original' / f'sample_{scenario_id}.pkl'
        data = np.load(input_path, allow_pickle=True)
        scene_score = x['score']
        traj_scores = x['traj_scores_asym_combined']
        obj_1 = x['objects_of_interest'][0]
        obj_2 = x['objects_of_interest'][1]
        score_1 = traj_scores[x['tracks_to_predict']['track_index'].index(obj_1)]
        score_2 = traj_scores[x['tracks_to_predict']['track_index'].index(obj_2)]

        # A few filters: one of the objects needs to be visible for the entire time
        index_1 = data['track_infos']['object_id'].index(obj_1)
        track_1 = data['track_infos']['trajs'][index_1]
        all_valid_1 = track_1[:, -1].sum() == 91

        index_2 = data['track_infos']['object_id'].index(obj_2)
        track_2 = data['track_infos']['trajs'][index_2]
        all_valid_2 = track_2[:, -1].sum() == 91

        if not (all_valid_1 or all_valid_2):
            continue

        # Set the SDC to be the one which is all valid, tie break by higher score
        if all_valid_1 and all_valid_2:
            sdc_id = obj_1 if score_1 > score_2 else obj_2
            sdc_index = index_1 if score_1 > score_2 else index_2
            sdc_score = score_1 if score_1 > score_2 else score_2
        elif all_valid_1:
            sdc_id, sdc_index, sdc_score = obj_1, index_1, score_1
            if track_2[:, -1].sum() < 50:
                continue
        else:
            sdc_id, sdc_index, sdc_score = obj_2, index_2, score_2
            if track_1[:, -1].sum() < 50:
                continue

        x['sdc_track_index'] = sdc_index
        x['sdc_score_index'] = x['tracks_to_predict']['track_index'].index(sdc_id) 
        x['sdc_score'] = sdc_score
        filtered_infos.append(x)

    # 1465, down from 1492: len(filtered_infos)
    infos = sorted(filtered_infos, key = lambda x: x['sdc_score'])
    scores = [x['sdc_score'] for x in infos]
    plt.hist(scores)
    plt.axvline(x=infos[-100]['sdc_score'], color='r')
    plt.savefig('sdc_scores.png')

    print(f'Saving {len(scores)} metas; median overall: {np.median(scores)}, median top-100: {np.median(scores[-100:])}')
    with open(args.output_path, 'wb') as f:
        pkl.dump(filtered_infos, f)