import numpy as np
from natsort import natsorted
import os


if __name__ == '__main__':
    input_path1 = 'raw_scenes_500'
    input_path2 = 'raw_scenes_hard_rand'

    scenes1 = natsorted([x for x in os.listdir(input_path1) if x.endswith('pkl')])
    ids1 = [np.load(input_path1 + '/' + x, allow_pickle=True)['id'] for x in scenes1]

    scenes2 = natsorted([x for x in os.listdir(input_path2) if x.endswith('pkl')]) 
    ids2 = [np.load(input_path2 + '/' + x, allow_pickle=True)['id'] for x in scenes2]

    print(f'Intersecting set between {input_path1} and {input_path2}')
    print(len(set(ids1).intersection(set(ids2))), 'overlapping ids')