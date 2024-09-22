import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="cat_dynamics/config_all.yaml")
    parser.add_argument('--dataset_name', type=str, default="idm_all")
    args = parser.parse_args()

    # TODO: add in code to make a dynamics dataloader, that uses track as well
    # Unsure what to map, but we want two models for sure:
    # 1. (dynamics + current_state + clipped_action) -> next_state
    # 2. (dynamics + delta current_state & previous_state) -> clipped_action (using tanh)

    # The action output is more important, in order to learn skills directly from replay (along with IDM) data

    breakpoint()
