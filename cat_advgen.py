import argparse
import numpy as np
from tqdm import trange
import time
import pygame
import os
import hashlib
import glob
import tempfile
import torch
import ffmpeg
from copy import deepcopy

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, ReplayTrafficParticipantPolicy
from metadrive.policy.idm_policy import WaymoIDMPolicy, TrajectoryIDMPOlicy
from metadrive.utils import clip
# Can swap in adv_generator_rule to use rule-based AdvGenerator too, for eval setting B in CAT paper
from advgen.adv_generator import AdvGenerator
from advgen.adv_generator_rule import AdvGenerator as AdvGeneratorRule
from advgen.adv_generator_hybrid import AdvGenerator as AdvGeneratorHybrid
from advgen.adv_generator_goose import AdvGenerator as AdvGeneratorGoose
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from saferl_algo import TD3, TD3_GRU, skill_model, reskill_model, utils
from saferl_plotter.logger import SafeLogger
import goose_train

def write_frame(env, dir_path, ep_timestep, debug=False):
    if debug:
        return
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    pygame.image.save(env.engine._top_down_renderer.canvas, f'{dir_path}/frame_{str(ep_timestep).zfill(3)}.png')

def write_video(dir_path_in, output, debug=False):
    if debug:
        return
    img_pattern = f'{dir_path_in}/*.png'
    ffmpeg.input(img_pattern, pattern_type='glob', framerate=10)\
            .output(output, loglevel='quiet')\
            .run(overwrite_output=True)
    for img_frame in glob.glob(img_pattern):
        os.remove(img_frame)

# TODO: save out_of_road status, for filtering bad trajectories!
def get_tracks(env, save_model_name):
    ret = {}
    names = env.engine.get_objects().keys()
    extra_ego_info = {}
    extra_other_info = {}
    for name in names:
        obj = env.engine.get_object(name).get(name)
        # chassis = obj.chassis.node()
        length, width = obj.LENGTH, obj.WIDTH
        if name == 'default_agent':
            params = obj.get_dynamics_parameters()
            max_engine_force = params['max_engine_force']
            max_brake_force = params['max_brake_force']
            wheel_friction = params['wheel_friction']
            max_steering = params['max_steering']
            mass = params['mass']
            dynamics = np.array([max_engine_force, max_brake_force, wheel_friction, max_steering, mass], dtype=np.float32)
            extra_ego_info['dynamics'] = dynamics

            # These don't depend on seed, always the same sequence...
            extra_ego_info['navi_info'] = obj.navigation.get_navi_info().astype(np.float32)
            extra_ego_info['policy'] = str(env.engine.get_policy(name))
            crash = env.vehicle.crash_vehicle if save_model_name != 'ego_replay' else env.vehicle.ego_crash_flag
            out_of_road = env._is_out_of_road(obj)
            extra_ego_info['out_of_road'] = out_of_road
            extra_ego_info['crash'] = crash
        else:
            crash = hasattr(obj, 'crash_vehicle') and obj.crash_vehicle
        
        if name != 'default_agent' and isinstance(obj, BaseVehicle):
            params = obj.get_dynamics_parameters() 
            max_engine_force = params['max_engine_force']
            max_brake_force = params['max_brake_force']
            wheel_friction = params['wheel_friction']
            max_steering = params['max_steering']
            mass = params['mass']
            dynamics = np.array([max_engine_force, max_brake_force, wheel_friction, max_steering, mass], dtype=np.float32)
            out_of_road = env._is_out_of_road(obj)
            
            # These don't depend on seed, always the same sequence...
            navi_info = obj.navigation.get_navi_info().astype(np.float32)
            policy_name = str(env.engine.get_policy(name))
            extra_other_info[name] = {'dynamics': dynamics, 'navi_info': navi_info, 'policy': policy_name,
                                      'out_of_road': out_of_road, 'crash': crash}

        # As in, should we actually just save the actions taken by IDM? let's try IK first tho...
        # Saving from IDM makes a lot of sense actually...or we can try to learn a policy? Still, 
        # let's see if we can "back out" action <-> traj follow
        info = np.concatenate([obj.position, obj.velocity, [obj.heading_theta, length, width, crash]]).astype(np.float32)
        ret[name] = info
    return ret, extra_ego_info, extra_other_info

def observe_other(env, other_id):
    if other_id not in env.engine.get_objects():
        return [-1] * 101
    obj_other = env.engine.get_object(other_id).get(other_id)
    scenario_obs = env.observations['default_agent']
    if observe_other.lateral_dist_idx == -1:
        observe_other.lateral_dist_idx = len(scenario_obs.state_observe(obj_other)) - 1

    other_obs = scenario_obs.observe(obj_other)
    lateral_dist = env.engine.map_manager.other_routes[other_id].local_coordinates(obj_other.position)[-1]
    lateral_obs = lateral_dist / scenario_obs.MAX_LATERAL_DIST
    lateral_obs = clip((lateral_obs + 1) / 2, 0.0, 1.0)
    other_obs[observe_other.lateral_dist_idx] = lateral_obs

    return other_obs
observe_other.lateral_dist_idx = -1

def observe_others(env):
    other_obs = {}
    names = env.engine.get_objects().keys()
    for name in names:
        if name == 'default_agent':
            continue
        if not isinstance(env.engine.get_object(name).get(name), BaseVehicle):
            continue
        other_obs[name] = observe_other(env, name)
    return other_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int,default=32)
    parser.add_argument('--AV_traj_num', type=int,default=1)
    parser.add_argument('--open_loop', action='store_true', help='Change adversarial generation to open loop (i.e., condition on ego replay)')

    parser.add_argument('--no_adv', action='store_true', help='Just do replay scenes, no adv generate, for consistency.')
    parser.add_argument('--use_hard', action='store_true', help='Use hard scenes instead of base 500')
    parser.add_argument('--hard_path', default='raw_scenes_hard', choices=['raw_scenes_hard', 'raw_scenes_hard_dist', 'raw_scenes_hard_rand'], help='Which hard set to use')

    parser.add_argument('--rule_based', action='store_true', help='Change adversarial generation to rule-based instead of learned')
    parser.add_argument('--goose_adv', action='store_true', help='Use GOOSE-adv')
    parser.add_argument('--goose_adv_path', default='goose_small_act0', help='GOOSE-adv frozen model path')
    parser.add_argument('--skill_based_adv_path', default='cat_reskill_initial0', help='Path for skill-based adversarial model')
    parser.add_argument('--model_adv_path', default='cat_initial0', help='Path for other model to use for adversary')
    parser.add_argument('--collision_offset', default='10', type=str, help='Amount behind calculated trajectory to takeover. -1 = inf, var = random, 10 = 10 steps, etc.')
    parser.add_argument('--skill_based_adv', action='store_true', help='Change adversarial generation to skill-based instead of prior-only')
    parser.add_argument('--model_adv', action='store_true', help='Change adversarial generation to another model instead of prior-only')
    parser.add_argument('--idm_adv', action='store_true', help='Change adversarial generation to IDM instead of prior-only')
    parser.add_argument('--current_model_adv', action='store_true', help='Actually use the current model being trained too')
    parser.add_argument('--current_model_prior', choices=['normal', 'adv'], default='normal', help='For current model skill, which prior to set')
    parser.add_argument('--learned_objective', default='', type=str, help='Use decision32 learned model instead of objective.')
    parser.add_argument('--learned_objective_mode', default='both', choices=['sc', 'diff', 'both'], type=str, help='Which learned_objective to use')

    parser.add_argument('--split', default='eval', choices=['eval', 'train'], help='Split between eval or train')

    parser.add_argument('--reactive', action='store_true', help='Background agents reactivity')
    parser.add_argument('--obs_all', action='store_true', help='Observe all agents if possible')
    parser.add_argument('--save_scenario', action='store_true', help='Visualize scenario BEV output')
    parser.add_argument('--scenario_id', default=-1, type=int, help='Specify a scenario to visualize')
    parser.add_argument('--repeat', action='store_true', help='Infinitely repeat the scenario')

    parser.add_argument('--ego_idm', action='store_true', help='Whether or not to use IDM for Ego')
    parser.add_argument('--save32', action='store_true', help='Whether or not to save all 32 possibile adv paths')

    parser.add_argument('--load_model', type=str, default='', help='Path to load model from for Ego')
    parser.add_argument('--no_prior', action='store_true', help='Skip skill prior, use random sampling from skill space')

    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates

    tmp_args = parser.parse_known_args()
    if tmp_args[0].goose_adv:
        assert tmp_args[0].AV_traj_num == 5, 'AV_traj_num must be 5 for GOOSE (# of GOOSE policy steps)'
        adv_generator = AdvGeneratorGoose(parser)
    elif tmp_args[0].skill_based_adv or tmp_args[0].idm_adv or tmp_args[0].model_adv or tmp_args[0].current_model_adv:
        assert np.sum([tmp_args[0].skill_based_adv, tmp_args[0].idm_adv, tmp_args[0].model_adv, tmp_args[0].current_model_adv]) == 1, 'Conflicting adv generation policy'
        adv_generator = AdvGeneratorHybrid(parser)
    elif tmp_args[0].rule_based:
        adv_generator = AdvGeneratorRule(parser)
    elif tmp_args[0].no_adv:
        adv_generator = AdvGeneratorRule(parser)
    else:
        adv_generator = AdvGenerator(parser)
    args = parser.parse_args()

    if args.save32:
        assert args.ego_idm and isinstance(adv_generator, AdvGenerator) and args.AV_traj_num == 1, \
            'Save32 requires ego idm and regular CAT advgen, with adv n1'

    # TODO: check that sum of these is 1, rather than different branches
    assert not (args.ego_idm and args.load_model != ''), 'Conflicting model'
    # TODO: allow for both rule_based and skill_based_adv together
    assert not (args.open_loop and (args.rule_based or args.skill_based_adv or args.no_adv)), 'Conflicting adv generation policy'
    assert not (args.no_adv and (args.rule_based or args.skill_based_adv or args.open_loop)), 'Conflicting adv generation policy'
    assert args.AV_traj_num >= 1, 'AV traj num must be at least 1'

    # Had to modify top_down_renderer for this...
    extra_args = dict(mode="top_down", film_size=(2200, 2200), flip=True)

    reactive = args.reactive
    repeat = args.repeat
    assert not repeat, 'Repeat not supported'

    save_scenario = args.save_scenario

    use_hard = args.use_hard
    hard_path = args.hard_path
    
    skill_based = 'saved_skill_models' in args.load_model or 'reskill' in args.load_model
    reskill = 'reskill' in args.load_model and 'saved_skill_models' not in args.load_model

    if reskill or not skill_based: 
        save_model_name = f'model_{args.load_model}' if args.load_model != '' \
                                                    else 'ego_idm' if args.ego_idm \
                                                    else 'ego_replay'
    else:
        expected_prefix = 'reskill/reskill/results/saved_skill_models/'
        well_formatted = args.load_model.startswith(expected_prefix)
        well_formatted = well_formatted and ('/' not in args.load_model.split(expected_prefix)[-1])
        assert well_formatted, 'Unexpected skill_based model path'

        # TODO: get a better name from this?
        #sus_hash = hashlib.shake_256(args.load_model.encode()).hexdigest(5)
        sus_hash = args.load_model.split(expected_prefix)[-1]
        prior_name = '_no_prior' if args.no_prior else ''
        save_model_name = f'model_skill{prior_name}_{sus_hash}'
    open_loop_name = '_open_loop' if args.open_loop else ''
    rule_based_name = '_rule_based' if args.rule_based else ''
    goose_name = '_goose' if args.goose_adv else ''
    reactive_name = '_reactive' if args.reactive else ''
    obs_all_name = '_obs_all' if args.obs_all else ''
    no_adv_name = '_normal' if args.no_adv else '_adv'
    if not use_hard:
        hard_name = ''
    else:
        hard_name = '_hard' if hard_path == 'raw_scenes_hard' else '_hard_dist' if 'dist' in hard_path else '_hard_rand'
    hybrid_name = adv_generator.hybrid_name if isinstance(adv_generator, AdvGeneratorHybrid) else ''
    av_traj_num = f'_n{args.AV_traj_num}'
    save_all_name = '_save32' if args.save32 else ''
    if not len(args.learned_objective):
        learned_name = ''
    elif args.learned_objective_mode == 'both':
        learned_name = '_learned_obj'
    else:
        learned_name = f'_learned_obj_{args.learned_objective_mode}'
    exp_name = f'{save_model_name}{open_loop_name}{reactive_name}{obs_all_name}{no_adv_name}{av_traj_num}{hard_name}{rule_based_name}{goose_name}{hybrid_name}{save_all_name}{learned_name}'
    
    env_name = args.split
    logger = SafeLogger(log_dir='./output', exp_name=exp_name, env_name=env_name, seed=args.seed, 
                        fieldnames=['crash_rate', 'out_of_road_rate', 'arrive_rate', 'route_completion'],
                        allow_overwrite=True, debug=args.debug)
    base_save_dir = logger.log_dir if not args.debug else ''
    # TODO: save info on rolled out trajectories for all in sim, termination reason, etc. 
    if not args.debug:
        os.makedirs(base_save_dir + '/videos', exist_ok=True)
        os.makedirs(base_save_dir + '/obs', exist_ok=True)


    config = {
            "use_render": False,
            "data_directory": './raw_scenes_500' if not use_hard else f'./{hard_path}',
            "force_reuse_object_name" :True,
            "sequential_seed": True,
            "vehicle_config": dict(show_navi_mark=False,
                                   show_dest_mark=False,
                                   lidar = dict(num_lasers=30,distance=50, num_others=3),
                                   side_detector = dict(num_lasers=30),
                                   lane_line_detector = dict(num_lasers=12)),
            "no_light": True,
            "no_static_vehicles": True,
            "sequential_seed": True,
            "num_scenarios": 100 if env_name == 'eval' else 400,
            "crash_vehicle_done": True,
            "start_scenario_index": 400 if (env_name == 'eval' and not use_hard) else 0
    }

    # Config to control background traffic behavior
    config["reactive_traffic"] = reactive
    config["spawn_once"] = True
    config["ignore_idm_region"] = True
    config["remove_after_destination"] = False
    config["traffic_need_navigation"] = True

    if args.load_model == '':
        config["agent_policy"] = WaymoIDMPolicy if args.ego_idm \
                                 else ReplayEgoCarPolicy 

    env = WaymoEnv(config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if skill_based and not reskill:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])
        prior_path = f'{args.load_model}/skill_prior_best.pth' if not args.no_prior else None
        vae_path = f'{args.load_model}/skill_vae_best.pth'
        # By default, always use benign prior; force use of shared library
        adv_prior_path = None
        # adv_prior_path = f'{args.load_model}/adv_skill_prior_best.pth' if 'adv_prior' in args.load_model else None
        device = torch.device('cpu')
        policy = skill_model.SkillModel(vae_path=vae_path, prior_path=prior_path, adv_prior_path=adv_prior_path, device=device)
    elif args.load_model != '':
        # Assume to load a saved TD3 model for now
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.discount,
            "tau": args.tau,
        }
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        if reskill:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kwargs["device"] = device
            meta_info = np.load(f'./models/{args.load_model}_meta.npy', allow_pickle=True).item()
            kwargs.update(meta_info)
            policy = reskill_model.ReSkillModel(**kwargs)
        else:
            policy = TD3.TD3(**kwargs)
        policy_file = args.load_model
        policy.load(f"./models/{policy_file}")
    else:
        policy = None
    
    if args.skill_based_adv:
        adv_generator.load_skill_model(env, args)
    elif args.model_adv:
        adv_generator.load_other_model(env, args)
    elif args.current_model_adv:
        adv_generator.load_current_model(env, args, policy)
    elif args.goose_adv:
        kwargs = {
            "state_dim": goose_train.state_dim,
            "action_dim": goose_train.action_dim,
            "goal_dim": goose_train.goal_dim,
            "max_action": goose_train.max_action,
            "discount": args.discount,
            "tau": args.tau
        }
        goose_policy = TD3_GRU.TD3GRU(**kwargs)
        goose_policy.load(f"./goose_models/{args.goose_adv_path}")
        goose_policy.shared_gru.eval()
        goose_policy.actor.eval()
        goose_policy.critic.eval()

    attack_cnt = 0
    route_completion = 0
    out_of_road_cnt = 0
    arrive_dest_cnt = 0
    time_cost = 0.
    tot_eps = 0

    if args.scenario_id == -1:
        pbar = trange(400, 500) if (env_name == 'eval' and not use_hard) else trange(0, 100) if use_hard else trange(0, 400)
    else:
        pbar = trange(args.scenario_id, args.scenario_id + 1)
    # TODO: limit FPS to make it slower and easier to visualize?
    for i in pbar:
        tot_eps += 1

        ######################## First Round : log the normal scenario ########################
        state = env.reset(force_seed=i)
        if skill_based:
            policy.reset_current_skill()
        
        done = False
        ep_timestep = 0
        adv_generator.before_episode(env)   # initialization before each episode
        if args.goose_adv:
            adv_generator.generate()
            adv_generator.frozen_new_episode(env)
            adv_generator.frozen_set_info(env)

        if args.obs_all:
            other_states = observe_others(env)

        env.render(**extra_args)
        env.engine._top_down_renderer.set_adv(adv_generator.adv_agent)
        env.vehicle.ego_crash_flag = False

        # TODO: consider using noisy action selection, add variance, for advgen stuff; 4 noisy + 1 concrete in normal
        #action = (
            #policy.select_action(np.array(state))
            #+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
        #).clip(-max_action, max_action)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Want to save the ego data even in this phase.
            if args.save32:
                normal_save_info = {'normal_tracks': []}
            if args.goose_adv:
                adv_generator.frozen_before_episode()
            while True:
                # Skip this phase if no_adv specified
                if args.goose_adv:
                    adv_generator.frozen_episode_step(env)
                if args.no_adv:
                    break
                adv_generator.log_AV_history()    # log the ego car's states at every step

                if policy is None: # i.e., is this replay only?
                    action = np.array([1.0, 0.0])
                    if args.save32:
                        track_info, _, _ = get_tracks(env, save_model_name)
                    state, r, done, info = env.step(action)
                else:
                    action = policy.select_action(np.array(state))
                    if args.save32:
                        track_info, _, _ = get_tracks(env, save_model_name)
                    state, r, done, info = env.step(action)
                if args.save32:
                    normal_save_info['normal_tracks'].append(track_info)
                if args.obs_all:
                    other_states = observe_others(env)

                ep_timestep += 1

                if done:
                    adv_generator.after_episode(update_AV_traj=(not args.open_loop), mode='train')
                    break
            if args.goose_adv:
                ego_crash = env.vehicle.crash_vehicle if save_model_name != 'ego_replay' else env.vehicle.ego_crash_flag
                adv_generator.frozen_after_episode(goose_policy, env, ego_crash, null_action_if_crash=True)
        
        ################ Second Round : create the adversarial counterpart #####################

        # Want to also log the observation (i.e. state) for replay/iterative skill embedding!
        save_info = {'tracks': [], 'extra_ego_info': [], 'extra_other_info': [], 'obs': [], 'actions': [], 
                     'done': None, 'route_completion': 0, 'gen_time': 0}
        if args.obs_all:
            save_info['other_obs'] = []
            save_info['other_actions'] = []
        save_info['adv_agent'] = adv_generator.adv_agent
        save_info['ego_agent'] = 'default_agent'
        # Want to save the original track too
        if args.save32:
            save_info.update(normal_save_info)
            old_save_info = deepcopy(save_info)

        # Build a history of (normal, adv1, adv2, ..., adv n-1) to generate final nth adv, to match training well
        # If n = 1, history is (normal); if n = 5, history is (normal, adv1, adv2, adv3, adv4)
        # TODO: if save32, iterate 32 times and save each one with a different key...
        n_iter = args.AV_traj_num if not args.save32 else 32
        if args.save32:
            initial_all_traj = None
            initial_all_traj_prob = None
            initial_all_col_prob = None
            initial_all_last_state = None
            initial_adv_traj_id = None

        for n_adv in range(n_iter):
            final_gen = (n_adv == args.AV_traj_num - 1) or args.save32
            # Reset save_info after each iteration
            if args.save32:
                save_info = deepcopy(old_save_info)

            state = env.reset(force_seed=i)
            if args.goose_adv:
                adv_generator.before_episode(env)
                adv_generator.generate()
                # No need to frozen_new_episode()
                adv_generator.frozen_set_info(env)
            if skill_based:
                policy.reset_current_skill()
            env.vehicle.ego_crash_flag = False
            done = False
            ep_timestep = 0

            t0 = time.time()
            if not args.goose_adv:
                adv_generator.before_episode(env)   # initialization before each episode
            if not args.no_adv and not args.goose_adv: # only adv generate if told to
                if not args.save32:
                    adv_generator.generate(mode='train')            # Adversarial scenario generation with the logged history corresponding to the current env 
                elif args.save32 and n_adv == 0:
                    adv_generator.generate(mode='train')            # Adversarial scenario generation with the logged history corresponding to the current env 
                    initial_all_traj = deepcopy(adv_generator.all_traj)
                    initial_all_traj_prob = deepcopy(adv_generator.all_traj_prob)
                    initial_all_col_prob = deepcopy(adv_generator.all_col_prob)
                    initial_all_last_state = deepcopy(adv_generator.all_last_state)
                    initial_adv_traj_id = adv_generator.adv_traj_id
                elif args.save32:
                    adv_generator.all_traj = deepcopy(initial_all_traj)
                    adv_generator.all_traj_prob = deepcopy(initial_all_traj_prob)
                    adv_generator.all_col_prob = deepcopy(initial_all_col_prob)
                    adv_generator.all_last_state = deepcopy(initial_all_last_state)
                    adv_generator.adv_traj_id = initial_adv_traj_id
                cur_adv_traj = adv_generator.adv_traj if not args.save32 else adv_generator.all_traj[n_adv]
                if args.save32:
                    save_info['decoder_last_state'] = adv_generator.all_last_state[n_adv]
                    save_info['adv_traj_id'] = adv_generator.adv_traj_id
                adv_generator.adv_traj = cur_adv_traj
                env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,cur_adv_traj) # set the adversarial traffic
            # Observe after, to update navi info
            if args.obs_all:
                other_states = observe_others(env)
            t1 = time.time()
            time_cost += t1 - t0
            
            if args.goose_adv:
                adv_generator.frozen_before_episode()
            with tempfile.TemporaryDirectory() as tmpdirname:
                while True:
                    if args.goose_adv:
                        adv_generator.frozen_episode_step(env)
                    adv_generator.log_AV_history()    # log the ego car's states at every step

                    if policy is None:
                        action = np.array([1.0, 0.0])
                    else:
                        action = policy.select_action(np.array(state))

                    if final_gen:
                        # action t should correspond to the action to be taken given state t
                        save_info['obs'].append(state)
                        if args.obs_all:
                            save_info['other_obs'].append(other_states)
                        track_info, extra_ego_info, extra_other_info = get_tracks(env, save_model_name)
                        save_info['tracks'].append(track_info)
                        save_info['extra_ego_info'].append(extra_ego_info)
                        save_info['extra_other_info'].append(extra_other_info)

                    if hasattr(adv_generator, 'before_step') and not args.goose_adv:
                        adv_generator.before_step(env, ep_timestep)
                    state, r, done, info = env.step(action)
                    if final_gen:
                        agent_policy = env.engine.get_policy('default_agent')
                        saved_action = action if not args.ego_idm else agent_policy.get_action_info()['action']
                        save_info['actions'].append(saved_action)
                        if args.obs_all:
                            other_names = [x for x in env.engine.get_objects().keys() if x != 'default_agent' and \
                                        isinstance(env.engine.get_object(x).get(x), BaseVehicle) and \
                                            x in other_states.keys()]
                            other_actions = {}
                            for name in other_names:
                                other_action = np.array([1.0, 0.0])
                                if isinstance(env.engine.get_policy(name), TrajectoryIDMPOlicy):
                                    other_action = env.engine.get_policy(name).get_action_info()['action']
                                other_actions[name] = other_action
                            save_info['other_actions'].append(other_actions)

                    if args.obs_all:
                        other_states = observe_others(env)

                    if final_gen:
                        env.render(**extra_args,text={'Generate': 'Safety-Critical Scenario'})
                        dir_path = f'{base_save_dir}/videos'
                        if save_scenario:
                            write_frame(env, tmpdirname, ep_timestep, debug=args.debug)


                    ep_timestep += 1
                    crash = env.vehicle.crash_vehicle if save_model_name != 'ego_replay' else env.vehicle.ego_crash_flag
                    if done or crash:
                        # post-processing after each episode
                        adv_generator.after_episode(update_AV_traj=(not args.open_loop), mode='train')
                        if args.goose_adv:
                            adv_generator.frozen_after_episode(goose_policy, env, crash, null_action_if_crash=True)
                        if not final_gen:
                            break

                        # Now in final_gen guaranteed, write last state info
                        save_info['obs'].append(state)
                        if args.obs_all:
                            save_info['other_obs'].append(other_states)
                        if policy is None:
                            action = np.array([1.0, 0.0])
                        else:
                            action = policy.select_action(np.array(state))

                        if args.ego_idm:
                            agent_policy = env.engine.get_policy('default_agent')
                            do_speed_control = 'default_agent'
                            saved_action = agent_policy.act(do_speed_control=do_speed_control)
                        else:
                            saved_action = action
                        save_info['actions'].append(saved_action)
                        if args.obs_all:
                            other_names = [x for x in env.engine.get_objects().keys() if x != 'default_agent' and \
                                            isinstance(env.engine.get_object(x).get(x), BaseVehicle) and \
                                                x in other_states.keys()]
                            other_actions = {}
                            for name in other_names:
                                other_action = np.array([1.0, 0.0])
                                if isinstance(env.engine.get_policy(name), TrajectoryIDMPOlicy):
                                    do_speed_control = name
                                    other_action = env.engine.get_policy(name).act(do_speed_control=do_speed_control)
                                other_actions[name] = other_action
                            save_info['other_actions'].append(other_actions)

                        track_info, extra_ego_info, extra_other_info = get_tracks(env, save_model_name)
                        save_info['tracks'].append(track_info)
                        save_info['extra_ego_info'].append(extra_ego_info)
                        save_info['extra_other_info'].append(extra_other_info)

                        save32_path = f'_traj{n_adv}' if args.save32 else ''
                        if save_scenario:
                            write_video(tmpdirname, f'{dir_path}/adv_{i}{save32_path}.mp4', debug=args.debug)
                        out_of_road = env._is_out_of_road(env.vehicle) or ('out_of_road' in info and info['out_of_road'])
                        arrive_destination = env._is_arrive_destination(env.vehicle) or ('arrive_dest' in info and info['arrive_dest'])

                        if crash:
                            attack_cnt += 1
                            save_info['done'] = 'crash'
                        elif out_of_road:
                            out_of_road_cnt += 1
                            save_info['done'] = 'out_of_road'
                        elif arrive_destination:
                            arrive_dest_cnt += 1
                            save_info['done'] = 'arrive'
                        else:
                            pass

                        route_completion += info['route_completion']
                        save_info['route_completion'] = info['route_completion']
                        save_info['gen_time'] = time_cost

                        if not args.debug:
                            np.save(f'{base_save_dir}/obs/adv_{i}{save32_path}.npy', save_info, allow_pickle=True)


                        if args.save32 and n_adv != n_iter - 1:
                            break
                        keys = ['crash', 'out_of_road', 'arrive', 'avg_route', 'avg_gen_time']
                        vals = [attack_cnt, out_of_road_cnt, arrive_dest_cnt, route_completion, time_cost]
                        benchmark_display = {k: v/tot_eps for k, v in zip(keys, vals)}
                        logger.update([v/tot_eps for v in vals[:4]], total_steps=i)
                        pbar.set_postfix(benchmark_display) # benchmarking the attack success rate and computational time
                        break

    env.close()