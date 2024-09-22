import numpy as np
import torch
import gym
import argparse
import geomdl
from geomdl import fitting
from geomdl import NURBS
import os
import pandas as pd

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.component.lane.point_lane import PointLane
from metadrive.utils.math import norm
from metadrive.policy.idm_policy import ScenarioIDMPolicy
from advgen.adv_generator_goose import AdvGenerator as AdvGeneratorGoose
from advgen.adv_generator_goose import get_polyline_dir, get_polyline_vel, get_polyline_yaw
from scipy.interpolate import NearestNDInterpolator

from safeshift.measure_utils import shift_rotate, rotate_shift

from saferl_algo import TD3_GRU,utils
from saferl_plotter.logger import SafeLogger

nurbs_deg = 3
nurbs_num_pt = 5
# Everything in adv-centric coordinates at timestep t
max_agents = 5
agent_d = 8
weight_min, weight_max = 0.1, 1000
dist_reward_shape = 0.1
accel_reward_shape = 0.01
yaw_reward_shape = 0.05
# dist_reward_shape = 0.0
# accel_reward_shape = 0.0
# yaw_reward_shape = 0.0
# TODO: should we revert this to False? But instead, just give a big reward bonus if satisfied
require_ineq = True
num_sub_goals = 6
distance_epsilon = 1

# TODO: change to
episode_max_step_freq = 500
episode_max_steps = 6
true_episode_max_steps = 6

# Empirically, FRENET is worse, but SELF_IDM could help enforce somewhat realistic visual results
# GOOSE did replay-only for self though.
FRENET = False
SELF_IDM = False

# reward shaping, requiring accel/yaw to be under threshold, etc.

# adv, ego, closest_other_ego_0, closest_other_ego_1, ..., closest_other_ego_6, 
# per each, we have (x, y, speed, accel, theta, valid) -> d = 6 * 8 = 48

# One-hot encoding
goal_dim = 3
action_dim = 3 * (nurbs_num_pt - 1) # for delta_x, delta_y, delta_weight
state_dim = int(max_agents * agent_d + action_dim + 2 * goal_dim)
gru_step = 5

# TODO: what should action space actually be, along with true_episode_max_steps...
max_action = 1
# TODO: should we remove some of these?
sub_goals = [
    [4, 8, 0.7],
    [8, 8, 0.7],
    [20, 8, 0.7],
    [4, 20, 2],
    [8, 20, 2],
    [20, 20, 2],
    [4, 1000, 20],
    [8, 1000, 20],
    [20, 1000, 20],
]


def safe_reset(env, force_seed=None):
    try:
        if force_seed is None:
            state, done = env.reset(), False
        else:
            state, done = env.reset(force_seed=force_seed), False
    except:
        state, done = env.reset(force_seed=0), False
        print('!!!!!!!!!!!!!Reset Bug!!!!!!!!!!!!!!')
    return state, done

def get_state(env, adv_generator, original_lane, traj_xy0, traj_h0, last_pos_map, last_speed_map, last_yaw_map):
    # x, y, vx, vy, heading, length, width, crash
    from cat_advgen import get_tracks
    track_info, _, _ = get_tracks(env, 'ego_idm')
    
    accel_map = {}
    yaw_rate_map = {}
    for k, v in track_info.items():
        if k not in last_pos_map:
            accel_map[k] = 0
            yaw_rate_map[k] = 0
            continue
        if k in last_yaw_map:
            yaw_rate_map[k] = (np.mod(v[4] - last_yaw_map[k] + np.pi, 2 * np.pi) - np.pi) * 10
        else:
            yaw_rate_map[k] = 0
        old_pos = last_pos_map[k]
        cur_pos = v[:2]
        if FRENET:
            cur_pos = traj_to_local(original_lane, [cur_pos])[0]
        else:
            pass
        cur_vel = (cur_pos - old_pos) * 10
        v[2:4] = cur_vel
        if k in last_speed_map:
            accel_map[k] = (np.linalg.norm(v[2:4]) - last_speed_map[k]) * 10
        else:
            accel_map[k] = 0

    ego_agent = 'default_agent'
    ego_data = track_info[ego_agent]
    adv_agent = adv_generator.adv_agent
    adv_present = adv_agent in track_info.keys()
    adv_data = None if not adv_present else track_info[adv_agent]
    other_keys = [k for k in track_info.keys() if k != ego_agent and k != adv_agent]
    other_data = np.array([track_info[k] for k in other_keys])
    if not len(other_data):
        other_data = np.empty((0, agent_d))
    other_to_ego_dists = np.linalg.norm(other_data[:, :2] - ego_data[:2], axis=-1)
    if len(other_to_ego_dists) > max_agents - 2:
        other_to_ego_idxs = np.argpartition(other_to_ego_dists, max_agents - 2)[:max_agents - 2]
    else:
        other_to_ego_idxs = np.arange(len(other_to_ego_dists))
    other_to_ego_idxs = other_to_ego_idxs[np.argsort(other_to_ego_dists[other_to_ego_idxs])]

    def data_to_state(x, k):
        if x is None:
            return np.zeros((agent_d,))
        if FRENET:
            traj_xy = traj_to_local(original_lane, [x[:2]])[0]
            traj_vxy = x[2:4]
            traj_h = [x[4]]
        else:
            traj_xy = shift_rotate(x[:2], -traj_xy0, -traj_h0)
            traj_vxy = shift_rotate(x[2:4], [0, 0], -traj_h0)
            traj_h = [x[4] - traj_h0]
        traj_a = [np.abs(accel_map[k])]
        traj_valid = [1]
        traj_angular_vel = [np.abs(yaw_rate_map[k])]
        return np.concatenate([traj_xy, traj_vxy, traj_h, traj_a, traj_angular_vel, traj_valid])
    
    all_data = [adv_data, ego_data, *([None] * (max_agents - 2))]
    if len(other_to_ego_idxs):
        all_data[2:2+len(other_to_ego_idxs)] = other_data[other_to_ego_idxs]
    n_append = (max_agents - 2) - len(other_to_ego_idxs)
    all_keys  = [adv_agent, ego_agent, *[other_keys[idx] for idx in other_to_ego_idxs], *([None] * n_append)]

    all_data = [data_to_state(x, k) for x, k in zip(all_data, all_keys)]
    state = np.array(all_data).flatten()
    return state, track_info

def get_curve(adv_generator, cur_traj, nurbs_deg=3, nurbs_num_pt=5):
    # Interpolate with nearest neighbor value
    cur_traj = np.array(cur_traj)
    cur_traj[~adv_generator.adv_valid] = np.nan
    cur_traj = pd.DataFrame(cur_traj).fillna(method='ffill').fillna(method='bfill').values

    traj_xy0, traj_h0 = cur_traj[0][:2], cur_traj[0][-1]
    traj_xy = np.array(cur_traj)[:, :2]

    original_lane = PointLane(traj_xy, width=2)
    if FRENET:
        traj_xy = traj_to_local(original_lane, traj_xy)
    else:
        traj_xy = shift_rotate(traj_xy, -traj_xy0, -traj_h0)

    # Convert into relative motion
    bspline = fitting.approximate_curve(traj_xy.tolist(), nurbs_deg, ctrlpts_size=nurbs_num_pt)
    curve = NURBS.Curve()
    curve.delta = 1/len(traj_xy)
    curve.degree = bspline.degree
    curve.ctrlpts = bspline.ctrlpts
    curve.knotvector = bspline.knotvector
    return original_lane, traj_xy0, traj_h0, curve, bspline

def traj_to_local(lane: PointLane, traj):
    return np.array([new_local_coordinates(lane, x[:2]) for x in traj])

def local_to_traj(lane: PointLane, traj):
    return np.array([lane.position(*x[:2]) for x in traj])

def new_local_coordinates(lane, position, only_in_lane_point=False):
    ret = []
    exclude_ret = []
    accumulate_len = 0

    # _debug = []
    for seg_idx, seg in enumerate(lane.segment_property):
        delta_x = position[0] - seg["start_point"][0]
        delta_y = position[1] - seg["start_point"][1]
        longitudinal = delta_x * seg["direction"][0] + delta_y * seg["direction"][1]
        lateral = delta_x * seg["lateral_direction"][0] + delta_y * seg["lateral_direction"][1]
        # _debug.append(longitudinal)

        if seg_idx == 0 and longitudinal < 0.0:
            return longitudinal, lateral
        elif seg_idx == len(lane.segment_property) - 1 and longitudinal >= seg["length"]:
            return accumulate_len + longitudinal, lateral

        if longitudinal < 0.0:
            dist_square = norm(delta_x, delta_y)
            if dist_square < seg["length"] * 2:
                current_long = accumulate_len + longitudinal
                current_lat = lateral
                return current_long, current_lat

        if not only_in_lane_point:
            ret.append([accumulate_len + longitudinal, longitudinal, lateral])
        else:
            if abs(lateral) <= lane.width / 2 and -1. <= accumulate_len + longitudinal <= lane.length + 1:
                ret.append([accumulate_len + longitudinal, longitudinal, lateral])
            else:
                exclude_ret.append([accumulate_len + longitudinal, longitudinal, lateral])
        accumulate_len += seg["length"]
    if len(ret) == 0:
        # for corner case
        ret = exclude_ret
    ret.sort(key=lambda seg: abs(seg[-1]))
    return ret[0][0], ret[0][-1]


def forward_simulate(env, adv_generator, original_lane, traj_xy0, traj_h0):
    sim_done = False
    
    episode_max_yaw = 0
    episode_max_accel = 0
    episode_min_dist = 1000
    last_pos_map = {}
    last_speed_map = {}
    last_yaw_map = {}

    sim_states = []
    num_timesteps = 0
    while True:
        if SELF_IDM and adv_generator.adv_agent in env.engine.get_objects().keys():
            adv_object = env.engine.get_object(adv_generator.adv_agent).get(adv_generator.adv_agent)
            # This is after set_adv_info has been invoked earlier
            if adv_generator.adv_agent in env.engine.map_manager.other_routes:
                adv_route = env.engine.map_manager.other_routes[adv_generator.adv_agent]
                env.engine.traffic_manager.add_policy(adv_generator.adv_agent, ScenarioIDMPolicy, adv_object, 0, adv_route, 1, True)
        state, track_info = get_state(env, adv_generator, original_lane, traj_xy0, traj_h0, last_pos_map, last_speed_map, last_yaw_map)
        sim_states.append(state)
        if state[:agent_d][-1]:
            adv_accel, adv_yaw_rate = state[:agent_d][5:7]
            episode_max_accel = max(np.abs(adv_accel), episode_max_accel)
            episode_max_yaw = max(np.abs(adv_yaw_rate), episode_max_yaw)
            raw_dist = np.linalg.norm((track_info[adv_generator.adv_agent][:2] - track_info['default_agent'][:2]))
            episode_min_dist = min(episode_min_dist, raw_dist)
        for obj_k, obj_v in track_info.items():
            last_yaw_map[obj_k] = obj_v[4]
            cur_pos = np.array(obj_v[:2].tolist())
            if FRENET:
                cur_pos = traj_to_local(original_lane, [cur_pos])[0]
            else:
                pass
            old_pos = None if obj_k not in last_pos_map else last_pos_map[obj_k]
            last_pos_map[obj_k] = cur_pos
            if old_pos is not None:
                cur_speed = np.linalg.norm(cur_pos - old_pos) * 10
                last_speed_map[obj_k] = cur_speed
        num_timesteps += 1
        if sim_done:
            break
        _, _, sim_done, _ = env.step(np.array([1.0, 0.0]))
    ego_crash = env.vehicle.crash_vehicle

    adv_traj = np.array([state[:2] for state in sim_states])
    adv_heading = np.array([state[4] for state in sim_states])
    adv_valid = np.array([state[7] for state in sim_states])
    adv_valid[np.argmax(adv_valid)] = 0
    adv_valid = adv_valid.astype(bool)

    adv_traj = adv_traj[adv_valid]
    adv_heading = adv_heading[adv_valid]
    adv_heading_vel = adv_heading[1:] - adv_heading[:-1]
    adv_heading_vel = (np.mod(adv_heading_vel + np.pi, 2 * np.pi) - np.pi) * 10
    adv_vel = np.linalg.norm(adv_traj[1:] - adv_traj[:-1], axis=-1) * 10
    adv_acc = (adv_vel[1:] - adv_vel[:-1]) * 10

    max_accel = np.abs(adv_acc).max() if len(adv_acc) else 0
    max_yaw = np.abs(adv_heading_vel).max() if len(adv_heading_vel) else 0

    achieved_goal = np.array([episode_min_dist, max_accel, max_yaw]) 
    return sim_states, ego_crash, num_timesteps, achieved_goal



def get_goal():
    # For now, just train simplest policy
    return np.array([0, 8, 0.7]), 'deceleration'

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, adv_generator, eval_episodes=100, mode='eval'):
    env = eval_env
    goal_vec, _ = get_goal()
    reward = 0
    episode_reward = 0
    episode_num = 0
    done = False

    cur_seed = None
    early_done = False
    next_seed = 400
    total_policy_steps = 0
    total_timesteps = 0
    episode_policy_steps = 0

    final_successes = 0
    final_crashes = 0
    final_dists = 0
    final_cnt = 0
    final_reward = 0
    while True:
        if done or early_done or episode_policy_steps >= episode_max_steps or cur_seed is None:
            if cur_seed is not None:
                print('#'*20)
                print(f"EVAL T: {total_policy_steps} ({total_timesteps}) Episode Num: {episode_num} Episode T: {episode_policy_steps} Reward: {episode_reward:.3f}")
                # Skip scenes which are effectively simulator issues
                if not early_done:
                    if done:
                        final_successes += 1
                    if dist_satisfied:
                        final_crashes += 1
                    final_dists += achieved_goal[0]
                    final_reward += episode_reward
                    final_cnt += 1
            if next_seed >= 400 + eval_episodes:
                break
            safe_reset(env, force_seed=next_seed)
            next_seed += 1
            print('#'*20)
            print('Starting scene', env.current_seed)
            adv_generator.before_episode(env)
            adv_generator.generate(mode=mode)
            original_lane, traj_xy0, traj_h0, curve, bspline = get_curve(adv_generator, adv_generator.adv_traj, nurbs_deg, nurbs_num_pt)
            curve.delta = 1/91
            curve_weights = np.array(curve.weights)
            ctrlpts = np.array(curve.ctrlpts)
            cur_seed = env.current_seed
            early_done = False
            episode_policy_steps = 0
            episode_reward = 0
            action = np.zeros((action_dim,))
            episode_num += 1
        
        episode_policy_steps += 1
        total_policy_steps += 1
        safe_reset(env, force_seed=cur_seed)
        adv_generator.before_episode(env)
        adv_generator.generate(mode=mode)
        curve.weights = curve_weights.tolist()
        curve.ctrlpts = ctrlpts.tolist()
        approx_xy = curve.evalpts

        if FRENET:
            adv_pos = local_to_traj(original_lane, approx_xy)
        else:
            adv_pos = rotate_shift(approx_xy, traj_xy0, traj_h0)

        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
        adv_vel = get_polyline_vel(adv_pos)
        adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))
        adv_generator.adv_traj = adv_traj
        env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_traj)	


        desired_goal = goal_vec
        curve_state = np.concatenate([curve_weights[1:], ctrlpts[1:].flatten()])
        states, ego_crash, sim_timesteps, achieved_goal = forward_simulate(env, adv_generator, original_lane, traj_xy0, traj_h0)
        total_timesteps += sim_timesteps

        print(f'{episode_policy_steps} - Achieved Goal: dist {achieved_goal[0].item():.3f}, accel {achieved_goal[1].item():.3f}, yaw_rate {achieved_goal[2].item():.3f}')
        states = np.array(states)[::gru_step]
        gru_states = policy.preprocess_feature_sequence(states)
        achieved_goal = torch.FloatTensor(achieved_goal).to(gru_states.device)
        desired_goal = torch.FloatTensor(desired_goal).to(gru_states.device)
        curve_state = torch.FloatTensor(curve_state).to(gru_states.device)
        policy_state = torch.cat([gru_states, curve_state, achieved_goal, desired_goal])

        dist_satisfied = ego_crash or achieved_goal[0] < distance_epsilon
        if require_ineq:
            ineq_satisfied = (achieved_goal[1:] < desired_goal[1:]).all()
            done = ineq_satisfied and dist_satisfied
        else:
            done = dist_satisfied

        penalty = 1 + dist_reward_shape * max(0, achieved_goal[0] - desired_goal[0] - distance_epsilon) \
                    + accel_reward_shape * max(0, achieved_goal[1] - desired_goal[1]) \
                    + yaw_reward_shape * max(0, achieved_goal[2] - desired_goal[2])
        reward = 0 if done else -penalty
        episode_reward += reward

        action = torch.FloatTensor(action).to(gru_states.device)
        reward = torch.FloatTensor([reward]).to(gru_states.device)
        done = torch.FloatTensor([done]).to(gru_states.device)

        if ego_crash:
            print(' -> EGO CRASH')
        if achieved_goal[0] < distance_epsilon:
            print(' -> DISTANCE EPSILON')
        # Now we go on to the next sim if reached goal
        if done:
            continue
        if achieved_goal[0] == 1000:
            early_done = True
            continue

        action = policy.select_action(policy_state)

        curve_weights[1:] += action[:4]
        curve_weights = np.clip(curve_weights, weight_min, weight_max)
        ctrlpts[1:, 0] += action[4:8]
        ctrlpts[1:, 1] += action[8:12]
    
    avg_reward = (final_reward/final_cnt).item()
    avg_dist = (final_dists/final_cnt).item()
    avg_crash = final_crashes/final_cnt
    avg_success = final_successes/final_cnt
    print("---------------------------------------")
    print(f"Evaluation over {final_cnt} episodes: Reward {avg_reward:.3f} Dist {avg_dist:.3f} Crash {avg_crash:.3f} Success {avg_success:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_dist, avg_crash, avg_success

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="MDWaymo")          
    # Lower due to HER
    parser.add_argument("--start_timesteps", default=250, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=100000, type=int)       # How often (time steps) we evaluate
    #parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.01, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--skill_model", default="")                 # Load in a skill prior/vae model
    parser.add_argument("--no_residual_agent", action='store_true') 
    parser.add_argument("--no_skill_agent", action='store_true')  

    parser.add_argument("--resume_timestep", type=int, default=0, help="when to start the experiment")   

    parser.add_argument('--OV_traj_num', type=int,default=32)       # number of opponent vehicle candidates
    parser.add_argument('--AV_traj_num', type=int,default=5)		# lens of ego traj deque (AV=Autonomous Vehicle is the same as EV(Ego vehcile) in the paper)
    parser.add_argument('--min_prob', type=float,default=0.1)		# The min probability of using raw data in ADV mode
    parser.add_argument('--mode', choices=['replay','cat'],\
                         help='Choose a mode (replay, cat)', default='cat')

    parser.add_argument('--rule_based', action='store_true', help='Change adversarial generation to rule-based instead of learned')
    parser.add_argument('--open_loop', action='store_true', help='Change adversarial generation to open-loop instead of learned')
    parser.add_argument('--skill_based_adv_path', default='cat_reskill_initial0', help='Path for skill-based adversarial model')
    parser.add_argument('--model_adv_path', default='cat_initial0', help='Path for other model to use for adversary')
    parser.add_argument('--collision_offset', default='10', type=str, help='Amount behind calculated trajectory to takeover. -1 = inf, var = random, 10 = 10 steps, etc.')
    parser.add_argument('--skill_based_adv', action='store_true', help='Change adversarial generation to skill-based instead of prior-only')
    parser.add_argument('--no_prior', action='store_true', help='Skip skill prior, use random sampling from skill space')
    parser.add_argument('--model_adv', action='store_true', help='Change adversarial generation to another model instead of prior-only')
    parser.add_argument('--idm_adv', action='store_true', help='Change adversarial generation to IDM instead of prior-only')
    parser.add_argument('--current_model_adv', action='store_true', help='Actually use the current model being trained too')
    parser.add_argument('--current_model_prior', choices=['normal', 'adv'], default='normal', help='For current model skill, which prior to set')
    parser.add_argument('--learned_objective', default='', type=str, help='Use decision32 learned model instead of objective.')
    parser.add_argument('--learned_objective_mode', default='both', choices=['sc', 'diff', 'both'], type=str, help='Which learned_objective to use')
    parser.add_argument('--guided', action='store_true', help='Use performance-guided generation instead of pure-random')


    parser.add_argument('--extra_tag', type=str, default='', help='Extra tag for experiment name and model')

    tmp_args = parser.parse_known_args()
    # For now, keeping it simple with ego idm only

    adv_generator = AdvGeneratorGoose(parser)
    args = parser.parse_args()

    
    file_name = 'goose'
    reskill = False
    if args.extra_tag != '':
        file_name = file_name + f'_{args.extra_tag}'

    logger = SafeLogger(exp_name=file_name, env_name=args.env, seed=args.seed,
                        fieldnames=['reward_adv', 'dist_adv', 'crash_rate_adv', 'success_adv'],
                        debug=args.debug)

    if args.save_model and not os.path.exists("./goose_models") and not args.debug:
        os.makedirs("./goose_models")
    

    config_train = dict(
                data_directory=os.path.join(os.path.dirname(__file__), "./raw_scenes_500"),
                start_scenario_index = 0,
                num_scenarios=400,
                sequential_seed = False,
                force_reuse_object_name = True,
                horizon = 50,
                no_light = True,
                no_static_vehicles = True,
                reactive_traffic = False,
                traffic_need_navigation = True,
                vehicle_config=dict(
                    lidar = dict(num_lasers=30,distance=50, num_others=3),
                    side_detector = dict(num_lasers=30),
                    lane_line_detector = dict(num_lasers=12)),
            )
    
    config_test = dict(
                data_directory=os.path.join(os.path.dirname(__file__), "./raw_scenes_500"),
                start_scenario_index = 400,
                num_scenarios=100,
                crash_vehicle_done=True,
                sequential_seed = True,
                force_reuse_object_name = True,
                horizon = 50,
                no_light = True,
                no_static_vehicles = True,
                traffic_need_navigation = True,
                reactive_traffic = False,
                vehicle_config=dict(
                    lidar = dict(num_lasers=30,distance=50, num_others=3),
                    side_detector = dict(num_lasers=30),
                    lane_line_detector = dict(num_lasers=12)),
            )

    config_train["agent_policy"] = WaymoIDMPolicy
    config_test["agent_policy"] = WaymoIDMPolicy

    # Set seeds
    env = WaymoEnv(config=config_train)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "goal_dim": goal_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3_GRU.TD3GRU(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./goose_models/{policy_file}")
    
    replay_buffer = utils.ReplayBufferPreserveGRU(policy.actor.l1.in_features, action_dim)
    goal_vec, _ = get_goal()

    reward = 0
    episode_reward = 0
    episode_num = 0
    done = False
    last_eval_step = 0

    num_timesteps_to_sim = args.max_timesteps
    total_timesteps = args.resume_timestep

    cur_seed = None
    early_done = False
    total_policy_steps = 0
    episode_policy_steps = 0
    if args.debug:
        episode_max_steps = true_episode_max_steps

    data_cache = {}

    while True:
        if done or early_done or episode_policy_steps >= episode_max_steps or cur_seed is None:
            if (episode_num + 1) % episode_max_step_freq == 0 and episode_max_steps < true_episode_max_steps:
                episode_max_steps *= 2
                episode_max_steps = min(episode_max_steps, true_episode_max_steps)
                print(f'< INCREASING MAX EPISODE STEP TO {episode_max_steps}>')
            if cur_seed is not None:
                print('#'*20)
                print(f"Total T: {total_policy_steps} ({total_timesteps}) Episode Num: {episode_num} Episode T: {episode_policy_steps} Reward: {episode_reward:.3f}")
            if total_timesteps - last_eval_step > args.eval_freq:
                print('#### PERFORMING EVAL ####')
                env.close()
                eval_env = WaymoEnv(config=config_test)
                eval_reward, eval_dist, eval_crash, eval_success = eval_policy(policy, eval_env, adv_generator, eval_episodes=100, mode='eval')
                eval_env.close()
                logger.update([eval_reward, eval_dist, eval_crash, eval_success], total_steps=total_timesteps)
                env = WaymoEnv(config=config_train)
                last_eval_step = total_timesteps
                if args.save_model and not args.debug: policy.save(f"./goose_models/{file_name}")

            safe_reset(env)
            print('#'*20)
            print('Starting scene', env.current_seed)
            if env.current_seed in data_cache:
                data_cache.pop(env.current_seed)

            adv_generator.before_episode(env)
            adv_generator.generate()
            original_lane, traj_xy0, traj_h0, curve, bspline = get_curve(adv_generator, adv_generator.adv_traj, nurbs_deg, nurbs_num_pt)
            curve.delta = 1/91
            curve_weights = np.array(curve.weights)
            ctrlpts = np.array(curve.ctrlpts)
            cur_seed = env.current_seed
            episode_policy_steps = 0
            episode_reward = 0
            last_policy_state = None
            early_done = False
            last_replay_buffer_state = None
            action = np.zeros((action_dim,))
            episode_num += 1
        else:
            # 1. Store data 
            #  -> (original_lane, traj_xy0, traj_h0, bspline, curve_weights, ctrlpts, cur_seed, episode_reward, episode_policy_steps, last_policy_state, last_replay_buffer_state, action)
            # 2. Select a new seed (0-399)
            # 3. If it exists, use it; otherwise build from scratch
            data_cache[env.current_seed] = (
                original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts,
                cur_seed, episode_reward, episode_policy_steps,
                last_policy_state, last_replay_buffer_state, action
            )
            new_seed = np.random.randint(0, 400)
            safe_reset(env, force_seed=new_seed)
            print('#'*20)

            adv_generator.before_episode(env)
            adv_generator.generate()
            if new_seed in data_cache:
                print('Resuming scene', env.current_seed)
                original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts, \
                cur_seed, episode_reward, episode_policy_steps, \
                last_policy_state, last_replay_buffer_state, action = data_cache[new_seed]
            else:
                print('Starting scene', env.current_seed)
                original_lane, traj_xy0, traj_h0, curve, bspline = get_curve(adv_generator, adv_generator.adv_traj, nurbs_deg, nurbs_num_pt)
                curve.delta = 1/91
                curve_weights = np.array(curve.weights)
                ctrlpts = np.array(curve.ctrlpts)
                cur_seed = env.current_seed
                episode_policy_steps = 0
                episode_reward = 0
                last_policy_state = None
                early_done = False
                last_replay_buffer_state = None
                action = np.zeros((action_dim,))
                episode_num += 1

        
        episode_policy_steps += 1
        total_policy_steps += 1
        safe_reset(env, force_seed=cur_seed)
        adv_generator.before_episode(env)
        adv_generator.generate()
        curve.weights = curve_weights.tolist()
        # TODO: make sure there's no exception here...
        curve.ctrlpts = ctrlpts.tolist()
        approx_xy = curve.evalpts

        if FRENET:
            adv_pos = local_to_traj(original_lane, approx_xy)
        else:
            adv_pos = rotate_shift(approx_xy, traj_xy0, traj_h0)

        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
        adv_vel = get_polyline_vel(adv_pos)
        adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))
        adv_generator.adv_traj = adv_traj
        env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_traj)	


        desired_goal = goal_vec
        curve_state = np.concatenate([curve_weights[1:], ctrlpts[1:].flatten()])
        states, ego_crash, sim_timesteps, achieved_goal = forward_simulate(env, adv_generator, original_lane, traj_xy0, traj_h0)
        total_timesteps += sim_timesteps

        states = np.array(states)[::gru_step]
        gru_states = policy.preprocess_feature_sequence(states)
        achieved_goal = torch.FloatTensor(achieved_goal).to(gru_states.device)
        desired_goal = torch.FloatTensor(desired_goal).to(gru_states.device)
        curve_state = torch.FloatTensor(curve_state).to(gru_states.device)
        policy_state = torch.cat([gru_states, curve_state, achieved_goal, desired_goal])
        replay_buffer_state = torch.FloatTensor(states).to(gru_states.device)

        dist_satisfied = ego_crash or achieved_goal[0] < distance_epsilon
        if require_ineq:
            ineq_satisfied = (achieved_goal[1:] < desired_goal[1:]).all()
            done = ineq_satisfied and dist_satisfied
        else:
            done = dist_satisfied

        penalty = 1 + dist_reward_shape * max(0, achieved_goal[0] - desired_goal[0] - distance_epsilon) \
                    + accel_reward_shape * max(0, achieved_goal[1] - desired_goal[1]) \
                    + yaw_reward_shape * max(0, achieved_goal[2] - desired_goal[2])
        reward = 0 if done else -penalty
        episode_reward += reward

        action = torch.FloatTensor(action).to(gru_states.device)
        reward = torch.FloatTensor([reward]).to(gru_states.device)
        done = torch.FloatTensor([done]).to(gru_states.device)
        if last_policy_state is None:
            last_policy_state = policy_state
            last_replay_buffer_state = replay_buffer_state
            replay_buffer.add(last_replay_buffer_state, action, replay_buffer_state, reward, done, curve_state, achieved_goal, desired_goal)
        else:
            replay_buffer.add(last_replay_buffer_state, action, replay_buffer_state, reward, done, curve_state, achieved_goal, desired_goal)
        print(f'{episode_policy_steps} (policy {total_policy_steps}, time {total_timesteps}) - Achieved Goal: dist {achieved_goal[0].item():.3f}, accel {achieved_goal[1].item():.3f}, yaw_rate {achieved_goal[2].item():.3f}, reward {reward.item():.3f}')
        
        # Now let's do HER buffer
        for her_goal_idx in np.random.choice(len(sub_goals), size=num_sub_goals, replace=False):
            her_goal = sub_goals[her_goal_idx]
            her_goal = torch.FloatTensor(her_goal).to(gru_states.device) 
            her_state = torch.clone(last_replay_buffer_state)
            her_next_state = torch.clone(replay_buffer_state)
            if require_ineq:
                her_ineq_satisfied = (achieved_goal[1:] < her_goal[1:]).all()
                her_done = (her_ineq_satisfied and achieved_goal[0] < her_goal[0]).to(float).unsqueeze(0)
            else:
                her_done = (achieved_goal[0] < her_goal[0]).to(float).unsqueeze(0)
            penalty = 1 + dist_reward_shape * max(0, achieved_goal[0] - her_goal[0]) \
                        + accel_reward_shape * max(0, achieved_goal[1] - her_goal[1]) \
                        + yaw_reward_shape * max(0, achieved_goal[2] - her_goal[2])
            her_reward = (-penalty) * (1 - her_done)
            replay_buffer.add(her_state, action, her_next_state, her_reward, her_done, curve_state, achieved_goal, her_goal)

        last_policy_state = policy_state
        last_replay_buffer_state = replay_buffer_state

        if args.debug:
            from matplotlib import pyplot as plt
            if FRENET:
                ego_pos = local_to_traj(original_lane, np.array([x[8:10] for x in states]))
            else:
                ego_pos = rotate_shift(np.array([x[8:10] for x in states]), traj_xy0, traj_h0)
            plt.clf()
            plt.plot(adv_pos[:, 0], adv_pos[:, 1], marker='.', color='blue')
            plt.plot(ego_pos[:, 0], ego_pos[:, 1], marker='.', color='red')
            plt.savefig('tmp_000.png')
            breakpoint()

        if total_timesteps >= num_timesteps_to_sim:
            break

        if ego_crash:
            print(' -> EGO CRASH')
        if achieved_goal[0] < distance_epsilon:
            print(' -> DISTANCE EPSILON')
        # Now we go on to the next sim if reached goal
        if done:
            continue
        if achieved_goal[0] == 1000:
            early_done = True
            continue

        # Now we do some training:
        if total_policy_steps >= (args.start_timesteps if args.batch_size < args.start_timesteps else args.batch_size):
            policy.train(replay_buffer, args.batch_size)
        
        if total_policy_steps < args.start_timesteps:
            action = np.random.rand(action_dim) * (max_action * 2) - max_action
        else:
            action = (
                policy.select_action(policy_state)
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
        
        if args.debug:
            action = policy.select_action(policy_state)

        curve_weights[1:] += action[:4]
        curve_weights = np.clip(curve_weights, weight_min, weight_max)
        ctrlpts[1:, 0] += action[4:8]
        ctrlpts[1:, 1] += action[8:12]
