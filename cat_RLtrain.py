import numpy as np
import torch
import gym
import argparse
import os
from tqdm import trange

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from advgen.adv_generator import AdvGenerator
from advgen.adv_generator_rule import AdvGenerator as AdvGeneratorRule
from advgen.adv_generator_hybrid import AdvGenerator as AdvGeneratorHybrid
from advgen.adv_generator_goose import AdvGenerator as AdvGeneratorGoose

from saferl_algo import TD3,TD3_GRU,utils,reskill_model
from saferl_plotter.logger import SafeLogger
import goose_train

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

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, adv_generator, eval_episodes=100, reskill=False):
    
    _rewards = [0.] * eval_episodes
    _costs = [0.] * eval_episodes

    for ep_num in range(eval_episodes):
        state, done = safe_reset(eval_env)
        if reskill:
            policy.reset_current_skill()
        adv_generator.before_episode(eval_env)
        while not done:
            adv_generator.log_AV_history()
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            # if eval_env.vehicle.crash_vehicle:
            #     print('Crash vehicle in main')

        _rewards[ep_num] = info['route_completion']
        _costs[ep_num] = float(eval_env.vehicle.crash_vehicle)
        adv_generator.after_episode(update_AV_traj=True,mode='eval')

    avg_reward_normal = sum(_rewards) / eval_episodes
    avg_cost_normal = sum(_costs) / eval_episodes


    _rewards = [0.] * eval_episodes
    _costs = [0.] * eval_episodes

    for ep_num in range(eval_episodes):
        state, done = safe_reset(eval_env)
        if reskill:
            policy.reset_current_skill()
        adv_generator.before_episode(eval_env)
        adv_generator.generate(mode='eval')
        episode_timesteps = 0
        # This is ok being env instead of eval_env, since env.engine == eval_env.engine always
        env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,adv_generator.adv_traj)
        while not done:
            episode_timesteps += 1
            action = policy.select_action(np.array(state))
            if hasattr(adv_generator, 'before_step'):
                adv_generator.before_step(eval_env, episode_timesteps - 1)
            state, reward, done, info = eval_env.step(action)
            # if eval_env.vehicle.crash_vehicle:
            #     print('Crash vehicle in adv')

        _rewards[ep_num] = info['route_completion']
        _costs[ep_num] = float(eval_env.vehicle.crash_vehicle)

    avg_reward_adv = sum(_rewards) / eval_episodes
    avg_cost_adv = sum(_costs) / eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: Reward_normal {avg_reward_normal:.3f} Cost_normal {avg_cost_normal: .3f} Reward_adv {avg_reward_adv:.3f} Cost_adv {avg_cost_adv: .3f}")
    print("---------------------------------------")
    return avg_reward_normal,avg_cost_normal,avg_reward_adv,avg_cost_adv



def goose_eval_policy(policy, goose_policy, eval_env, adv_generator, eval_episodes=100, reskill=False):
    # env = eval_env

    # pbar = trange(400, 400+eval_episodes)

    # # Do something much more similar to cat_advgen.py here
    # for i in pbar:
    #     breakpoint()
    
    # For now, just return default values since we don't need to know the progress
    return 0, 0, 0, 0


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="MDWaymo")          
    parser.add_argument("--start_timesteps", default=10000, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25000, type=int)       # How often (time steps) we evaluate
    #parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
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
    parser.add_argument('--goose_adv', action='store_true', help='Use GOOSE-adv')
    parser.add_argument('--goose_adv_path', default='goose_small_act0', help='GOOSE-adv frozen model path')
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
    if tmp_args[0].goose_adv:
        assert tmp_args[0].AV_traj_num == 5, 'AV_traj_num must be 5 for GOOSE (# of GOOSE policy steps)'
        adv_generator = AdvGeneratorGoose(parser)
    elif tmp_args[0].skill_based_adv or tmp_args[0].idm_adv or tmp_args[0].model_adv or tmp_args[0].current_model_adv:
        assert np.sum([tmp_args[0].skill_based_adv, tmp_args[0].idm_adv, tmp_args[0].model_adv, tmp_args[0].current_model_adv]) == 1, 'Conflicting adv generation policy'
        adv_generator = AdvGeneratorHybrid(parser)
    elif tmp_args[0].rule_based:
        adv_generator = AdvGeneratorRule(parser)
    else:
        adv_generator = AdvGenerator(parser)
    args = parser.parse_args()

    
    file_name = args.mode
    reskill = False
    if args.skill_model != '':
        file_name = file_name + '_reskill'
        reskill = True
    if args.no_skill_agent:
        file_name = file_name + '_no_sk_agent'
    if args.no_residual_agent:
        file_name = file_name + '_no_res_agent'
    if args.rule_based:
        file_name = file_name + '_heuristic'
    if args.goose_adv:
        file_name = file_name + '_goose'
    if args.open_loop:
        file_name = file_name + '_open'
    if args.guided:
        file_name = file_name + '_guided'
    if isinstance(adv_generator, AdvGeneratorHybrid):
        file_name = file_name + adv_generator.hybrid_name
    if len(args.learned_objective):
        if args.learned_objective_mode == 'both':
            file_name = file_name + '_learned_obj'
        else:
            file_name = file_name + f'_learned_obj_{args.learned_objective_mode}'
    if args.extra_tag != '':
        file_name = file_name + f'_{args.extra_tag}'

    logger = SafeLogger(exp_name=file_name, env_name=args.env, seed=args.seed,
                        fieldnames=['route_completion_normal','crash_rate_normal','route_completion_adv','crash_rate_adv'],
                        debug=args.debug)
    # TODO: log config? To be able to restore. Also, log to disk as well or no

    if args.save_model and not os.path.exists("./models") and not args.debug:
        os.makedirs("./models")

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

    # Set seeds
    env = WaymoEnv(config=config_train)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
        assert args.start_timesteps == 0, 'For reskill, no random exploration off policy'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs["device"] = device
        prior_path = f'{args.skill_model}/skill_prior_best.pth'
        vae_path = f'{args.skill_model}/skill_vae_best.pth'
        kwargs["vae_path"] = vae_path
        kwargs["prior_path"] = prior_path
        kwargs["no_residual_agent"] = args.no_residual_agent
        kwargs["no_skill_agent"] = args.no_skill_agent
        policy = reskill_model.ReSkillModel(**kwargs)
    else:
        policy = TD3.TD3(**kwargs)
    

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    
    # TODO: add this to cat_advgen too
    # TODO: warm start to enable or not? In first half (i.e. 500k out of 1e6 steps)
    # Warm start options: [no warm start, regular adv gen warm start, adv gen + IDM warm start]
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

    # TODO: add two of these for both agents
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    if reskill:
        replay_buffer = utils.ReplayBuffer(state_dim, policy.n_features)
        residual_buffer = utils.ReplayBuffer(policy.residual_agent.actor.l1.in_features, action_dim)

    
    # For guided generation, keep track of recent performances per episode
    # Seen keeps track of the episode nums when it is 
    train_recent_performance = [{'seen': -1, 'cost': []} for _ in range(config_train['num_scenarios'])]
    guided_random = np.random.RandomState(args.seed)
    def select_scenario(next_episode_num):
        choices = np.arange(len(train_recent_performance))
        seen_cost = np.array([next_episode_num - x['seen'] for x in train_recent_performance]) 
        perf_cost = np.array([0 if not len(x['cost']) else np.mean(x['cost'][-10:]) for x in train_recent_performance])
        base_prob = np.array([1] * len(choices))

        weights = [0.0, 0.1, 1.0]
        prob_dist = weights[0] * seen_cost + weights[1] * perf_cost + weights[2] * base_prob
        prob_dist = prob_dist / np.sum(prob_dist)

        selected_seed = int(guided_random.choice(choices, p=prob_dist))
        return selected_seed
    
    def update_recent_perf(episode_num, selected_seed, cost):
        train_recent_performance[selected_seed]['seen'] = episode_num
        train_recent_performance[selected_seed]['cost'].append(cost)

    if args.guided:
        current_scenario = select_scenario(0)
        state, done = safe_reset(env, force_seed=current_scenario)
    else:
        state, done = safe_reset(env)
    if reskill:
        policy.reset_current_skill()
    adv_generator.before_episode(env)
    is_adv_episode = False
    reward = 0
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0

    last_eval_step = 0

    for t in range(args.resume_timestep, int(args.max_timesteps)):
        
        episode_timesteps += 1
        if args.goose_adv and is_adv_episode:
            adv_generator.frozen_episode_step(env)

        adv_generator.log_AV_history()

        if reskill:
            # obs_res = torch.cat((obs, self.current_skill, a_dec), 1).cpu().detach().numpy()
            last_residual_state = policy.last_obs_res
            last_residual_action = policy.last_a_res
            last_residual_done = done
            last_residual_reward = reward

            last_skill_index = policy.last_skill_index
            # If *last* timestep was the start of a new skill
            if ((episode_timesteps - 2) % policy.seq_len) == 0:
                last_skill_state = policy.last_state
                last_skill_action = policy.last_noise_vec
                last_skill_done = done
                # Store the start of episode_reward before the skill started
                last_skill_episode_reward = episode_reward - reward

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
        
        # Calling select_action updates policy.last*
        if reskill and episode_timesteps > 1:
            # obs_res = torch.cat((obs, self.current_skill, a_dec), 1).cpu().detach().numpy()
            next_residual_state = np.concatenate([state[np.newaxis, :], policy.last_skill, policy.last_a_dec], 1)
            # Done must be False, otherwise if condition wouldn't hold
            residual_buffer.add(last_residual_state, last_residual_action, next_residual_state, last_residual_reward, last_residual_done)

            # When a skill ends
            if last_skill_index > policy.last_skill_index:
                last_skill_reward = episode_reward - last_skill_episode_reward
                next_skill_state = state[np.newaxis, :]
                replay_buffer.add(last_skill_state, last_skill_action, next_skill_state, last_skill_reward, last_skill_done)
        
        # Perform action
        if is_adv_episode and hasattr(adv_generator, 'before_step'):
            adv_generator.before_step(env, episode_timesteps - 1)
        next_state, reward, done, info = env.step(action)
        # try:
        # 	next_state, reward, done, info = env.step(action)
        # except:
        # 	done = True
        # 	print('!!!!!!!!!!!!!Step Bug!!!!!!!!!!!!!!')
        done_bool = float(done)

        # Store data in replay buffer
        if not reskill:
            replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        episode_cost += info['cost']

        if not reskill:
            # Train agent after collecting sufficient data
            if t >= (args.start_timesteps if args.batch_size < args.start_timesteps else args.batch_size):
                policy.train(replay_buffer, args.batch_size)
        else:
            target_timesteps = int(args.batch_size * policy.seq_len)
            if t - args.resume_timestep >= (args.start_timesteps if target_timesteps < args.start_timesteps else target_timesteps):
                policy.train(replay_buffer, residual_buffer, args.batch_size)

        if done:
            adv_generator.after_episode(update_AV_traj= args.mode=='cat')

            # Do one additional action, to store new constants for reskill
            if reskill:
                # obs_res = torch.cat((obs, self.current_skill, a_dec), 1).cpu().detach().numpy()
                last_residual_state = policy.last_obs_res
                last_residual_action = policy.last_a_res
                last_residual_done = done
                last_residual_reward = reward

                last_skill_index = policy.last_skill_index
                # If *last* timestep was the start of a new skill; given we want to add an additional timestep
                if ((episode_timesteps + 1 - 2) % policy.seq_len) == 0:
                    last_skill_state = policy.last_state
                    last_skill_action = policy.last_noise_vec
                    last_skill_done = done
                    # Store the start of episode_reward before the skill started
                    last_skill_episode_reward = episode_reward - reward

                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
            
                next_residual_state = np.concatenate([state[np.newaxis, :], policy.last_skill, policy.last_a_dec], 1)
                # Done must be False, otherwise if condition wouldn't hold
                residual_buffer.add(last_residual_state, last_residual_action, next_residual_state, last_residual_reward, True)

                last_skill_reward = episode_reward - last_skill_episode_reward
                next_skill_state = state[np.newaxis, :]
                replay_buffer.add(last_skill_state, last_skill_action, next_skill_state, last_skill_reward, True)
                # replay_buffer.next_state[-4:] - replay_buffer.state[-5:-1]

            print('#'*20)
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            print(f"arrive destination: {info['arrive_dest']} , route_completion: {info['route_completion']}, out of road:{info['out_of_road']}  ")

            # Reset environment
            if args.goose_adv and is_adv_episode:
                ego_crash = env.vehicle.crash_vehicle
                adv_generator.frozen_after_episode(goose_policy, env, ego_crash)
                adv_gen_done = adv_generator.frozen_should_end(env)
                if adv_gen_done:
                    # If adv_gen_done, we should call frozen_new_episode, otherwise no need...
                    adv_generator.frozen_new_episode(env)

            # Evaluate episode
            if t - last_eval_step > args.eval_freq:
                last_eval_step = t
                env.close()
                eval_env = WaymoEnv(config=config_test)
                if args.goose_adv:
                    evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv = goose_eval_policy(policy, goose_policy, eval_env, adv_generator, reskill=reskill)
                else:
                    evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv = eval_policy(policy, eval_env, adv_generator, reskill=reskill)
                eval_env.close()
                logger.update([evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv], total_steps=t + 1)
                
                env = WaymoEnv(config=config_train)

                if args.save_model and not args.debug: policy.save(f"./models/{file_name}")
            
            
            if args.guided:
                # Introduce guided stuff, based on past performance
                update_recent_perf(episode_num, current_scenario, episode_cost)
                current_scenario = select_scenario(episode_num + 1)
                state, done = safe_reset(env, current_scenario)
            else:
                state, done = safe_reset(env)

            if reskill:
                policy.reset_current_skill()
            adv_generator.before_episode(env)

            if args.mode == 'cat' and np.random.random() > max(1-(2*t/args.max_timesteps)*(1-args.min_prob),args.min_prob):
                print('ADVGEN')
                adv_generator.generate()	
                is_adv_episode = True
            else:
                print('NORMAL')
                is_adv_episode = False

            if args.goose_adv and is_adv_episode:
                adv_generator.frozen_resume_episode(env)
                adv_generator.frozen_set_info(env)
                adv_generator.frozen_before_episode()
            else:
                env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,adv_generator.adv_traj)	
            
            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            episode_num += 1 
