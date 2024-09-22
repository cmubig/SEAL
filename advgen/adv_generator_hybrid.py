import numpy as np
from collections import deque
import logging
import copy
import os
import bezier
import hashlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import tensorflow as tf
import advgen.utils, advgen.structs, advgen.globals
from advgen.modeling.vectornet import VectorNet
from advgen.adv_utils import process_data
from saferl_algo import reskill_model, skill_model, TD3
from metadrive.component.lane.point_lane import PointLane
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import ScenarioIDMPolicy

from advgen.adv_generator import load_objective_model
from decision32.train import clean_traj

MDAgentTypeConvert = dict(
    VEHICLE = 1,
    PEDESTRIAN = 2,
    CYCLIST = 3,
    OTHERS = 4,
)

MDMapTypeConvert = dict(
    LANE_FREEWAY = 1,
    LANE_SURFACE_STREET = 2,
    LANE_BIKE_LANE = 3,
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6,
    ROAD_LINE_SOLID_SINGLE_WHITE = 7,
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8,
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9,
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10,
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11,
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12,
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13,
    ROAD_EDGE_BOUNDARY = 15,
    ROAD_EDGE_MEDIAN = 16,
    STOP_SIGN = 17,
    CROSSWALK = 18,
    SPEED_BUMP = 19,
)

MDLightTypeConvert = dict(
    LANE_STATE_UNKNOWN = 0,
    LANE_STATE_ARROW_STOP = 1,
    LANE_STATE_ARROW_CAUTION = 2,
    LANE_STATE_ARROW_GO = 3,
    LANE_STATE_STOP = 4,
    LANE_STATE_CAUTION = 5,
    LANE_STATE_GO = 6,
    LANE_STATE_FLASHING_STOP = 7,
    LANE_STATE_FLASHING_CAUTION = 8,
)

def moving_average(data, window_size):
    interval = np.pad(data,window_size//2,'edge')
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(interval, window, 'valid')
    return res


def get_polyline_dir(polyline):
    if polyline.ndim == 1:
        return np.zeros(3)
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]
    diff = polyline_post - polyline
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir

def get_polyline_yaw(polyline):
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    diff = polyline_post - polyline
    polyline_yaw = np.arctan2(diff[:,1],diff[:,0])
    polyline_yaw[-1] = polyline_yaw[-2]
    #polyline_yaw = np.where(polyline_yaw<0,polyline_yaw+2*np.pi,polyline_yaw)
    for i in range(len(polyline_yaw)-1):
        if polyline_yaw[i+1] - polyline_yaw[i] > 1.5*np.pi:
            polyline_yaw[i+1] -= 2*np.pi
        elif polyline_yaw[i] - polyline_yaw[i+1] > 1.5*np.pi:
            polyline_yaw[i+1] += 2*np.pi
    return moving_average(polyline_yaw, window_size = 5)

def get_polyline_vel(polyline):
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]
    diff = polyline_post - polyline
    polyline_vel = diff / 0.1
    return polyline_vel

###   l1 [xa, ya, xb, yb]   l2 [xa, ya, xb, yb]
def Intersect(l1, l2):
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]

    if a*b < 0 and c*d < 0:
        return True
    else:
        return False


class AssignedSkillPolicy(BasePolicy):
    def __init__(self, control_object, model, env):
        super(AssignedSkillPolicy, self).__init__(control_object=control_object)
        self.model = model
        self.env = env
    
    def act(self, *args, **kwargs):
        from cat_advgen import observe_other
        state = observe_other(self.env, self.control_object.name)
        action = self.model.select_action(np.array(state))
        return action

class AssignedOtherPolicy(BasePolicy):
    def __init__(self, control_object, model, env):
        super(AssignedOtherPolicy, self).__init__(control_object=control_object)
        self.model = model
        self.env = env
    
    def act(self, *args, **kwargs):
        from cat_advgen import observe_other
        state = observe_other(self.env, self.control_object.name)
        action = self.model.select_action(np.array(state))
        return action

class AssignedCurrentPolicy(BasePolicy):
    def __init__(self, control_object, model, env):
        super(AssignedCurrentPolicy, self).__init__(control_object=control_object)
        self.model = model
        self.env = env
    
    def act(self, *args, **kwargs):
        from cat_advgen import observe_other
        state = observe_other(self.env, self.control_object.name)
        action = self.model.select_action(np.array(state))
        return action

class AdvGenerator():
    def __init__(self,parser):
        advgen.utils.add_argument(parser)
        parser.set_defaults(other_params=['l1_loss','densetnt', 'goals_2D', 'enhance_global_graph' ,'laneGCN' ,'point_sub_graph', 'laneGCN-4' ,'stride_10_2' ,'raster' ,'train_pair_interest'])
        parser.set_defaults(mode_num=32)
        args = parser.parse_args()
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
        logger = logging.getLogger(__name__)    
        advgen.utils.init(args,logger)

        self.objective_model, self.objective_model_mode = load_objective_model(args)

        self.rule_based = args.rule_based
        if self.objective_model is not None and self.rule_based:
            raise ValueError("Cannot apply objective_model to rule_based")

        self.idm_adv = args.idm_adv
        self.model_adv = args.model_adv
        self.current_model_adv = args.current_model_adv
        self.current_model_prior = args.current_model_prior

        if not self.rule_based:
            self.model = VectorNet(args).to(0)
            self.model.eval()

            self.model.load_state_dict(torch.load('./advgen/pretrained/densetnt.bin'))

        if self.idm_adv:
            self.skill_path = None
            self.other_path = None
            self.skill_based_policy = None
            self.reskill = False
            self.hybrid_name = '_idm'
        elif self.current_model_adv:
            self.skill_path = None
            self.other_path = None
            self.reskill = False
            self.hybrid_name = '_current'
            if self.current_model_prior == 'adv':
                self.hybrid_name += '_adv_prior'
        elif self.model_adv:
            self.skill_path = None
            self.skill_based_policy = None
            self.reskill = False
            self.hybrid_name = f'_model_{args.model_adv_path}'
            self.other_path = args.model_adv_path
        else:
            self.skill_path = args.skill_based_adv_path
            self.other_path = None
            self.reskill = 'reskill' in self.skill_path and 'saved_skill_models' not in self.skill_path
            
            if self.reskill:
                sus_hash = hashlib.shake_256(self.skill_path.encode()).hexdigest(5)
                self.hybrid_name = f'_reskill_{sus_hash}'
            else:
                expected_prefix = 'reskill/reskill/results/saved_skill_models/'
                well_formatted = self.skill_path.startswith(expected_prefix)
                well_formatted = well_formatted and ('/' not in self.skill_path.split(expected_prefix)[-1])
                assert well_formatted, 'Unexpected skill_based model path'
                # TODO: get a better name from this?
                #sus_hash = hashlib.shake_256(args.load_model.encode()).hexdigest(5)
                sus_hash = self.skill_path.split(expected_prefix)[-1]
                prior_name = '_no_prior' if args.no_prior else ''

                self.hybrid_name = f'_skill{prior_name}_{sus_hash}'

        if args.collision_offset == 'var':
            collision_rand = np.random.RandomState(42)
            collision_choices = [0, 5, 10, 15, 20, 25, 30, 35, 40]
            self.collision_offset_gen = lambda: collision_choices[int(collision_rand.rand()*len(collision_choices))]
            collision_offset_name = 'var'
        else:
            collision_val = np.inf if args.collision_offset == '-1' else int(args.collision_offset)
            self.collision_offset_gen = lambda: collision_val
            collision_offset_name = str(collision_val)

        collision_offset_name = f'_co_{collision_offset_name}' if collision_offset_name != '10'  else ''
        self.collision_offset = self.collision_offset_gen()
        self.hybrid_name += collision_offset_name

        self.skill_based_policy = None
        self.other_model_policy = None
        self.current_model_policy = None
        self.args = args
        self.open_loop = args.open_loop

        self.storage = {}
    
    def load_other_model(self, env, args):
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
        policy = TD3.TD3(**kwargs)
        policy_file = self.other_path
        policy.load(f"./models/{policy_file}")
        self.other_model_policy = policy

    def load_skill_model(self, env, args):
        if self.idm_adv:
            return
        if not self.reskill:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0] 
            max_action = float(env.action_space.high[0])
            prior_path = f'{self.skill_path}/skill_prior_best.pth' if not args.no_prior else None
            vae_path = f'{self.skill_path}/skill_vae_best.pth'
            adv_prior_path = f'{self.skill_path}/adv_skill_prior_best.pth' if 'adv_prior' in self.skill_path else None
            device = torch.device("cpu")
            policy = skill_model.SkillModel(vae_path=vae_path, prior_path=prior_path, adv_prior_path=adv_prior_path, device=device)
            self.skill_based_policy = policy
        else:
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kwargs["device"] = device
            meta_info = np.load(f'./models/{self.skill_path}_meta.npy', allow_pickle=True).item()
            kwargs.update(meta_info)
            # TODO: provide parameter to not override, i.e. use benign prior instead of assuming Adversarial Prior
            if os.path.exists(meta_info['prior_path'].replace('skill_prior_best.pth', 'adv_skill_prior_best.pth')):
                meta_info['prior_path'] = meta_info['prior_path'].replace('skill_prior_best.pth', 'adv_skill_prior_best.pth')
            policy = reskill_model.ReSkillModel(**kwargs)
            policy_file = self.skill_path
            policy.load(f"./models/{policy_file}")
            self.skill_based_policy = policy
    
    def load_current_model(self, env, args, current_policy):
        # Intentionally do a *shallow* copy, so that elements like current_policy.skill_vae point to the *same* object
        # -> i.e., so setting weights affects both
        if isinstance(current_policy, reskill_model.ReSkillModel):
            self.current_model_policy = copy.copy(current_policy)
            if self.current_model_prior != 'adv':
                return
            new_prior_path = self.current_model_policy.prior_path.replace('skill_prior_best.pth', 'adv_skill_prior_best.pth')
            assert os.path.exists(new_prior_path), 'Adv path must exist'
            self.current_model_policy.prior_path = new_prior_path
            skill_prior = torch.load(new_prior_path, map_location=self.current_model_policy.device)
            for i in skill_prior.bijectors:
                i.device = self.current_model_policy.device
            self.current_model_policy.skill_prior = skill_prior
        elif isinstance(current_policy, TD3.TD3):
            self.current_model_policy = copy.copy(current_policy)
        else:
            raise NotImplementedError('current policy for training must be either a ReSkill or TD3 model.')
    
    def before_episode(self,env):
        self.env = env
        self.ego_traj = []
        self.ego_prob = 1.
        self.adv_traj = []
        self.adv_traj_timestep = 0
        self.adv_name = None
        # Reset collision offset per episode, to support variabile offsets
        self.collision_offset = self.collision_offset_gen()

        if self.skill_based_policy is not None:
            self.skill_based_policy.reset_current_skill()
        elif self.current_model_policy is not None and hasattr(self.current_model_policy, 'reset_current_skill'):
            self.current_model_policy.reset_current_skill()
        self.assigned_policy = False

        if not self.storage.get(self.env.current_seed):
            traffic_motion_feat,adv_agent,ego_navigation_route,ego_hist,adv_past,adv_navigation_route = self._parse()
            
            AV_trajs = deque(maxlen=self.args.AV_traj_num)
            AV_hist = deque(maxlen=self.args.AV_traj_num)
            AV_probs = deque(maxlen=self.args.AV_traj_num)
            
            for _ in range(self.args.AV_traj_num):
                AV_trajs.append(ego_navigation_route)
                AV_hist.append(ego_hist)
                AV_probs.append(1.)

            AV_trajs_eval = deque(maxlen=1)
            AV_trajs_eval.append(ego_navigation_route)
            AV_hist_eval = deque(maxlen=1)
            AV_hist_eval.append(ego_hist)
            
            ego_obj = self.env.engine.get_objects(['default_agent']).get('default_agent')
            try:
                adv_obj = self.env.engine.get_objects([adv_agent]).get(adv_agent)
            except:
                adv_obj = ego_obj
            self.storage[self.env.current_seed] = dict(
                traffic_motion_feat = traffic_motion_feat,
                adv_agent = adv_agent,
                adv_past = adv_past,
                adv_navigation_route = adv_navigation_route,
                adv_info = dict(w=adv_obj.top_down_width,l=adv_obj.top_down_length),
                ego_info = dict(w=ego_obj.top_down_width,l=ego_obj.top_down_length),
                AV_trajs = AV_trajs,
                AV_probs = AV_probs,
                AV_trajs_eval = AV_trajs_eval,
                AV_hist = AV_hist,
                AV_hist_eval = AV_hist_eval
            )
        
        self.ego_vel = []
        self.ego_heading = []
        
    def after_episode(self,update_AV_traj=False,mode='train'):
        if update_AV_traj and not self.open_loop:
            latest_ego_hist = np.array(self.ego_traj)[:11]
            latest_ego_traj = np.array(self.ego_traj)[11:91]
            if len(latest_ego_traj)<10:
                print('Ignore traj less than 1s')
                return
            else:
                print('Latest traj is updated')
            if mode == 'train':
                self.storage[self.env.current_seed]['AV_trajs'].append(latest_ego_traj)
                self.storage[self.env.current_seed]['AV_hist'].append(latest_ego_hist)
                self.storage[self.env.current_seed]['AV_probs'].append(self.ego_prob)
            elif mode == 'eval':
                self.storage[self.env.current_seed]['AV_trajs_eval'].append(latest_ego_traj)
                self.storage[self.env.current_seed]['AV_hist_eval'].append(latest_ego_hist)
            else:
                raise NotImplementedError

    

    def log_AV_history(self,action_prob=1.0):
        obj = self.env.engine.get_object('default_agent').get('default_agent')
        self.ego_traj.append(obj.position)
        self.ego_vel.append(obj.velocity)
        self.ego_heading.append(obj.heading_theta)
        self.ego_prob *= action_prob


    def _parse(self):
        scenario_data = self.env.engine.data_manager._scenario[self.env.current_seed]
        
        default_agent = scenario_data['metadata']['sdc_id']
        objects_of_interest = scenario_data['metadata']['objects_of_interest']
        assert len(objects_of_interest) == 2 and default_agent in objects_of_interest
        objects_of_interest.remove(default_agent)
        adv_agent = objects_of_interest[0]
        
        raw_map_features = scenario_data['map_features']
        raw_dynamic_map_states = scenario_data['dynamic_map_states']
        raw_tracks_features = scenario_data['tracks']

        tracks_ids = list(raw_tracks_features.keys())
        tracks_ids.remove(default_agent)
        tracks_ids.remove(adv_agent)
        tracks_ids = [default_agent,adv_agent] + tracks_ids
       
        map_features = {
        'roadgraph_samples/dir': np.full([20000,3], -1 , dtype=np.float32),
        'roadgraph_samples/id': np.full([20000,1], -1 , dtype=np.int64),
        'roadgraph_samples/type': np.full([20000,1], -1 , dtype=np.int64),
        'roadgraph_samples/valid': np.full([20000,1], 0 , dtype=np.int64),
        'roadgraph_samples/xyz': np.full([20000,3], -1 , dtype=np.float32),
        }

        state_features = {
            'state/id': np.full([128,], -1 , dtype=np.int64),
            'state/type': np.full([128,], 0 , dtype=np.int64),
            'state/is_sdc': np.full([128,], 0 , dtype=np.int64),
            'state/tracks_to_predict': np.full([128,], 0 , dtype=np.int64),
            'state/current/bbox_yaw': np.full([128,1],-1 , dtype=np.float32),
            'state/current/height': np.full([128,1], -1 , dtype=np.float32),
            'state/current/length': np.full([128,1], -1 , dtype=np.float32),
            'state/current/valid':np.full([128,1], 0 , dtype=np.int64),
            'state/current/vel_yaw':np.full([128,1], -1 , dtype=np.float32),
            'state/current/velocity_x': np.full([128,1], -1 , dtype=np.float32),
            'state/current/velocity_y': np.full([128,1], -1 , dtype=np.float32),
            'state/current/width': np.full([128,1], -1 , dtype=np.float32),
            'state/current/x': np.full([128,1], -1 , dtype=np.float32),
            'state/current/y': np.full([128,1], -1 , dtype=np.float32),
            'state/current/z': np.full([128,1], -1 , dtype=np.float32),
            'state/past/bbox_yaw': np.full([128,10], -1 , dtype=np.float32),
            'state/past/height': np.full([128,10], -1 , dtype=np.float32),
            'state/past/length': np.full([128,10], -1 , dtype=np.float32),
            'state/past/valid':np.full([128,10], 0 , dtype=np.int64),
            'state/past/vel_yaw':np.full([128,10], -1 , dtype=np.float32),
            'state/past/velocity_x': np.full([128,10], -1 , dtype=np.float32),
            'state/past/velocity_y': np.full([128,10], -1 , dtype=np.float32),
            'state/past/width': np.full([128,10], -1 , dtype=np.float32),
            'state/past/x': np.full([128,10], -1 , dtype=np.float32),
            'state/past/y': np.full([128,10], -1 , dtype=np.float32),
            'state/past/z': np.full([128,10], -1 , dtype=np.float32),
            'state/future/bbox_yaw': np.full([128,80], -1 , dtype=np.float32),
            'state/future/height': np.full([128,80], -1 , dtype=np.float32),
            'state/future/length': np.full([128,80], -1 , dtype=np.float32),
            'state/future/valid':np.full([128,80], 0 , dtype=np.int64),
            'state/future/vel_yaw':np.full([128,80], -1 , dtype=np.float32),
            'state/future/velocity_x': np.full([128,80], -1 , dtype=np.float32),
            'state/future/velocity_y': np.full([128,80], -1 , dtype=np.float32),
            'state/future/width': np.full([128,80], -1 , dtype=np.float32),
            'state/future/x': np.full([128,80], -1 , dtype=np.float32),
            'state/future/y': np.full([128,80], -1 , dtype=np.float32),
            'state/future/z': np.full([128,80], -1 , dtype=np.float32),

        }

        traffic_light_features = {
            'traffic_light_state/current/state': np.full([1,16], -1 , dtype=np.int64),
            'traffic_light_state/current/valid': np.full([1,16], 0 , dtype=np.int64),
            'traffic_light_state/current/id': np.full([1,16], -1 , dtype=np.int64),
            'traffic_light_state/current/x': np.full([1,16], -1 , dtype=np.float32),
            'traffic_light_state/current/y': np.full([1,16], -1 , dtype=np.float32),
            'traffic_light_state/current/z': np.full([1,16], -1 , dtype=np.float32),
            'traffic_light_state/past/state': np.full([10,16], -1 , dtype=np.int64),
            'traffic_light_state/past/valid': np.full([10,16], 0 , dtype=np.int64),
            'traffic_light_state/past/x': np.full([10,16], -1 , dtype=np.float32),
            'traffic_light_state/past/y': np.full([10,16], -1 , dtype=np.float32),
            'traffic_light_state/past/z': np.full([10,16], -1 , dtype=np.float32),
            'traffic_light_state/past/id': np.full([10,16], -1 , dtype=np.int64),
        }

        count = 0

        for k,v in raw_map_features.items():
            _id = int(k)
            _type = MDMapTypeConvert[v['type']] if v['type'] in MDMapTypeConvert else -1

            if _type in [17]:
                _poly = v['position']
            elif _type in [18,19] or v['type'] == 'DRIVEWAY':
                _poly = v['polygon']
            else:
                _poly = v['polyline']
            
            _dir = get_polyline_dir(_poly)

            
            # clip > 20000
            try:
                map_features['roadgraph_samples/xyz'][count:count+len(_poly)] = _poly
                map_features['roadgraph_samples/dir'][count:count+len(_poly)] = _dir
                map_features['roadgraph_samples/id'][count:count+len(_poly)] = _id
                map_features['roadgraph_samples/type'][count:count+len(_poly)] = _type
                map_features['roadgraph_samples/valid'][count:count+len(_poly)] = 1
            except:
                map_features['roadgraph_samples/xyz'][count:20000] = _poly[:20000-count]
                map_features['roadgraph_samples/dir'][count:20000] = _dir[:20000-count]
                map_features['roadgraph_samples/id'][count:20000] = _id
                map_features['roadgraph_samples/type'][count:20000] = _type
                map_features['roadgraph_samples/valid'][count:20000] = 1
                break

            count += len(_poly)


        tracks_ids = tracks_ids[:128]

        state_features['state/id'][:len(tracks_ids)] = tracks_ids
        state_features['state/is_sdc'][0] = 1
        state_features['state/tracks_to_predict'][:2] = 1

        for i, track_id in enumerate(tracks_ids):

            track_data = raw_tracks_features.get(track_id)
            
            # construct ego navigation route
            if i == 0:
                ego_navigation_route = track_data['state']['position'][11:,:2]
                ego_hist = track_data['state']['position'][:11, :2]
            
            if i == 1:
                adv_past = track_data['state']['position'][:11,:2]
                adv_navigation_route = track_data['state']['position'][11:,:2]

            state_features['state/type'][i] = MDAgentTypeConvert[track_data['type']]

            for j in range(0,10):
                state_features['state/past/x'][i][j] = track_data['state']['position'][j][0]
                state_features['state/past/y'][i][j] = track_data['state']['position'][j][1]
                state_features['state/past/z'][i][j] = track_data['state']['position'][j][2]
                state_features['state/past/bbox_yaw'][i][j] = track_data['state']['heading'][j]
                state_features['state/past/velocity_x'][i][j] = track_data['state']['velocity'][j][0]
                state_features['state/past/velocity_y'][i][j] = track_data['state']['velocity'][j][1]
                state_features['state/past/vel_yaw'][i][j] = np.arctan2(track_data['state']['velocity'][j][1],track_data['state']['velocity'][j][0])
                state_features['state/past/width'][i][j] = track_data['state']['width'][j]
                state_features['state/past/height'][i][j] = track_data['state']['height'][j]
                state_features['state/past/length'][i][j] = track_data['state']['length'][j]
                state_features['state/past/valid'][i][j] = track_data['state']['valid'][j]
            

            for j in range(10,11):
                state_features['state/current/x'][i] = track_data['state']['position'][j][0]
                state_features['state/current/y'][i] = track_data['state']['position'][j][1]
                state_features['state/current/z'][i] = track_data['state']['position'][j][2]
                state_features['state/current/bbox_yaw'][i] = track_data['state']['heading'][j]
                state_features['state/current/velocity_x'][i] = track_data['state']['velocity'][j][0]
                state_features['state/current/velocity_y'][i] = track_data['state']['velocity'][j][1]
                state_features['state/current/vel_yaw'][i] = np.arctan2(track_data['state']['velocity'][j][1],track_data['state']['velocity'][j][0])
                state_features['state/current/width'][i] = track_data['state']['width'][j]
                state_features['state/current/height'][i] = track_data['state']['height'][j]
                state_features['state/current/length'][i] = track_data['state']['length'][j]
                state_features['state/current/valid'][i] = track_data['state']['valid'][j]

            
            for j in range(11,91):
                state_features['state/future/x'][i][j-11] = track_data['state']['position'][j][0]
                state_features['state/future/y'][i][j-11] = track_data['state']['position'][j][1]
                state_features['state/future/z'][i][j-11] = track_data['state']['position'][j][2]
                state_features['state/future/bbox_yaw'][i][j-11] = track_data['state']['heading'][j]
                state_features['state/future/velocity_x'][i][j-11] = track_data['state']['velocity'][j][0]
                state_features['state/future/velocity_y'][i][j-11] = track_data['state']['velocity'][j][1]
                state_features['state/future/vel_yaw'][i][j-11] = np.arctan2(track_data['state']['velocity'][j][1],track_data['state']['velocity'][j][0])
                state_features['state/future/width'][i][j-11] = track_data['state']['width'][j]
                state_features['state/future/height'][i][j-11] = track_data['state']['height'][j]
                state_features['state/future/length'][i][j-11] = track_data['state']['length'][j]
                state_features['state/future/valid'][i][j-11] = track_data['state']['valid'][j]
        
        for i,v in enumerate(raw_dynamic_map_states.values()):
            if i == 16: break

            if v['type'] != 'TRAFFIC_LIGHT': continue

            for j in range(0,10):
                _state = v['state']['object_state'][j]
                if _state: 
                    traffic_light_features['traffic_light_state/past/state'][j][i] = MDLightTypeConvert[_state]
                    traffic_light_features['traffic_light_state/past/valid'][j][i] = 1
                    traffic_light_features['traffic_light_state/past/id'][j][i] = int(v['lane'])
                    traffic_light_features['traffic_light_state/past/x'][j][i] = v['stop_point'][0]
                    traffic_light_features['traffic_light_state/past/y'][j][i] = v['stop_point'][1]
                    traffic_light_features['traffic_light_state/past/z'][j][i] = v['stop_point'][2]
            
            _state = v['state']['object_state'][10]
            if _state: 
                traffic_light_features['traffic_light_state/current/state'][0][i] = MDLightTypeConvert[_state]
                traffic_light_features['traffic_light_state/current/valid'][0][i] = 1
                traffic_light_features['traffic_light_state/current/id'][0][i] = int(v['lane'])
                traffic_light_features['traffic_light_state/current/x'][0][i] = v['stop_point'][0]
                traffic_light_features['traffic_light_state/current/y'][0][i] = v['stop_point'][1]
                traffic_light_features['traffic_light_state/current/z'][0][i] = v['stop_point'][2]
        
        features_description = {}
        features_description.update(map_features)
        features_description.update(state_features)
        features_description.update(traffic_light_features)
        features_description['scenario/id'] = np.array(['template'])
        features_description['state/objects_of_interest'] = state_features['state/tracks_to_predict'].copy()
        for k,v in features_description.items():
            if self.rule_based:
                # TODO: convert to CPU
                with tf.device("CPU:0"):
                    features_description[k] = tf.convert_to_tensor(v)
            else:
                features_description[k] = tf.convert_to_tensor(v)
        
        return features_description,adv_agent,ego_navigation_route,ego_hist,adv_past,adv_navigation_route

    @property
    def adv_agent(self):
        return self.storage[self.env.current_seed].get('adv_agent')
    
    def generate(self,mode='train'):
        if self.rule_based:
            traffic_motion_feat = self.storage[self.env.current_seed].get('traffic_motion_feat')
            if mode == 'train':
                # Could be jagged, but each element by main index can be float32 type
                trajs_AV = np.array(self.storage[self.env.current_seed].get('AV_trajs'), dtype='object')[0].astype(np.float32)
            elif mode == 'eval':
                trajs_AV = np.array(self.storage[self.env.current_seed].get('AV_trajs_eval'), dtype='object')[0].astype(np.float32)
                
            trajs_OV = np.array(self.storage[self.env.current_seed].get('adv_navigation_route'))
            _AV_len = int(len(trajs_AV)*0.75)
            a = trajs_OV[0]
            b = trajs_OV[int(_AV_len/3)]
            c = trajs_AV[int(_AV_len*2/3)]
            d = trajs_AV[_AV_len]
            points = np.array([[a[0],b[0],c[0],d[0]], [a[1],b[1],c[1],d[1]]])
            curve = bezier.Curve(points, degree=3)
            s_vals = np.linspace(0.0, 1.0, _AV_len)
            res = curve.evaluate_multi(s_vals).transpose((1,0))
            
            adv_past = self.storage[self.env.current_seed].get('adv_past')
            adv_pos = np.concatenate((adv_past,res,trajs_AV[_AV_len+1:]),axis=0)
            adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
            adv_vel = get_polyline_vel(adv_pos)
            self.adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))

            # Try to set self.adv_traj_timestep
            # TODO: do same bbox collision check
            traj_OV = adv_pos[11:][::5]
            yaw_OV = get_polyline_yaw(adv_pos[11:])[::5].reshape(-1,1)
            width_OV = self.storage[self.env.current_seed]['adv_info']['w']
            length_OV = self.storage[self.env.current_seed]['adv_info']['l']
            cos_theta = np.cos(yaw_OV)
            sin_theta = np.sin(yaw_OV)
            bbox_OV = np.concatenate((traj_OV,yaw_OV,\
                    traj_OV[:,0].reshape(-1,1)+0.5*length_OV*cos_theta+0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)+0.5*length_OV*sin_theta-0.5*width_OV*cos_theta,\
                    traj_OV[:,0].reshape(-1,1)+0.5*length_OV*cos_theta-0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)+0.5*length_OV*sin_theta+0.5*width_OV*cos_theta,\
                    traj_OV[:,0].reshape(-1,1)-0.5*length_OV*cos_theta-0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)-0.5*length_OV*sin_theta+0.5*width_OV*cos_theta,\
                    traj_OV[:,0].reshape(-1,1)-0.5*length_OV*cos_theta+0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)-0.5*length_OV*sin_theta-0.5*width_OV*cos_theta),axis=1)


            traj_AV = trajs_AV[::5].astype(np.float32)
            yaw_AV = get_polyline_yaw(trajs_AV.astype(np.float32))[::5].reshape(-1,1)
            width_AV = self.storage[self.env.current_seed]['ego_info']['w']
            length_AV = self.storage[self.env.current_seed]['ego_info']['l']
            cos_theta = np.cos(yaw_AV)
            sin_theta = np.sin(yaw_AV)

            bbox_AV = np.concatenate((traj_AV,yaw_AV,\
                traj_AV[:,0].reshape(-1,1)+0.5*length_AV*cos_theta+0.5*width_AV*sin_theta,\
                traj_AV[:,1].reshape(-1,1)+0.5*length_AV*sin_theta-0.5*width_AV*cos_theta,\
                traj_AV[:,0].reshape(-1,1)+0.5*length_AV*cos_theta-0.5*width_AV*sin_theta,\
                traj_AV[:,1].reshape(-1,1)+0.5*length_AV*sin_theta+0.5*width_AV*cos_theta,\
                traj_AV[:,0].reshape(-1,1)-0.5*length_AV*cos_theta-0.5*width_AV*sin_theta,\
                traj_AV[:,1].reshape(-1,1)-0.5*length_AV*sin_theta+0.5*width_AV*cos_theta,\
                traj_AV[:,0].reshape(-1,1)-0.5*length_AV*cos_theta+0.5*width_AV*sin_theta,\
                traj_AV[:,1].reshape(-1,1)-0.5*length_AV*sin_theta-0.5*width_AV*cos_theta),axis=1)
                
            # Want to also know what timestep collision is predicted to occur at...this is already every 5 timesteps too
            '''
            B-A  F-E
            | |  | |
            C-D  G-H
            '''
            min_dist = 1000000
            res_timestep = 0
            for timestep, ((Cx1,Cy1,yaw1,xA,yA,xB,yB,xC,yC,xD,yD),(Cx2,Cy2,yaw2,xE,yE,xF,yF,xG,yG,xH,yH)) in enumerate(zip(bbox_AV,bbox_OV)):
                ego_adv_dist = np.linalg.norm([Cx1-Cx2,Cy1-Cy2])
                if ego_adv_dist < min_dist:
                    min_dist = ego_adv_dist
                    res_timestep = timestep
                if ego_adv_dist >= np.linalg.norm([0.5*length_AV,0.5*width_AV]) + np.linalg.norm([0.5*length_OV,0.5*width_OV]):
                    pass
                elif Intersect([xA,yA,xB,yB],[xE,yE,xF,yF]) or Intersect([xA,yA,xB,yB],[xF,yF,xG,yG]) or\
                    Intersect([xA,yA,xB,yB],[xG,yG,xH,yH]) or Intersect([xA,yA,xB,yB],[xH,yH,xE,yE]) or\
                    Intersect([xB,yB,xC,yC],[xE,yE,xF,yF]) or Intersect([xB,yB,xC,yC],[xF,yF,xG,yG]) or\
                    Intersect([xB,yB,xC,yC],[xG,yG,xH,yH]) or Intersect([xB,yB,xC,yC],[xH,yH,xE,yE]) or\
                    Intersect([xC,yC,xD,yD],[xE,yE,xF,yF]) or Intersect([xC,yC,xD,yD],[xF,yF,xG,yG]) or\
                    Intersect([xC,yC,xD,yD],[xG,yG,xH,yH]) or Intersect([xC,yC,xD,yD],[xH,yH,xE,yE]) or\
                    Intersect([xD,yD,xA,yA],[xE,yE,xF,yF]) or Intersect([xD,yD,xA,yA],[xF,yF,xG,yG]) or\
                    Intersect([xD,yD,xA,yA],[xG,yG,xH,yH]) or Intersect([xD,yD,xA,yA],[xH,yH,xE,yE]):
                    res_timestep = timestep
                    break

            self.adv_traj_timestep = int(len(adv_past) + 5 * res_timestep)

            return traffic_motion_feat,self.adv_traj,trajs_AV,True

        traffic_motion_feat = self.storage[self.env.current_seed].get('traffic_motion_feat')
        # Whenever the latest_ego_traj shape is less than 80
        if mode == 'train':
            trajs_AV = np.array(self.storage[self.env.current_seed].get('AV_trajs'), dtype='object')
            hist_AV = np.array(self.storage[self.env.current_seed].get('AV_hist'), dtype='object')
            probs_AV = np.array(self.storage[self.env.current_seed].get('AV_probs'))
        elif mode == 'eval':
            trajs_AV = np.array(self.storage[self.env.current_seed].get('AV_trajs_eval'))
            hist_AV = np.array(self.storage[self.env.current_seed].get('AV_hist_eval'))
            probs_AV = [1.]

        batch_data = process_data(traffic_motion_feat,self.args)
        with torch.no_grad():
            pred_trajectory, pred_score, _, pred_last_states = self.model(batch_data[0], 'cuda')

        # return traffic_motion_feat,pred_trajectory,trajs_AV

        trajs_OV = pred_trajectory[1]

        probs_OV = pred_score[1]
        probs_OV[6:] = probs_OV[6]
        probs_OV = np.exp(probs_OV)
        probs_OV = probs_OV / np.sum(probs_OV)


        res = np.zeros(32)
        res_timestep = np.zeros(32)
        min_dist = np.full(32,fill_value=1000000)

        # TODO: add in randomization or no? For now, keep as is.
        for j,prob_OV in enumerate(probs_OV):
            P1 = prob_OV
            traj_OV = trajs_OV[j][::5]
            yaw_OV = get_polyline_yaw(trajs_OV[j])[::5].reshape(-1,1)
            width_OV = self.storage[self.env.current_seed]['adv_info']['w']
            length_OV = self.storage[self.env.current_seed]['adv_info']['l']
            cos_theta = np.cos(yaw_OV)
            sin_theta = np.sin(yaw_OV)
            bbox_OV = np.concatenate((traj_OV,yaw_OV,\
                    traj_OV[:,0].reshape(-1,1)+0.5*length_OV*cos_theta+0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)+0.5*length_OV*sin_theta-0.5*width_OV*cos_theta,\
                    traj_OV[:,0].reshape(-1,1)+0.5*length_OV*cos_theta-0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)+0.5*length_OV*sin_theta+0.5*width_OV*cos_theta,\
                    traj_OV[:,0].reshape(-1,1)-0.5*length_OV*cos_theta-0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)-0.5*length_OV*sin_theta+0.5*width_OV*cos_theta,\
                    traj_OV[:,0].reshape(-1,1)-0.5*length_OV*cos_theta+0.5*width_OV*sin_theta,\
                    traj_OV[:,1].reshape(-1,1)-0.5*length_OV*sin_theta-0.5*width_OV*cos_theta),axis=1)


            for i,prob_AV in enumerate(probs_AV):
                P2 = prob_AV
                traj_AV = trajs_AV[i][::5].astype(np.float32)
                yaw_AV = get_polyline_yaw(trajs_AV[i].astype(np.float32))[::5].reshape(-1,1)
                width_AV = self.storage[self.env.current_seed]['ego_info']['w']
                length_AV = self.storage[self.env.current_seed]['ego_info']['l']
                cos_theta = np.cos(yaw_AV)
                sin_theta = np.sin(yaw_AV)
                

                bbox_AV = np.concatenate((traj_AV,yaw_AV,\
                    traj_AV[:,0].reshape(-1,1)+0.5*length_AV*cos_theta+0.5*width_AV*sin_theta,\
                    traj_AV[:,1].reshape(-1,1)+0.5*length_AV*sin_theta-0.5*width_AV*cos_theta,\
                    traj_AV[:,0].reshape(-1,1)+0.5*length_AV*cos_theta-0.5*width_AV*sin_theta,\
                    traj_AV[:,1].reshape(-1,1)+0.5*length_AV*sin_theta+0.5*width_AV*cos_theta,\
                    traj_AV[:,0].reshape(-1,1)-0.5*length_AV*cos_theta-0.5*width_AV*sin_theta,\
                    traj_AV[:,1].reshape(-1,1)-0.5*length_AV*sin_theta+0.5*width_AV*cos_theta,\
                    traj_AV[:,0].reshape(-1,1)-0.5*length_AV*cos_theta+0.5*width_AV*sin_theta,\
                    traj_AV[:,1].reshape(-1,1)-0.5*length_AV*sin_theta-0.5*width_AV*cos_theta),axis=1)
                
                P3 = 0
                uncertainty = 1.
                alpha = 0.99
                # Want to also know what timestep collision is predicted to occur at...this is already every 5 timesteps too
                '''
                B-A  F-E
                | |  | |
                C-D  G-H
                '''
                for timestep, ((Cx1,Cy1,yaw1,xA,yA,xB,yB,xC,yC,xD,yD),(Cx2,Cy2,yaw2,xE,yE,xF,yF,xG,yG,xH,yH)) in enumerate(zip(bbox_AV,bbox_OV)):
                    uncertainty *= alpha
                    ego_adv_dist = np.linalg.norm([Cx1-Cx2,Cy1-Cy2])
                    if ego_adv_dist < min_dist[j]:
                        min_dist[j] = ego_adv_dist
                        res_timestep[j] = timestep
                    if ego_adv_dist >= np.linalg.norm([0.5*length_AV,0.5*width_AV]) + np.linalg.norm([0.5*length_OV,0.5*width_OV]):
                        pass
                    elif Intersect([xA,yA,xB,yB],[xE,yE,xF,yF]) or Intersect([xA,yA,xB,yB],[xF,yF,xG,yG]) or\
                        Intersect([xA,yA,xB,yB],[xG,yG,xH,yH]) or Intersect([xA,yA,xB,yB],[xH,yH,xE,yE]) or\
                        Intersect([xB,yB,xC,yC],[xE,yE,xF,yF]) or Intersect([xB,yB,xC,yC],[xF,yF,xG,yG]) or\
                        Intersect([xB,yB,xC,yC],[xG,yG,xH,yH]) or Intersect([xB,yB,xC,yC],[xH,yH,xE,yE]) or\
                        Intersect([xC,yC,xD,yD],[xE,yE,xF,yF]) or Intersect([xC,yC,xD,yD],[xF,yF,xG,yG]) or\
                        Intersect([xC,yC,xD,yD],[xG,yG,xH,yH]) or Intersect([xC,yC,xD,yD],[xH,yH,xE,yE]) or\
                        Intersect([xD,yD,xA,yA],[xE,yE,xF,yF]) or Intersect([xD,yD,xA,yA],[xF,yF,xG,yG]) or\
                        Intersect([xD,yD,xA,yA],[xG,yG,xH,yH]) or Intersect([xD,yD,xA,yA],[xH,yH,xE,yE]):
                        P3 = uncertainty
                        res_timestep[j] = timestep
                        break

                res[j] += P1*P2*P3

        if np.any(res):
            adv_traj_id = np.argmax(res)
        else:
            adv_traj_id = np.argmin(min_dist)

        if self.objective_model is not None:
            total_scores = None
            for ego_hist, ego_traj in zip(hist_AV, trajs_AV):
                ego_full = np.concatenate((ego_hist, ego_traj), axis=0)
                ego_full = np.concatenate((ego_full, np.ones((len(ego_full), 1))), axis=1).astype(np.float64)
                clean_advs = []
                for traj_OV in trajs_OV:
                    traj_full = np.concatenate((self.storage[self.env.current_seed].get('adv_past'), traj_OV), axis=0)
                    traj_yaw = get_polyline_yaw(traj_full).reshape(-1, 1)
                    base_ones = np.ones((len(traj_full), 1))
                    # Want index (0, 1) = (x, y), (4) = (heading), (-1) = (valid)
                    traj_full = np.concatenate((traj_full, base_ones, base_ones, traj_yaw, base_ones), axis=1)
                    traj_full_clean = clean_traj(ego_full, traj_full)
                    clean_advs.append(torch.from_numpy(traj_full_clean).to(torch.float32))
                scores = self.objective_model(None, clean_advs, clean_advs)
                if total_scores is None:
                    total_scores = torch.clone(scores)
                else:
                    total_scores += scores
            if self.objective_model_mode == 'both':
                adv_traj_id = total_scores.mean(dim=-1).argmax().item()
            elif self.objective_model_mode == 'sc':
                adv_traj_id = total_scores[:, 0].argmax().item()
            elif self.objective_model_mode == 'diff':
                adv_traj_id = total_scores[:, 1].argmax().item()
            else:
                raise ValueError('Unexpected objective_model_mode')

        adv_future = trajs_OV[adv_traj_id]
        adv_past = self.storage[self.env.current_seed].get('adv_past')
        self.adv_traj_timestep = int(len(adv_past) + 5 * res_timestep[adv_traj_id])
        adv_pos = np.concatenate((adv_past,adv_future),axis=0)
        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
        adv_vel = get_polyline_vel(adv_pos)
        self.adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))

        return traffic_motion_feat,self.adv_traj,trajs_AV,any(res)
    
    def before_step(self, env, ep_timestep):
        if ep_timestep < self.adv_traj_timestep - self.collision_offset - 1:
            return
        if self.assigned_policy:
            return
        if self.adv_agent in env.engine.get_objects().keys():
            adv_object = env.engine.get_object(self.adv_agent).get(self.adv_agent)
            if self.idm_adv:
                # This is after set_adv_info has been invoked earlier
                if self.adv_agent in env.engine.map_manager.other_routes:
                    adv_route = env.engine.map_manager.other_routes[self.adv_agent]
                    env.engine.traffic_manager.add_policy(self.adv_agent, ScenarioIDMPolicy, adv_object, 0, adv_route, 1)
                    self.assigned_policy = True
            elif self.model_adv:
                self.assigned_policy = True
                env.engine.traffic_manager.add_policy(self.adv_agent, AssignedOtherPolicy, adv_object, self.other_model_policy, env)
            elif self.current_model_adv:
                self.assigned_policy = True
                env.engine.traffic_manager.add_policy(self.adv_agent, AssignedCurrentPolicy, adv_object, self.current_model_policy, env)
            else:
                self.assigned_policy = True
                env.engine.traffic_manager.add_policy(self.adv_agent, AssignedSkillPolicy, adv_object, self.skill_based_policy, env)
        