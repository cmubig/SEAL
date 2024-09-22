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

        self.rule_based = True
        if self.objective_model is not None and self.rule_based:
            raise ValueError("Cannot apply objective_model to rule_based")

        self.idm_adv = args.idm_adv
        self.model_adv = args.model_adv
        self.current_model_adv = args.current_model_adv
        self.current_model_prior = args.current_model_prior

        self.data_cache = {}

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
        self.original_ego_traj = []
        self.ego_prob = 1.
        self.adv_traj = []
        self.original_adv_traj = []
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
            traffic_motion_feat,adv_agent,ego_navigation_route,ego_hist,adv_past,adv_navigation_route,adv_valid = self._parse()
            
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
                adv_valid = adv_valid,
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
        # GOOSE should not update the state deques
        return
    

    def log_AV_history(self,action_prob=1.0):
        # GOOSE should not update the state deques
        return


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
                adv_valid = track_data['state']['valid'][:]

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
        
        return features_description,adv_agent,ego_navigation_route,ego_hist,adv_past,adv_navigation_route,adv_valid

    @property
    def adv_agent(self):
        return self.storage[self.env.current_seed].get('adv_agent')
    
    @property
    def adv_valid(self):
        return self.storage[self.env.current_seed].get('adv_valid')
    
    def generate(self,mode='train'):
        AV_trajs = np.array(self.storage[self.env.current_seed].get('AV_trajs'))[0]
        AV_hist = np.array(self.storage[self.env.current_seed].get('AV_hist'))[0]
        self.original_ego_traj = list(np.concatenate([AV_hist, AV_trajs]))

        trajs_OV = np.array(self.storage[self.env.current_seed].get('adv_navigation_route'))
        adv_past = self.storage[self.env.current_seed].get('adv_past')
        adv_pos = np.concatenate((adv_past,trajs_OV),axis=0)
        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
        adv_vel = get_polyline_vel(adv_pos)
        self.adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))
        self.original_adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))

        return self.adv_traj
    
    
    def before_step(self, env, ep_timestep):
        # Precaution at end of episodes
        if ep_timestep >= len(self.adv_traj):
            return
        # TODO: instead of ep_timesteps, should it be a few steps ahead of our location?
        to_set = [np.array(x) for x in self.adv_traj[ep_timestep:]]
        env.engine.traffic_manager.set_adv_info(self.adv_agent,to_set)	

        if self.adv_agent in env.engine.get_objects().keys():
            adv_object = env.engine.get_object(self.adv_agent).get(self.adv_agent)
            # This is after set_adv_info has been invoked earlier
            if self.adv_agent in env.engine.map_manager.other_routes:
                adv_route = env.engine.map_manager.other_routes[self.adv_agent]
                # Extra parameter to ignore_others
                env.engine.traffic_manager.add_policy(self.adv_agent, ScenarioIDMPolicy, adv_object, 0, adv_route, 1, True)
    
    def frozen_new_episode(self, env):
        import goose_train
        new_seed = env.current_seed
        if new_seed in self.data_cache:
            self.data_cache.pop(env.current_seed)
        original_lane, traj_xy0, traj_h0, curve, bspline = goose_train.get_curve(self, self.adv_traj, goose_train.nurbs_deg, goose_train.nurbs_num_pt)
        curve.delta = 1/91
        curve_weights = np.array(curve.weights)
        ctrlpts = np.array(curve.ctrlpts)
        cur_seed = env.current_seed
        episode_policy_steps = 0
        episode_reward = 0
        last_policy_state = None
        last_replay_buffer_state = None
        reward = 0
        done = 0
        early_done = 0
        ego_crash = False
        action = np.zeros((goose_train.action_dim,))

        self.data_cache[new_seed] = (
            original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts,
            cur_seed, episode_reward, episode_policy_steps,
            last_policy_state, last_replay_buffer_state, action,
            ego_crash, reward, early_done, done
        )

    def frozen_resume_episode(self, env):
        # Generate if needed
        new_seed = env.current_seed
        if new_seed in self.data_cache:
            return
        self.frozen_new_episode(env)
    
    def frozen_set_info(self, env):
        import goose_train
        new_seed = env.current_seed
        assert new_seed in self.data_cache, f'Seed {new_seed} must be in cache already'
        original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts, \
        cur_seed, episode_reward, episode_policy_steps, \
        last_policy_state, last_replay_buffer_state, action, \
        ego_crash, reward, early_done, done = self.data_cache[new_seed]

        episode_policy_steps += 1
        goose_train.safe_reset(env, force_seed=cur_seed)
        self.before_episode(env)
        self.generate()
        curve.weights = curve_weights.tolist()
        curve.ctrlpts = ctrlpts.tolist()
        approx_xy = curve.evalpts

        if goose_train.FRENET:
            adv_pos = goose_train.local_to_traj(original_lane, approx_xy)
        else:
            adv_pos = goose_train.rotate_shift(approx_xy, traj_xy0, traj_h0)

        adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
        adv_vel = get_polyline_vel(adv_pos)
        adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))
        self.adv_traj = adv_traj
        env.engine.traffic_manager.set_adv_info(self.adv_agent, adv_traj)	

        self.data_cache[new_seed] = (
            original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts,
            cur_seed, episode_reward, episode_policy_steps,
            last_policy_state, last_replay_buffer_state, action,
            ego_crash, reward, early_done, done
        )
    
    def frozen_before_episode(self):
        self.episode_max_yaw = 0
        self.episode_max_accel = 0
        self.episode_min_dist = 1000
        self.last_pos_map = {}
        self.last_speed_map = {}
        self.last_yaw_map = {}

        self.sim_states = []
        self.num_timesteps = 0
    
    def frozen_episode_step(self, env):
        import goose_train
        new_seed = env.current_seed
        assert new_seed in self.data_cache, f'Seed {new_seed} must be in cache already'
        original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts, \
        cur_seed, episode_reward, episode_policy_steps, \
        last_policy_state, last_replay_buffer_state, action, \
        ego_crash, reward, early_done, done = self.data_cache[new_seed]
        if goose_train.SELF_IDM and self.adv_agent in env.engine.get_objects().keys():
            adv_object = env.engine.get_object(self.adv_agent).get(self.adv_agent)
            # This is after set_adv_info has been invoked earlier
            if self.adv_agent in env.engine.map_manager.other_routes:
                adv_route = env.engine.map_manager.other_routes[self.adv_agent]
                env.engine.traffic_manager.add_policy(self.adv_agent, ScenarioIDMPolicy, adv_object, 0, adv_route, 1, True)

        state, track_info = goose_train.get_state(env, self, original_lane, traj_xy0, traj_h0, self.last_pos_map, self.last_speed_map, self.last_yaw_map)
        self.sim_states.append(state)
        if state[:goose_train.agent_d][-1]:
            adv_accel, adv_yaw_rate = state[:goose_train.agent_d][5:7]
            self.episode_max_accel = max(np.abs(adv_accel), self.episode_max_accel)
            self.episode_max_yaw = max(np.abs(adv_yaw_rate), self.episode_max_yaw)
            raw_dist = np.linalg.norm((track_info[self.adv_agent][:2] - track_info['default_agent'][:2]))
            self.episode_min_dist = min(self.episode_min_dist, raw_dist)
        for obj_k, obj_v in track_info.items():
            self.last_yaw_map[obj_k] = obj_v[4]
            cur_pos = np.array(obj_v[:2].tolist())
            if goose_train.FRENET:
                cur_pos = goose_train.traj_to_local(original_lane, [cur_pos])[0]
            else:
                pass
            old_pos = None if obj_k not in self.last_pos_map else self.last_pos_map[obj_k]
            self.last_pos_map[obj_k] = cur_pos
            if old_pos is not None:
                cur_speed = np.linalg.norm(cur_pos - old_pos) * 10
                self.last_speed_map[obj_k] = cur_speed
        self.num_timesteps += 1
    
    def frozen_after_episode(self, goose_policy, env, ego_crash, null_action_if_crash=False):
        import goose_train
        new_seed = env.current_seed
        assert new_seed in self.data_cache, f'Seed {new_seed} must be in cache already'
        original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts, \
        cur_seed, episode_reward, episode_policy_steps, \
        last_policy_state, last_replay_buffer_state, action, \
        _, reward, early_done, done = self.data_cache[new_seed]

        adv_traj = np.array([state[:2] for state in self.sim_states])
        adv_heading = np.array([state[4] for state in self.sim_states])
        adv_valid = np.array([state[7] for state in self.sim_states])
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

        achieved_goal = np.array([self.episode_min_dist, max_accel, max_yaw]) 
        if achieved_goal[0] == 1000:
            early_done = True

        states = self.sim_states
        states = np.array(states)[::goose_train.gru_step]
        gru_states = goose_policy.preprocess_feature_sequence(states)
        achieved_goal = torch.FloatTensor(achieved_goal).to(gru_states.device)
        desired_goal = goose_train.get_goal()[0]
        curve_state = np.concatenate([curve_weights[1:], ctrlpts[1:].flatten()])
        desired_goal = torch.FloatTensor(desired_goal).to(gru_states.device)
        curve_state = torch.FloatTensor(curve_state).to(gru_states.device)
        policy_state = torch.cat([gru_states, curve_state, achieved_goal, desired_goal])

        dist_satisfied = ego_crash or achieved_goal[0] < goose_train.distance_epsilon
        if goose_train.require_ineq:
            ineq_satisfied = (achieved_goal[1:] < desired_goal[1:]).all()
            done = ineq_satisfied and dist_satisfied
        else:
            done = dist_satisfied

        penalty = 1 + goose_train.dist_reward_shape * max(0, achieved_goal[0] - desired_goal[0] - goose_train.distance_epsilon) \
                    + goose_train.accel_reward_shape * max(0, achieved_goal[1] - desired_goal[1]) \
                    + goose_train.yaw_reward_shape * max(0, achieved_goal[2] - desired_goal[2])
        reward = 0 if done else -penalty
        episode_reward += reward

        action = goose_policy.select_action(policy_state)
        # TODO: should this be dist_satisfied or just done
        if null_action_if_crash and done:
            action = np.zeros_like(action)

        curve_weights[1:] += action[:4]
        curve_weights = np.clip(curve_weights, goose_train.weight_min, goose_train.weight_max)
        ctrlpts[1:, 0] += action[4:8]
        ctrlpts[1:, 1] += action[8:12]

        self.data_cache[new_seed] = (
            original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts,
            cur_seed, episode_reward, episode_policy_steps,
            last_policy_state, last_replay_buffer_state, action,
            ego_crash, reward, early_done, done
        )
    
    def frozen_should_end(self, env):
        import goose_train
        new_seed = env.current_seed
        assert new_seed in self.data_cache, f'Seed {new_seed} must be in cache already'
        original_lane, traj_xy0, traj_h0, bspline, curve, curve_weights, ctrlpts, \
        cur_seed, episode_reward, episode_policy_steps, \
        last_policy_state, last_replay_buffer_state, action, \
        ego_crash, reward, early_done, done = self.data_cache[new_seed] 
        return done or early_done or episode_policy_steps >= goose_train.episode_max_steps