import logging
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable

import gym
import numpy as np
from panda3d.core import PNMImage

from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_module.vehicle_panel import VehiclePanel
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from metadrive.constants import TerminationState
from metadrive.engine.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, initialize_global_config, get_global_config
from metadrive.manager.agent_manager import AgentManager
from metadrive.manager.record_manager import RecordManager
from metadrive.manager.replay_manager import ReplayManager
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.obs.observation_base import ObservationBase
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.utils import convert_recorded_scenario_exported
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos

BASE_DEFAULT_CONFIG = dict(

    # ===== agent =====
    random_agent_model=False,
    agent_policy=EnvInputPolicy,

    # ===== multi-agent =====
    num_agents=1,  # Note that this can be set to >1 in MARL envs, or set to -1 for as many vehicles as possible.
    is_multi_agent=False,
    allow_respawn=False,
    delay_done=0,  # How many steps for the agent to stay static at the death place after done.
    # Whether only return single agent-like observation and action space
    return_single_space=False,

    # ===== Action =====
    manual_control=False,
    controller="keyboard",  # "joystick" or "keyboard"
    decision_repeat=5,
    discrete_action=False,
    discrete_steering_dim=5,
    discrete_throttle_dim=5,
    # When discrete_action=True: If True, use MultiDiscrete action space. Otherwise, use Discrete.
    use_multi_discrete=False,
    action_check=False,

    # ===== Rendering =====
    use_render=False,  # pop a window to render or not
    debug=False,
    disable_model_compression=False,  # disable compression if you wish to launch the window quicker.
    cull_scene=True,  # only for debug use
    use_chase_camera_follow_lane=False,  # If true, then vision would be more stable.
    camera_height=2.2,
    camera_dist=7.5,
    camera_pitch=None,  # degree
    camera_smooth=True,  # degree
    camera_fov=65,
    prefer_track_agent=None,
    draw_map_resolution=1024,  # Drawing the map in a canvas of (x, x) pixels.
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,  # height
    show_logo=True,

    # ===== Vehicle =====
    vehicle_config=dict(
        increment_steering=False,
        vehicle_model="default",
        show_navi_mark=True,
        extra_action_dim=0,
        enable_reverse=False,
        random_navi_mark_color=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        show_line_to_navi_mark=False,
        use_special_color=False,
        no_wheel_friction=False,

        # ===== use image =====
        image_source="rgb_camera",  # take effect when only when image_observation == True

        # ===== vehicle spawn and destination =====
        navigation_module=None,  # a class type for self-defined navigation
        need_navigation=True,
        spawn_lane_index=None,
        spawn_longitude=5.0,
        spawn_lateral=0.0,
        destination=None,
        spawn_position_heading=None,
        spawn_velocity=None,  # m/s
        spawn_velocity_car_frame=False,

        # ==== others ====
        overtake_stat=False,  # we usually set to True when evaluation
        random_color=False,
        random_agent_model=False,  # this will be overwritten by env.config["random_agent_model"]
        # The shape of vehicle are predefined by its class. But in special scenario (WaymoVehicle) we might want to
        # set to arbitrary shape.
        width=None,
        length=None,
        height=None,
        mass=None,

        # ===== vehicle module config =====
        lidar=dict(
            num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
        ),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        show_lidar=False,
        mini_map=(84, 84, 250),  # buffer length, width
        rgb_camera=(84, 84),  # buffer length, width
        depth_camera=(84, 84, False),  # buffer length, width, view_ground
        main_camera=None,  # buffer length, width
        show_side_detector=False,
        show_lane_line_detector=False,

        # NOTE: rgb_clip will be modified by env level config when initialization
        rgb_clip=True,  # clip 0-255 to 0-1
        stack_size=3,  # the number of timesteps for stacking image observation
        rgb_to_grayscale=False,
        gaussian_noise=0.0,
        dropout_prob=0.0,
        light=False,  # vehicle light, only available when enabling render-pipeline
    ),

    # ===== Agent config =====
    target_vehicle_configs={DEFAULT_AGENT: dict(use_special_color=False, spawn_lane_index=None)},

    # ===== Engine Core config =====
    window_size=(1200, 900),  # or (width, height), if set to None, it will be automatically determined
    physics_world_step_size=2e-2,
    show_fps=True,
    global_light=True,
    # only render physics world without model, a special debug option
    debug_physics_world=False,
    # debug static world
    debug_static_world=False,
    # (Deprecated) set to true only when on headless machine and use rgb image!!!!!!
    # turn on to profile the efficiency
    pstats=False,
    # if need running in offscreen
    image_observation=False,
    # this is an advanced feature for accessing image with moving them to ram!
    image_on_cuda=False,
    # accelerate the lidar perception
    _disable_detector_mask=False,
    # clip rgb to (0, 1)
    rgb_clip=True,
    # None: unlimited, number: fps
    force_render_fps=None,
    # if set to True all objects will be force destroyed when call clear()
    force_destroy=False,
    # number of buffering objects for each class.
    # we will maintain a set of buffers in the engine to store the used objects and can reuse them
    # when possible. But it is possible that some classes of objects are always forcefully respawn
    # and thus those used objects are stored in the buffer and never be reused.
    num_buffering_objects=200,
    # turn on to use render pipeline, which provides advanced rendering effects (Beta)
    render_pipeline=False,
    # daytime is only available when using render-pipeline
    daytime="19:00",  # use string like "13:40", We usually set this by editor in toolkit

    # ===== Mesh Terrain =====
    # road will have a marin whose width is determined by this value, unit: [m]
    drivable_region_extension=6,
    # height scale for mountains, unit: [m]
    height_scale=120,

    # ===== Others =====
    # The maximum distance used in PGLOD. Set to None will use the default values.
    max_distance=None,
    # Force to generate objects in the left lane.
    _debug_crash_object=False,
    horizon=None,  # The maximum length of each environmental episode. Set to None to remove this constraint
    max_step_per_agent=None,  # The maximum length of each agent episode. Raise max_step termination when reaches.
    show_interface_navi_mark=True,
    show_mouse=True,
    show_skybox=True,
    show_terrain=True,
    show_interface=True,
    show_policy_mark=False,  # show marks for policies for debugging multi-policy setting
    show_coordinates=False,  # show coordinates for maps and objects for debug
    interface_panel=[MiniMap, RGBCamera, VehiclePanel],
    multi_thread_render=True,
    multi_thread_render_mode="Cull",  # or "Cull/Draw"
    preload_models=True,  # preload pedestrian Object for avoiding lagging when creating it for the first time

    # record/replay metadata
    record_episode=False,  # when replay_episode is not None ,this option will be useless
    replay_episode=None,  # set the replay file to enable replay
    only_reset_when_replay=False,  # Scenario will only be initialized, while future trajectories will not be replayed
    force_reuse_object_name=False,  # If True, when restoring objects, use the same ID as in dataset
)


class BaseEnv(gym.Env):
    # Force to use this seed if necessary. Note that the recipient of the forced seed should be explicitly implemented.
    _DEBUG_RANDOM_SEED = None
    DEFAULT_AGENT = DEFAULT_AGENT

    @classmethod
    def default_config(cls) -> "Config":
        return Config(BASE_DEFAULT_CONFIG)

    # ===== Intialization =====
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        merged_config = self._merge_extra_config(config)
        global_config = self._post_process_config(merged_config)
        global_config["vehicle_config"]["main_camera"] = global_config["window_size"]

        self.config = global_config
        initialize_global_config(self.config)
        self.logger = logging.getLogger(self.logger_name)

        # agent check
        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        else:
            assert not self.config["image_on_cuda"], "Image on cuda don't support Multi-agent!"
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1)

        # observation and action space
        self.agent_manager = self._get_agent_manager()

        # lazy initialization, create the main vehicle in the lazy_init() func
        # self.engine: Optional[BaseEngine] = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.dones = None
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

    def _merge_extra_config(self, config: Union[dict, "Config"]) -> "Config":
        """Check, update, sync and overwrite some config."""
        return config

    def _post_process_config(self, config):
        """Add more special process to merged config"""
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        config["vehicle_config"]["rgb_clip"] = config["rgb_clip"]
        return config

    def _get_observations(self) -> Dict[str, "ObservationBase"]:
        raise NotImplementedError()

    def _get_observation_space(self):
        return {v_id: obs.observation_space for v_id, obs in self.observations.items()}

    def _get_action_space(self):
        if self.is_multi_agent:
            return {
                v_id: self.config["agent_policy"].get_input_space()
                for v_id in self.config["target_vehicle_configs"].keys()
            }
        else:
            return {DEFAULT_AGENT: self.config["agent_policy"].get_input_space()}

    def _get_agent_manager(self):
        return AgentManager(init_observations=self._get_observations(), init_action_space=self._get_action_space())

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        if engine_initialized():
            return
        engine = initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
        self._after_lazy_init()

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def _after_lazy_init(self):
        pass

    # ===== Run-time =====
    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]):
        actions = self._preprocess_actions(actions)
        engine_info = self._step_simulator(actions)
        o, r, d, i = self._get_step_return(actions, engine_info=engine_info)
        return o, r, d, i

    def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
        if not self.is_multi_agent:
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        else:
            if self.config["action_check"]:
                # Check whether some actions are not provided.
                given_keys = set(actions.keys())
                have_keys = set(self.vehicles.keys())
                assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
                    given_keys, have_keys
                )
            else:
                # That would be OK if extra actions is given. This is because, when evaluate a policy with naive
                # implementation, the "termination observation" will still be given in T=t-1. And at T=t, when you
                # collect action from policy(last_obs) without masking, then the action for "termination observation"
                # will still be computed. We just filter it out here.
                actions = {v_id: actions[v_id] for v_id in self.vehicles.keys()}
        return actions

    def _step_simulator(self, actions):
        # Note that we use shallow update for info dict in this function! This will accelerate system.
        scene_manager_before_step_infos = self.engine.before_step(actions)
        # step all entities
        self.engine.step(self.config["decision_repeat"])
        # update states, if restore from episode data, position and heading will be force set in update_state() function

        scene_manager_after_step_infos = self.engine.after_step(crash_vehicle=self.vehicle.crash_vehicle, env=self)

        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

    def reward_function(self, vehicle_id: str) -> Tuple[float, Dict]:
        """
        Override this func to get a new reward function
        :param vehicle_id: name of this base vehicle
        :return: reward, reward info
        """
        raise NotImplementedError()

    def cost_function(self, vehicle_id: str) -> Tuple[float, Dict]:
        raise NotImplementedError()

    def done_function(self, vehicle_id: str) -> Tuple[bool, Dict]:
        raise NotImplementedError()

    def render(self,
               mode='human',
               text: Optional[Union[dict, str]] = None,
               return_bytes=False,
               *args,
               **kwargs) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param mode: 'rgb'/'human'
        :param text:text to show
        :return: when mode is 'rgb', image array is returned
        """
        if mode in ["top_down", "topdown", "bev", "birdview"]:
            ret = self._render_topdown(text=text, *args, **kwargs)
            return ret
        assert self.config["use_render"] or self.engine.mode != RENDER_MODE_NONE, \
            ("Panda Renderring is off now, can not render. Please set config['use_render'] = True!")

        self.engine.render_frame(text)

        if mode != "human" and self.config["image_observation"]:
            # fetch img from img stack to be make this func compatible with other render func in RL setting
            return self.vehicle.observations.img_obs.get_image()

        if mode == "rgb_array":
            assert self.config["use_render"], "You should create a Panda3d window before rendering images!"
            # if not hasattr(self, "temporary_img_obs"):
            #     from metadrive.obs.image_obs import ImageObservation
            #     image_source = "rgb_camera"
            #     assert len(self.vehicles) == 1, "Multi-agent not supported yet!"
            #     self.temporary_img_obs = ImageObservation(self.vehicles[DEFAULT_AGENT].config, image_source, False)
            # # else:
            # #     raise ValueError("Not implemented yet!")
            # self.temporary_img_obs.observe(self.vehicles[DEFAULT_AGENT])
            # return self.temporary_img_obs.get_image()
            return self.engine.get_window_image(return_bytes=return_bytes)

        # logging.warning("You do not set 'image_observation' or 'image_observation' to True, so no image will be returned!")
        return None

    def reset(self, force_seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param force_seed: The seed to set the env.
        :return: None
        """
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self._reset_global_seed(force_seed)
        if self.engine is None:
            raise ValueError(
                "Current MetaDrive instance is broken. Please make sure there is only one active MetaDrive "
                "environment exists in one process. You can try to call env.close() and then call "
                "env.reset() to rescue this environment. However, a better and safer solution is to check the "
                "singleton of MetaDrive and restart your program."
            )
        self.engine.reset()
        if self._top_down_renderer is not None:
            self._top_down_renderer.reset(self.current_map)

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        assert (len(self.vehicles) == self.num_agents) or (self.num_agents == -1)
        assert self.config is self.engine.global_config is get_global_config(), "Inconsistent config may bring errors!"
        return self._get_reset_return()

    def _get_reset_return(self):
        ret = {}
        self.engine.after_step()
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
        return ret if self.is_multi_agent else self._wrap_as_single_agent(ret)

    def _get_step_return(self, actions, engine_info):
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.vehicles.items():
            rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            done_function_result, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            done = done_function_result or self.dones[v_id]
            self.dones[v_id] = done
            o = self.observations[v_id].observe(v)
            obses[v_id] = o

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])


        # For extreme scenario only. Force to terminate all vehicles if the environmental step exceeds 5 times horizon.
        should_external_done = False
        if self.config["horizon"] is not None:
            should_external_done = self.episode_step > 5 * self.config["horizon"]
        if should_external_done:
            for k in self.dones:
                self.dones[k] = True

        dones = {k: self.dones[k] for k in self.vehicles.keys()}
        for v_id, r in rewards.items():
            self.episode_rewards[v_id] += r
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            self.episode_lengths[v_id] += 1
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]

        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                   self._wrap_as_single_agent(dones), self._wrap_as_single_agent(step_infos)
        else:
            return obses, rewards, dones, step_infos

    def close(self):
        if self.engine is not None:
            close_engine()

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # Sleep two seconds
        raise KeyboardInterrupt("'Esc' is pressed. MetaDrive exits now.")

    def capture(self, file_name=None):
        if not hasattr(self, "_capture_img"):
            self._capture_img = PNMImage()
        self.engine.win.getScreenshot(self._capture_img)
        if file_name is None:
            file_name = "main_{}.png".format(time.time())
        self._capture_img.write(file_name)

    def for_each_vehicle(self, func, *args, **kwargs):
        return self.agent_manager.for_each_active_agents(func, *args, **kwargs)

    @property
    def vehicle(self):
        """A helper to return the vehicle only in the single-agent environment!"""
        assert len(self.vehicles) == 1, (
            "env.vehicle is only supported in single-agent environment!"
            if len(self.vehicles) > 1 else "Please initialize the environment first!"
        )
        ego_v = self.vehicles[DEFAULT_AGENT]
        return ego_v

    def get_single_observation(self, vehicle_config: "Config"):
        if self.config["image_observation"]:
            o = ImageStateObservation(vehicle_config)
        else:
            o = LidarStateObservation(vehicle_config)
        return o

    def _wrap_as_single_agent(self, data):
        return data[next(iter(self.vehicles.keys()))]

    def seed(self, seed=None):
        if seed is not None:
            set_global_random_seed(seed)

    @property
    def current_seed(self):
        return self.engine.global_random_seed

    @property
    def observations(self):
        """
        Return observations of active and controllable vehicles
        :return: Dict
        """
        return self.agent_manager.get_observations()

    @property
    def observation_space(self) -> gym.Space:
        """
        Return observation spaces of active and controllable vehicles
        :return: Dict
        """
        ret = self.agent_manager.get_observation_spaces()
        if (not self.is_multi_agent) or self.config["return_single_space"]:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def action_space(self) -> gym.Space:
        """
        Return observation spaces of active and controllable vehicles
        :return: Dict
        """
        ret = self.agent_manager.get_action_spaces()
        if (not self.is_multi_agent) or self.config["return_single_space"]:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def vehicles(self):
        """
        Return all active vehicles
        :return: Dict[agent_id:vehicle]
        """
        return self.agent_manager.active_agents

    @property
    def vehicles_including_just_terminated(self):
        """
        Return all vehicles that occupy some space in current environments
        :return: Dict[agent_id:vehicle]
        """
        ret = self.agent_manager.active_agents
        ret.update(self.agent_manager.just_terminated_agents)
        return ret

    def setup_engine(self):
        """
        Engine setting after launching
        """
        self.engine.accept("r", self.reset)
        # self.engine.accept("c", self.capture)
        self.engine.register_manager("agent_manager", self.agent_manager)
        self.engine.register_manager("record_manager", RecordManager())
        self.engine.register_manager("replay_manager", ReplayManager())

    @property
    def current_map(self):
        return self.engine.current_map

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else get_np_random(None).randint(0, int(1e4))
        self.seed(current_seed)

    @property
    def maps(self):
        return self.engine.map_manager.maps

    def _render_topdown(self, text, *args, **kwargs):
        return self.engine.render_topdown(text, *args, **kwargs)

    @property
    def main_camera(self):
        return self.engine.main_camera

    @property
    def current_track_vehicle(self):
        return self.engine.current_track_vehicle

    @property
    def _top_down_renderer(self):
        return self.engine._top_down_renderer

    @property
    def episode_step(self):
        return self.engine.episode_step if self.engine is not None else 0

    def export_scenarios(
        self,
        policies: Union[dict, Callable],
        scenario_index: Union[list, int],
        max_episode_length=None,
        verbose=False,
        suppress_warning=False,
        render_topdown=False,
        return_done_info=True,
        to_dict=True
    ):
        """
        We export scenarios into a unified format with 10hz sample rate
        """
        def _act(observation):
            if isinstance(policies, dict):
                ret = {}
                for id, o in observation.items():
                    ret[id] = policies[id](o)
            else:
                ret = policies(observation)
            return ret

        if self.is_multi_agent:
            assert isinstance(policies, dict), "In MARL setting, policies should be mapped to agents according to id"
        else:
            assert isinstance(policies, Callable), "In single agent case, policy should be a callable object, taking" \
                                                   "observation as input."
        scenarios_to_export = dict()
        if isinstance(scenario_index, int):
            scenario_index = [scenario_index]
        self.config["record_episode"] = True
        done_info = {}
        for index in scenario_index:
            obs = self.reset(force_seed=index)
            done = False
            count = 0
            info = None
            while not done:
                obs, reward, done, info = self.step(_act(obs))
                count += 1
                if max_episode_length is not None and count > max_episode_length:
                    done = True
                    info[TerminationState.MAX_STEP] = True
                if count > 10000 and not suppress_warning:
                    logging.warning(
                        "Episode length is too long! If this behavior is intended, "
                        "set suppress_warning=True to disable this message"
                    )
                if render_topdown:
                    self.render("topdown")
            episode = self.engine.dump_episode()
            if verbose:
                logging.info("Finish scenario {} with {} steps.".format(index, count))
            scenarios_to_export[index] = convert_recorded_scenario_exported(episode, to_dict=to_dict)
            done_info[index] = info
        self.config["record_episode"] = False
        if return_done_info:
            return scenarios_to_export, done_info
        else:
            return scenarios_to_export

    def export_single_scenario(self):
        """
        Similar export_scenarios, this function transform the internal recorded frames to a standard
        scenario description.
        """
        episode = self.engine.dump_episode()
        return convert_recorded_scenario_exported(episode)

    @property
    def logger_name(self):
        return __file__
