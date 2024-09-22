import math
import os
from collections import deque
from typing import Union, Optional

import numpy as np
import seaborn as sns
from panda3d._rplight import RPSpotLight
from panda3d.bullet import BulletVehicle, BulletBoxShape, ZUp
from panda3d.core import Material, Vec3, TransformState
from panda3d.core import NodePath

from metadrive.base_class.base_object import BaseObject
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.point_lane import PointLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.component.vehicle_module.depth_camera import DepthCamera
from metadrive.component.vehicle_module.distance_detector import SideDetector, LaneLineDetector
from metadrive.component.vehicle_module.lidar import Lidar
from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.component.vehicle_navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.constants import MetaDriveType, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.engine.engine_utils import get_engine, engine_initialized
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils import Config, safe_clip_for_small_array
from metadrive.utils.math import get_vertical_vector, norm, clip
from metadrive.utils.math import wrap_to_pi
from metadrive.utils.pg.utils import ray_localization
from metadrive.utils.pg.utils import rect_region_detection
from metadrive.utils.utils import get_object_from_node


class BaseVehicleState:
    def __init__(self):
        self.init_state_info()

    def init_state_info(self):
        """
        Call this before reset()/step()
        """
        self.crash_vehicle = False
        self.crash_human = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False

        # traffic light
        self.red_light = False
        self.yellow_light = False
        self.green_light = False

        # lane line detection
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False

        # contact results, a set containing objects type name for rendering
        self.contact_results = set()


class BaseVehicle(BaseObject, BaseVehicleState):
    """
    Vehicle chassis and its wheels index
                    0       1
                    II-----II
                        |
                        |  <---chassis/wheelbase
                        |
                    II-----II
                    2       3
    """
    COLLISION_MASK = CollisionGroup.Vehicle
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.BASE_VEHICLE)
    MAX_LENGTH = 10
    MAX_WIDTH = 2.5
    MAX_STEERING = 60

    # LENGTH = None
    # WIDTH = None
    # HEIGHT = None

    TIRE_RADIUS = None
    TIRE_MODEL_CORRECT = 1  # correct model left right error
    TIRE_TWO_SIDED = False  # tires for some vehicles need two-sided enabled for correctly visualization
    LATERAL_TIRE_TO_CENTER = None
    TIRE_WIDTH = 0.4
    FRONT_WHEELBASE = None
    REAR_WHEELBASE = None
    LIGHT_POSITION = (-0.67, 1.62, 0.05)

    # MASS = None

    CHASSIS_TO_WHEEL_AXIS = 0.2
    SUSPENSION_LENGTH = 15
    SUSPENSION_STIFFNESS = 40

    # for random color choosing
    MATERIAL_COLOR_COEFF = 2  # to resist other factors, since other setting may make color dark
    MATERIAL_METAL_COEFF = 0.1  # 0-1
    MATERIAL_ROUGHNESS = 0.8  # smaller to make it more smooth, and reflect more light
    MATERIAL_SHININESS = 128  # 0-128 smaller to make it more smooth, and reflect more light
    MATERIAL_SPECULAR_COLOR = (3, 3, 3, 3)

    # control
    STEERING_INCREMENT = 0.05

    # save memory, load model once
    model_collection = {}
    path = None

    def __init__(
        self,
        vehicle_config: Union[dict, Config] = None,
        name: str = None,
        random_seed=None,
        position=None,
        heading=None
    ):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters. And self.physics_config defines the physics feature of vehicles, such as length/width
        :param vehicle_config: mostly, vehicle module config
        :param random_seed: int
        """
        # check
        assert vehicle_config is not None, "Please specify the vehicle config."
        assert engine_initialized(), "Please make sure game engine is successfully initialized!"

        # NOTE: it is the game engine, not vehicle drivetrain
        # self.engine = get_engine()
        BaseObject.__init__(self, name, random_seed, self.engine.global_config["vehicle_config"])
        BaseVehicleState.__init__(self)
        self.update_config(vehicle_config)
        use_special_color = self.config["use_special_color"]

        # build vehicle physics model
        vehicle_chassis = self._create_vehicle_chassis()
        self.add_body(vehicle_chassis.getChassis())
        self.system = vehicle_chassis
        self.chassis = self.origin
        self.wheels = self._create_wheel()

        # light experimental!
        self.light = None
        self._light_direction_queue = None
        self._light_models = None
        self.light_name = None

        # powertrain config
        self.increment_steering = self.config["increment_steering"]
        self.enable_reverse = self.config["enable_reverse"]
        self.max_steering = self.config["max_steering"]

        # visualization
        self._use_special_color = use_special_color
        self._add_visualization()

        # modules, get observation by using these modules
        self.navigation: Optional[NodeNetworkNavigation] = None
        self.lidar: Optional[Lidar] = None  # detect surrounding vehicles
        self.side_detector: Optional[SideDetector] = None  # detect road side
        self.lane_line_detector: Optional[LaneLineDetector] = None  # detect nearest lane lines
        self.image_sensors = {}

        # state info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = (0, 0)
        self.last_heading_dir = self.heading
        self.dist_to_left_side = None
        self.dist_to_right_side = None
        self.last_velocity = 0
        self.last_speed = 0

        # step info
        self.out_of_route = None
        self.on_lane = None
        self.spawn_place = (0, 0)
        self._init_step_info()

        # others
        self._add_modules_for_vehicle()
        self.takeover = False
        self.expert_takeover = False
        self.energy_consumption = 0
        self.break_down = False

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

        if self.engine.current_map is not None:
            self.reset(position=position, heading=heading)

    def _add_modules_for_vehicle(self, ):
        """
        This function is related to the self.update_config, which will create modules if needed for resetting a new
        vehicle
        """
        config = self.config

        # add routing module
        self.add_navigation()  # default added

        # add distance detector/lidar
        self.side_detector = SideDetector(
            config["side_detector"]["num_lasers"], config["side_detector"]["distance"],
            self.engine.global_config["vehicle_config"]["show_side_detector"]
        )

        self.lane_line_detector = LaneLineDetector(
            config["lane_line_detector"]["num_lasers"], config["lane_line_detector"]["distance"],
            self.engine.global_config["vehicle_config"]["show_lane_line_detector"]
        )

        self.lidar = Lidar(
            config["lidar"]["num_lasers"], config["lidar"]["distance"],
            self.engine.global_config["vehicle_config"]["show_lidar"]
        )

        # vision modules
        self.setup_sensors()

    def _available_sensors(self):
        def _main_cam():
            assert self.engine.main_camera is not None, "Main camera doesn't exist"
            return self.engine.main_camera

        return {"rgb_camera": RGBCamera, "mini_map": MiniMap, "depth_camera": DepthCamera, "main_camera": _main_cam}

    def setup_sensors(self):
        if self.engine.global_config["image_observation"]:
            sensors = self._available_sensors()
            self.add_image_sensor(self.config["image_source"], sensors[self.config["image_source"]]())

    def get_camera(self, camera_type_str):
        sensors = self._available_sensors()
        if camera_type_str not in sensors:
            raise ValueError("Can not get {}, available type: {}".format(camera_type_str, sensors.keys()))
        if camera_type_str not in self.image_sensors:
            self.add_image_sensor(camera_type_str, sensors[camera_type_str]())
        return self.image_sensors[camera_type_str]

    def _add_modules_for_vehicle_when_reset(self):
        config = self.config

        # add routing module
        if self.navigation is None or self.name != 'default_agent':
            self.add_navigation()  # default added

        # add distance detector/lidar
        if self.side_detector is None:
            self.side_detector = SideDetector(
                config["side_detector"]["num_lasers"], config["side_detector"]["distance"],
                self.engine.global_config["vehicle_config"]["show_side_detector"]
            )

        if self.lane_line_detector is None:
            self.lane_line_detector = LaneLineDetector(
                config["lane_line_detector"]["num_lasers"], config["lane_line_detector"]["distance"],
                self.engine.global_config["vehicle_config"]["show_lane_line_detector"]
            )

        if self.lidar is None:
            self.lidar = Lidar(
                config["lidar"]["num_lasers"], config["lidar"]["distance"],
                self.engine.global_config["vehicle_config"]["show_lidar"]
            )

        # vision modules
        # self.add_image_sensor("rgb_camera", RGBCamera())
        # self.add_image_sensor("mini_map", MiniMap())
        # self.add_image_sensor("depth_camera", DepthCamera())

    def _init_step_info(self):
        # done info will be initialized every frame
        self.init_state_info()
        self.out_of_route = False  # re-route is required if is false
        self.on_lane = True  # on lane surface or not

    @staticmethod
    def _preprocess_action(action):
        if action is None:
            return None, {"raw_action": None}
        action = safe_clip_for_small_array(action, -1, 1)
        return action, {'raw_action': (action[0], action[1])}

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        # if action is None:
        #     action = [0, 0]

        # This *DOES* trigger for all agents

        self._init_step_info()
        action, step_info = self._preprocess_action(action)

        self.last_position = self.position  # 2D vector
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar
        self.last_heading_dir = self.heading
        if action is not None:
            self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
        if self.increment_steering:
            self._set_incremental_action(action)
        else:
            self._set_action(action)
        return step_info

    def after_step(self):
        if self.navigation is not None:
            self.navigation.update_localization(self)
        self._state_check()
        self.update_dist_to_left_right()
        step_energy, episode_energy = self._update_energy_consumption()
        self.out_of_route = self._out_of_route()
        step_info = self._update_overtake_stat()
        my_policy = self.engine.get_policy(self.name)
        step_info.update(
            {
                "velocity": float(self.speed),
                "steering": float(self.steering),
                "acceleration": float(self.throttle_brake),
                "step_energy": step_energy,
                "episode_energy": episode_energy,
                "policy": my_policy.name if my_policy is not None else my_policy
            }
        )
        return step_info

    def _out_of_route(self):
        left, right = self._dist_to_route_left_right()
        return True if right < 0 or left < 0 else False

    def _update_energy_consumption(self):
        """
        The calculation method is from
        https://www.researchgate.net/publication/262182035_Reduction_of_Fuel_Consumption_and_Exhaust_Pollutant_Using_Intelligent_Transport_System
        default: 3rd gear, try to use ae^bx to fit it, dp: (90, 8), (130, 12)
        :return: None
        """
        distance = norm(self.last_position[0] - self.position[0], self.last_position[1] - self.position[1]) / 1000  # km
        step_energy = 3.25 * np.exp(0.01 * self.speed_km_h) * distance / 100
        # step_energy is in Liter, we return mL
        step_energy = step_energy * 1000
        self.energy_consumption += step_energy  # L/100 km
        return step_energy, self.energy_consumption

    def reset(
        self,
        vehicle_config=None,
        name=None,
        random_seed=None,
        position: np.ndarray = None,
        heading: float = 0.0,
        *args,
        **kwargs
    ):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to spawn place
        """
        if name is not None:
            self.rename(name)

        if random_seed is not None:
            assert isinstance(random_seed, int)
            self.seed(random_seed)
            self.sample_parameters()
        if vehicle_config is not None:
            self.update_config(vehicle_config)

        # Update some modules that might not be initialized before
        self._add_modules_for_vehicle_when_reset()

        map = self.engine.current_map
        self.set_pitch(0)
        self.set_roll(0)
        if position is not None:
            # Highest priority
            pass
        elif self.config["spawn_position_heading"] is None:
            # spawn_lane_index has second priority
            lane = map.road_network.get_lane(self.config["spawn_lane_index"])
            position = lane.position(self.config["spawn_longitude"], self.config["spawn_lateral"])
            heading = lane.heading_theta_at(self.config["spawn_longitude"])
        else:
            assert self.config["spawn_position_heading"] is not None, "At least setting one initialization method"
            position = self.config["spawn_position_heading"][0]
            heading = self.config["spawn_position_heading"][1]

        self.spawn_place = position
        self.set_heading_theta(heading)
        self.set_static(False)
        # self.set_wheel_friction(self.config["wheel_friction"])

        if len(position) == 2:
            self.set_position(position, height=self.HEIGHT / 2)
        elif len(position) == 3:
            self.set_position(position[:2], height=position[-1])
        else:
            raise ValueError()

        self.update_map_info(map)
        self.body.clearForces()
        self.body.setLinearVelocity(Vec3(0, 0, 0))
        self.body.setAngularVelocity(Vec3(0, 0, 0))
        self.system.resetSuspension()
        self._apply_throttle_brake(0.0)
        # np.testing.assert_almost_equal(self.position, pos, decimal=4)

        # done info
        self._init_step_info()

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.spawn_place
        self.last_heading_dir = self.heading
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar

        self.update_dist_to_left_right()
        self.takeover = False
        self.energy_consumption = 0

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()
        self.expert_takeover = False
        if self.config["need_navigation"]:
            assert self.navigation

        if self.config["spawn_velocity"] is not None:
            self.set_velocity(self.config["spawn_velocity"], in_local_frame=self.config["spawn_velocity_car_frame"])

        # clean lights
        if self.config["light"]:
            self.add_light()
        else:
            self.remove_light()

        # self.add_light()

    """------------------------------------------- act -------------------------------------------------"""

    def remove_light(self):
        if self.light is not None:
            if self.use_render_pipeline:
                self.engine.render_pipeline.remove_light(self.light)
                self.engine.taskMgr.remove(self.light_name)
            self.light_name = None
            self.light = None
            for m in self._light_models:
                m.removeNode()
            self._light_models = None
            self._light_direction_queue = None

    def add_light(self):
        """
        Experimental feature
        """
        # assert self.use_render_pipeline, "Can be Enabled when using render pipeline"
        if self.light is None:
            self._light_models = []
            for y in [-1, 1]:
                light_model = self.loader.loadModel(AssetLoader.file_path("models", "sphere.egg"))
                light_model.reparentTo(self.origin)
                light_model.setPos(self.LIGHT_POSITION[0] * y, self.LIGHT_POSITION[1], self.LIGHT_POSITION[2])
                light_model.setScale(0.13, 0.05, 0.13)
                material = Material()
                material.setBaseColor((1, 1, 1, 1))
                material.setShininess(128)
                material.setEmission((1, 1, 1, 1))
                light_model.setMaterial(material, True)
                self._light_models.append(light_model)
            if self.use_render_pipeline:
                self.light_name = "light_{}".format(self.id)
                self.light = RPSpotLight()
                self.light.set_color_from_temperature(3 * 1000.0)
                self.light.setRadius(500)
                self.light.setFov(100)
                self.light.energy = 600
                self.light.casts_shadows = False
                self.light.shadow_map_resolution = 128
                self.engine.render_pipeline.add_light(self.light)
                self.engine.taskMgr.add(self._update_light_pos, self.light_name)
                self._light_direction_queue = []

    def _update_light_pos(self, task):
        pos = self.convert_to_world_coordinates([self.LENGTH / 2, 0], self.position)
        self.light.set_pos(*pos, self.get_z())
        self._light_direction_queue.append([*self.heading, 0])
        idx = min(len(self._light_direction_queue), 50)
        pos = np.mean(self._light_direction_queue[-idx:], axis=0)
        self.light.set_direction(pos[0], pos[1], pos[2])
        return task.cont

    def set_steering(self, steering):
        steering = float(steering)
        self.system.setSteeringValue(steering, 0)
        self.system.setSteeringValue(steering, 1)
        self.steering = steering

    def set_throttle_brake(self, throttle_brake):
        throttle_brake = float(throttle_brake)
        self._apply_throttle_brake(throttle_brake)
        self.throttle_brake = throttle_brake

    def _set_action(self, action):
        if action is None:
            return
        steering = action[0]
        self.throttle_brake = action[1]
        self.steering = steering
        self.system.setSteeringValue(self.steering * self.max_steering, 0)
        self.system.setSteeringValue(self.steering * self.max_steering, 1)
        self._apply_throttle_brake(action[1])

    def _set_incremental_action(self, action: np.ndarray):
        if action is None:
            return
        self.throttle_brake = action[1]
        self.steering += action[0] * self.STEERING_INCREMENT
        self.steering = clip(self.steering, -1, 1)
        steering = self.steering * self.max_steering
        self.system.setSteeringValue(steering, 0)
        self.system.setSteeringValue(steering, 1)
        self._apply_throttle_brake(action[1])

    def _apply_throttle_brake(self, throttle_brake):
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]
        for wheel_index in range(4):
            if throttle_brake >= 0:
                self.system.setBrake(2.0, wheel_index)
                if self.speed_km_h > self.max_speed_km_h:
                    self.system.applyEngineForce(0.0, wheel_index)
                else:
                    self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
            else:
                if self.enable_reverse:
                    self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
                    self.system.setBrake(0, wheel_index)
                else:
                    self.system.applyEngineForce(0.0, wheel_index)
                    self.system.setBrake(abs(throttle_brake) * max_brake_force, wheel_index)

    """---------------------------------------- vehicle info ----------------------------------------------"""

    def update_dist_to_left_right(self):
        self.dist_to_left_side, self.dist_to_right_side = self._dist_to_route_left_right()

    def _dist_to_route_left_right(self):
        # TODO
        if self.navigation is None or self.navigation.current_ref_lanes is None:
            return 0, 0
        current_reference_lane = self.navigation.current_ref_lanes[0]
        _, lateral_to_reference = current_reference_lane.local_coordinates(self.position)
        lateral_to_left = lateral_to_reference + self.navigation.get_current_lane_width() / 2
        lateral_to_right = self.navigation.get_current_lateral_range(self.position, self.engine) - lateral_to_left
        return lateral_to_left, lateral_to_right

    # @property
    # def heading_theta(self):
    #     """
    #     Get the heading theta of vehicle, unit [rad]
    #     :return:  heading in rad
    #     """
    #     return wrap_to_pi(self.origin.getH() / 180 * math.pi)

    # @property
    # def velocity(self) -> np.ndarray:
    #     return self.speed * self.velocity_direction
    #
    # @property
    # def velocity_km_h(self) -> np.ndarray:
    #     return self.speed * self.velocity_direction * 3.6

    @property
    def chassis_velocity_direction(self):
        raise DeprecationWarning(
            "This API returns the direction of velocity which is approximately heading direction. "
            "Deprecate it and make things easy"
        )
        direction = self.system.getForwardVector()
        return np.asarray([direction[0], direction[1]])

    """---------------------------------------- some math tool ----------------------------------------------"""

    def heading_diff(self, target_lane):
        lateral = None
        if isinstance(target_lane, StraightLane):
            lateral = np.asarray(get_vertical_vector(target_lane.end - target_lane.start)[1])
        elif isinstance(target_lane, CircularLane):
            if not target_lane.is_clockwise():
                lateral = self.position - target_lane.center
            else:
                lateral = target_lane.center - self.position
        elif isinstance(target_lane, PointLane):
            lateral = target_lane.lateral_direction(target_lane.local_coordinates(self.position)[0])

        lateral_norm = norm(lateral[0], lateral[1])
        forward_direction = self.heading
        # print(f"Old forward direction: {self.forward_direction}, new heading {self.heading}")
        forward_direction_norm = norm(forward_direction[0], forward_direction[1])
        if not lateral_norm * forward_direction_norm:
            return 0
        cos = (
            (forward_direction[0] * lateral[0] + forward_direction[1] * lateral[1]) /
            (lateral_norm * forward_direction_norm)
        )
        # return cos
        # Normalize to 0, 1
        return clip(cos, -1.0, 1.0) / 2 + 0.5

    def lane_distance_to(self, vehicle, lane: AbstractLane = None) -> float:
        assert self.navigation is not None, "a routing and localization module should be added " \
                                            "to interact with other vehicles"
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    """-------------------------------------- for vehicle making ------------------------------------------"""

    @property
    def LENGTH(self):
        raise NotImplementedError()

    @property
    def HEIGHT(self):
        raise NotImplementedError()

    @property
    def WIDTH(self):
        raise NotImplementedError()

    def _create_vehicle_chassis(self):
        # self.LENGTH = type(self).LENGTH
        # self.WIDTH = type(self).WIDTH
        # self.HEIGHT = type(self).HEIGHT

        # assert self.LENGTH < BaseVehicle.MAX_LENGTH, "Vehicle is too large!"
        # assert self.WIDTH < BaseVehicle.MAX_WIDTH, "Vehicle is too large!"

        chassis = BaseRigidBodyNode(self.name, MetaDriveType.VEHICLE)
        self._node_path_list.append(chassis)

        chassis_shape = BulletBoxShape(Vec3(self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        ts = TransformState.makePos(Vec3(0, 0, self.HEIGHT / 2))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        physics_world = get_engine().physics_world
        vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        vehicle_chassis.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(vehicle_chassis)
        return vehicle_chassis

    def _add_visualization(self):
        if self.render:
            [path, scale, offset, HPR] = self.path
            if path not in BaseVehicle.model_collection:
                car_model = self.loader.loadModel(AssetLoader.file_path("models", path))
                car_model.setTwoSided(False)
                BaseVehicle.model_collection[path] = car_model
                car_model.setScale(scale)
                # model default, face to y
                car_model.setHpr(*HPR)
                car_model.setPos(offset[0], offset[1], offset[-1])
                car_model.setZ(-self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS + offset[-1])
            else:
                car_model = BaseVehicle.model_collection[path]
            car_model.instanceTo(self.origin)
            if self.config["random_color"]:
                material = Material()
                material.setBaseColor(
                    (
                        self.panda_color[0] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.2
                    )
                )
                material.setMetallic(self.MATERIAL_METAL_COEFF)
                material.setSpecular(self.MATERIAL_SPECULAR_COLOR)
                material.setRefractiveIndex(1.5)
                material.setRoughness(self.MATERIAL_ROUGHNESS)
                material.setShininess(self.MATERIAL_SHININESS)
                material.setTwoside(False)
                self.origin.setMaterial(material, True)

    def _create_wheel(self):
        f_l = self.FRONT_WHEELBASE
        r_l = -self.REAR_WHEELBASE
        lateral = self.LATERAL_TIRE_TO_CENTER
        axis_height = self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS
        radius = self.TIRE_RADIUS
        wheels = []
        for k, pos in enumerate([Vec3(lateral, f_l, axis_height), Vec3(-lateral, f_l, axis_height),
                                 Vec3(lateral, r_l, axis_height), Vec3(-lateral, r_l, axis_height)]):
            wheel = self._add_wheel(pos, radius, True if k < 2 else False, True if k == 0 or k == 2 else False)
            wheels.append(wheel)
        return wheels

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        self._node_path_list.append(wheel_np)

        if self.render:
            model = 'right_tire_front.gltf' if front else 'right_tire_back.gltf'
            model_path = AssetLoader.file_path("models", os.path.dirname(self.path[0]), model)
            wheel_model = self.loader.loadModel(model_path)
            wheel_model.setTwoSided(self.TIRE_TWO_SIDED)
            wheel_model.reparentTo(wheel_np)
            wheel_model.set_scale(1 * self.TIRE_MODEL_CORRECT if left else -1 * self.TIRE_MODEL_CORRECT)
        wheel = self.system.create_wheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))

        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(self.SUSPENSION_LENGTH)
        wheel.setSuspensionStiffness(self.SUSPENSION_STIFFNESS)
        wheel.setWheelsDampingRelaxation(4.8)
        wheel.setWheelsDampingCompression(1.2)
        wheel_friction = self.config["wheel_friction"] if not self.config["no_wheel_friction"] else 0
        wheel.setFrictionSlip(wheel_friction)
        wheel.setRollInfluence(0.5)
        return wheel

    def add_image_sensor(self, name: str, sensor: ImageBuffer):
        self.image_sensors[name] = sensor
        self.engine.graphicsEngine.render_frame()
        self.engine.graphicsEngine.render_frame()

    def add_navigation(self):
        if not self.config["need_navigation"]:
            return
        navi = self.config["navigation_module"]
        if navi is None:
            navi = NodeNetworkNavigation if self.engine.current_map.road_network_type == NodeRoadNetwork \
                else EdgeNetworkNavigation
        self.navigation = navi(
            # self.engine,
            show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
            random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
            show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
            show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"],
            panda_color=self.panda_color,
            name=self.name,
            vehicle_config=self.config
        )

    def update_map_info(self, map):
        """
        Update map information that are used by this vehicle, after reset()
        This function will query the map about the spawn position and destination of current vehicle,
        and update the navigation module by feeding the information of spawn point and destination.

        For the spawn position, if it is not specify in the config["spawn_lane_index"], we will automatically
        select one lane based on the localization results.

        :param map: new map
        :return: None
        """
        if not self.config["need_navigation"]:
            return
        possible_lanes = ray_localization(self.heading, self.spawn_place, self.engine, use_heading_filter=False)
        possible_lane_indexes = [lane_index for lane, lane_index, dist in possible_lanes]

        if len(possible_lanes) == 0 and self.config["spawn_lane_index"] is None:
            if map.road_network_type != EdgeRoadNetwork:
                from metadrive.utils.error_class import NavigationError
                raise NavigationError("Can't find valid lane for navigation.")
            # Overriding with nearest lane for scenario maps
            closest_lane_id, closest_lane_dist = -1, np.inf
            for lane_id, lane_info in map.road_network.graph.items():
                lane_ = lane_info.lane
                lane_dist = lane_.distance(self.spawn_place)
                if lane_dist < closest_lane_dist:
                    closest_lane_dist = lane_dist
                    closest_lane_id = lane_id
            lane = map.road_network.get_lane(closest_lane_id)
            possible_lanes = [[lane, closest_lane_id, closest_lane_dist]]
            possible_lane_indexes = [closest_lane_id]

        if self.config["spawn_lane_index"] is not None and self.config["spawn_lane_index"] in possible_lane_indexes:
            idx = possible_lane_indexes.index(self.config["spawn_lane_index"])
            lane, new_l_index = possible_lanes[idx][:-1]
        else:
            assert len(possible_lanes) > 0
            lane, new_l_index = possible_lanes[0][:-1]

        dest = self.config["destination"]
        self.navigation.reset(
            map,
            current_lane=lane,
            destination=dest if dest is not None else None,
            random_seed=self.engine.global_random_seed
        )
        assert lane is not None, "spawn place is not on road!"
        self.navigation.update_localization(self)

    def _state_check(self):
        """
        Check States and filter to update info
        """
        result_1 = self.engine.physics_world.static_world.contactTest(self.chassis.node(), True)
        result_2 = self.engine.physics_world.dynamic_world.contactTest(self.chassis.node(), True)
        contacts = set()
        for contact in result_1.getContacts() + result_2.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            node = node0 if node1.getName() == MetaDriveType.VEHICLE else node1
            name = node.getName()
            if name == MetaDriveType.LINE_SOLID_SINGLE_WHITE:
                self.on_white_continuous_line = True
            elif name == MetaDriveType.LINE_SOLID_SINGLE_YELLOW:
                self.on_yellow_continuous_line = True
            elif name == MetaDriveType.LINE_BROKEN_SINGLE_YELLOW or name == MetaDriveType.LINE_BROKEN_SINGLE_WHITE:
                self.on_broken_line = True
            elif name == MetaDriveType.TRAFFIC_LIGHT:
                light = get_object_from_node(node)
                if light.status == MetaDriveType.LIGHT_GREEN:
                    self.green_light = True
                elif light.status == MetaDriveType.LIGHT_RED:
                    self.red_light = True
                elif light.status == MetaDriveType.LIGHT_YELLOW:
                    self.yellow_light = True
                elif light.status == MetaDriveType.LIGHT_UNKNOWN:
                    # unknown didn't add
                    continue
                else:
                    raise ValueError("Unknown light status: {}".format(light.status))
                name = light.status
            # they work with the function in collision_callback.py to double-check the collision
            elif name == MetaDriveType.VEHICLE:
                self.crash_vehicle = True
                self.ego_crash_flag = True
            elif name == MetaDriveType.BUILDING:
                self.crash_building = True
            elif MetaDriveType.is_traffic_object(name):
                self.crash_object = True
            elif name in [MetaDriveType.PEDESTRIAN, MetaDriveType.CYCLIST]:
                self.crash_human = True
            else:
                # didn't add
                continue
            contacts.add(name)
        # side walk detect
        res = rect_region_detection(
            self.engine,
            self.position,
            np.rad2deg(self.heading_theta),
            self.LENGTH,
            self.WIDTH,
            CollisionGroup.Sidewalk,
            in_static_world=True if not self.render else False
        )
        if res.hasHit() and res.getNode().getName() == MetaDriveType.BOUNDARY_LINE:
            self.crash_sidewalk = True
            contacts.add(MetaDriveType.BOUNDARY_LINE)

        # only for visualization detection
        if self.render:
            res = rect_region_detection(
                self.engine,
                self.position,
                np.rad2deg(self.heading_theta),
                self.LENGTH,
                self.WIDTH,
                CollisionGroup.LaneSurface,
                in_static_world=True if not self.engine.global_config["debug_static_world"] else False
            )
            if not res.hasHit() or res.getNode().getName() != MetaDriveType.LANE_SURFACE_STREET:
                contacts.add(MetaDriveType.GROUND)

        self.contact_results.update(contacts)

    def destroy(self):
        super(BaseVehicle, self).destroy()
        if self.navigation is not None:
            self.navigation.destroy()
        self.navigation = None
        self.wheels = None
        if self.side_detector is not None:
            self.side_detector.destroy()
            self.side_detector = None
        if self.lane_line_detector is not None:
            self.lane_line_detector.destroy()
            self.lane_line_detector = None
        if self.lidar is not None:
            self.lidar.destroy()
            self.lidar = None
        if len(self.image_sensors) != 0:
            for sensor in self.image_sensors.values():
                if sensor is not self.engine.main_camera:
                    sensor.destroy()
        self.image_sensors = {}
        if self.light is not None:
            self.remove_light()

    def set_velocity(self, direction, *args, **kwargs):
        super(BaseVehicle, self).set_velocity(direction, *args, **kwargs)
        self.last_velocity = self.velocity
        self.last_speed = self.speed

    def set_state(self, state):
        super(BaseVehicle, self).set_state(state)
        self.set_throttle_brake(float(state["throttle_brake"]))
        self.set_steering(float(state["steering"]))
        self.last_velocity = self.velocity
        self.last_speed = self.speed
        self.last_position = self.position
        self.last_heading_dir = self.heading

    def set_panda_pos(self, pos):
        super(BaseVehicle, self).set_panda_pos(pos)
        self.last_position = self.position

    def set_position(self, position, height=None):
        super(BaseVehicle, self).set_position(position, height)
        self.last_position = self.position

    def get_state(self):
        """
        Fetch more information
        """
        state = super(BaseVehicle, self).get_state()
        state.update(
            {
                "steering": self.steering,
                "throttle_brake": self.throttle_brake,
                "crash_vehicle": self.crash_vehicle,
                "crash_object": self.crash_object,
                "crash_building": self.crash_building,
                "crash_sidewalk": self.crash_sidewalk,
                "size": (self.LENGTH, self.WIDTH, self.HEIGHT),
                "length": self.LENGTH,
                "width": self.WIDTH,
                "height": self.HEIGHT,
            }
        )
        if self.navigation is not None:
            state.update(self.navigation.get_state())
        return state

    # def get_raw_state(self):
    #     ret = dict(position=self.position, heading=self.heading, velocity=self.velocity)
    #     return ret

    def get_dynamics_parameters(self):
        # These two can be changed on the fly
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]

        # These two can only be changed in init
        wheel_friction = self.config["wheel_friction"]
        assert self.max_steering == self.config["max_steering"]
        max_steering = self.max_steering

        mass = self.config["mass"] if self.config["mass"] else self.MASS

        ret = dict(
            max_engine_force=max_engine_force,
            max_brake_force=max_brake_force,
            wheel_friction=wheel_friction,
            max_steering=max_steering,
            mass=mass
        )
        return ret

    def _update_overtake_stat(self):
        if self.config["overtake_stat"] and self.lidar.available:
            surrounding_vs = self.lidar.get_surrounding_vehicles()
            routing = self.navigation
            ckpt_idx = routing._target_checkpoints_index
            for surrounding_v in surrounding_vs:
                if surrounding_v.lane_index[:-1] == (routing.checkpoints[ckpt_idx[0]], routing.checkpoints[ckpt_idx[1]
                                                                                                           ]):
                    if self.lane.local_coordinates(self.position)[0] - \
                            self.lane.local_coordinates(surrounding_v.position)[0] < 0:
                        self.front_vehicles.add(surrounding_v)
                        if surrounding_v in self.back_vehicles:
                            self.back_vehicles.remove(surrounding_v)
                    else:
                        self.back_vehicles.add(surrounding_v)
        return {"overtake_vehicle_num": self.get_overtake_num()}

    def get_overtake_num(self):
        return len(self.front_vehicles.intersection(self.back_vehicles))

    def __del__(self):
        super(BaseVehicle, self).__del__()
        # self.engine = None
        self.lidar = None
        self.mini_map = None
        self.rgb_camera = None
        self.navigation = None
        self.wheels = None

    @property
    def reference_lanes(self):
        return self.navigation.current_ref_lanes

    def set_wheel_friction(self, new_friction):
        raise ValueError()
        for wheel in self.wheels:
            wheel.setFrictionSlip(new_friction)

    @property
    def overspeed(self):
        return True if self.lane.speed_limit < self.speed_km_h else False

    @property
    def replay_done(self):
        return self._replay_done if hasattr(self, "_replay_done") else (
            self.crash_building or self.crash_vehicle or
            # self.on_white_continuous_line or
            self.on_yellow_continuous_line
        )

    @property
    def current_action(self):
        return self.last_current_action[-1]

    @property
    def last_current(self):
        return self.last_current_action[0]

    def detach_from_world(self, physics_world):
        if self.navigation is not None:
            self.navigation.detach_from_world()
        if self.lidar is not None:
            self.lidar.detach_from_world()
        if self.side_detector is not None:
            self.side_detector.detach_from_world()
        if self.lane_line_detector is not None:
            self.lane_line_detector.detach_from_world()
        super(BaseVehicle, self).detach_from_world(physics_world)

    def attach_to_world(self, parent_node_path, physics_world):
        if self.config["show_navi_mark"] and self.config["need_navigation"]:
            self.navigation.attach_to_world(self.engine)
        if self.lidar is not None and self.config["show_lidar"]:
            self.lidar.attach_to_world(self.engine)
        if self.side_detector is not None and self.config["show_side_detector"]:
            self.side_detector.attach_to_world(self.engine)
        if self.lane_line_detector is not None and self.config["show_lane_line_detector"]:
            self.lane_line_detector.attach_to_world(self.engine)
        super(BaseVehicle, self).attach_to_world(parent_node_path, physics_world)

    def set_break_down(self, break_down=True):
        self.break_down = break_down
        # self.set_static(True)

    @property
    def max_speed_km_h(self):
        return self.config["max_speed_km_h"]

    @property
    def max_speed_m_s(self):
        return self.config["max_speed_km_h"] / 3.6

    @property
    def top_down_length(self):
        return self.LENGTH

    @property
    def top_down_width(self):
        return self.WIDTH

    @property
    def lane(self):
        return self.navigation.current_lane

    @property
    def lane_index(self):
        return self.navigation.current_lane.index

    @property
    def panda_color(self):
        c = super(BaseVehicle, self).panda_color
        if self._use_special_color:
            color = sns.color_palette("colorblind")
            rand_c = color[2]  # A pretty green
            c = rand_c
        return c

    def before_reset(self):
        for obj in [self.navigation, self.lidar, self.side_detector, self.lane_line_detector]:
            if obj is not None and hasattr(obj, "before_reset"):
                obj.before_reset()

    """------------------------------------------- overwrite -------------------------------------------------"""

    def convert_to_world_coordinates(self, vector, origin):
        return super(BaseVehicle, self).convert_to_world_coordinates([-vector[-1], vector[0]], origin)

    def convert_to_local_coordinates(self, vector, origin):
        ret = super(BaseVehicle, self).convert_to_local_coordinates(vector, origin)
        return np.array([ret[1], -ret[0]])

    @property
    def heading_theta(self):
        return wrap_to_pi(super(BaseVehicle, self).heading_theta + np.pi / 2)

    def set_heading_theta(self, heading_theta, in_rad=True) -> None:
        """
        Set heading theta for this object. Vehicle local frame has a 90 degree offset
        :param heading_theta: float in rad
        :param in_rad: when set to True, heading theta should be in rad, otherwise, in degree
        """
        super(BaseVehicle, self).set_heading_theta(heading_theta - np.pi / 2, in_rad)
        self.last_heading_dir = self.heading

    @property
    def roll(self):
        """
        Return the roll of this object
        """
        return np.deg2rad(self.origin.getR())

    def set_roll(self, roll):
        self.origin.setR(roll)

    @property
    def pitch(self):
        """
        Return the pitch of this object
        """
        return np.deg2rad(self.origin.getP())

    def set_pitch(self, pitch):
        self.origin.setP(pitch)

    def show_coordinates(self):
        if self.coordinates_debug_np is not None:
            self.coordinates_debug_np.reparentTo(self.origin)
            return
        height = self.HEIGHT + 0.2
        self.coordinates_debug_np = NodePath("debug coordinate")
        # 90 degrees offset
        x = self.engine.add_line([0, 0, height], [0, 2, height], [1, 1, 1, 1], 2)
        y = self.engine.add_line([0, 0, height], [-1, 0, height], [1, 1, 1, 1], 2)
        z = self.engine.add_line([0, 0, height], [0, 0, height + 0.5], [1, 1, 1, 1], 2)
        x.reparentTo(self.coordinates_debug_np)
        y.reparentTo(self.coordinates_debug_np)
        z.reparentTo(self.coordinates_debug_np)
        self.coordinates_debug_np.reparentTo(self.origin)
