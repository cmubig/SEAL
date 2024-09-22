import numpy as np

from metadrive.component.pg_space import BlockParameterSpace, Parameter
from metadrive.component.vehicle_navigation_module.base_navigation import BaseNavigation
from metadrive.utils.math import norm, clip
from metadrive.utils.math import panda_vector
from metadrive.utils.math import wrap_to_pi


class TrajectoryNavigation(BaseNavigation):
    """
    This module enabling follow a given reference trajectory given a map
    # TODO(LQY): make this module a general module for navigation
    """
    DISCRETE_LEN = 8  # m

    def __init__(
        self,
        show_navi_mark: bool = False,
        random_navi_mark_color=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        super(TrajectoryNavigation, self).__init__(
            show_navi_mark=show_navi_mark,
            random_navi_mark_color=random_navi_mark_color,
            show_dest_mark=show_dest_mark,
            show_line_to_dest=show_line_to_dest,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        # self.reference_trajectory = None
        # self.current_map_lane = None
        # self.is_on_lane = False

    def reset(self, map=None, current_lane=None, destination=None, random_seed=None):

        # We do not want to store map within the navigation module!
        # TODO(PZH): In future, we can let all navigation module get latest map on-the-fly instead of
        #  caching a class local variable.
        super(TrajectoryNavigation, self).reset(map=map, current_lane=current_lane)
        # self.reference_trajectory = self.get_trajectory()
        if self.reference_trajectory is not None:
            self.set_route(None, None)

    @property
    def reference_trajectory(self):
        if self.name != 'default_agent':
            return self.engine.map_manager.other_routes[self.name]
        return self.engine.map_manager.current_sdc_route

    @property
    def current_ref_lanes(self):
        return [self.reference_trajectory]

    def set_route(self, current_lane_index: str, destination: str):
        self.checkpoints = self.discretize_reference_trajectory()
        self._target_checkpoints_index = [0, 1] if len(self.checkpoints) >= 2 else [0, 0]
        # update routing info
        # assert len(self.checkpoints
        #            ) >= 2, "Can not find a route from {} to {}".format(current_lane_index[0], destination)

        self._navi_info.fill(0.0)
        # self.current_ref_lanes = [self.reference_trajectory]
        self.next_ref_lanes = None
        # self.current_lane = self.final_lane = self.reference_trajectory
        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], 1.8))

    # def get_trajectory(self):
    #     """This function breaks Multi-agent Waymo Env since we don't set this in map_manager."""
    #     return self.engine.map_manager.current_sdc_route

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DISCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.position(i * self.DISCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret

    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        if self.reference_trajectory is None:
            # self.current_map_lane = None
            # self.is_on_lane = False
            return

        # def distance(s, r, ref):
        #     a = s - ref.length
        #     b = 0 - s
        #     return ((a if a > 0 else 0) + (b if b > 0 else 0), abs(r))
        # def on_lane(long_dist, lat_dist, ref):
        #     return long_dist < ego_vehicle.LENGTH / 2 and lat_dist < ref.width / 2 + ego_vehicle.WIDTH / 2

        # Update actual map lane distances
        # long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position, only_in_lane_point=True)

        # self.is_on_lane = False
        # ref_dist_long, ref_dist_lat = distance(long, lat, self.reference_trajectory)
        # if on_lane(ref_dist_long, ref_dist_lat, self.reference_trajectory):
        #     self.is_on_lane = True
        # elif self.current_map_lane is not None:
        #     lane_ = self.current_map_lane
        #     lane_long, lane_lat = lane_.local_coordinates(ego_vehicle.position)
        #     if on_lane(*distance(lane_long, lane_lat, lane_), lane_):
        #         lane_heading = lane_.heading_theta_at(long)
        #         # Roughly 30 degrees permitted
        #         same_direction = abs(wrap_to_pi(lane_heading) - wrap_to_pi(ego_vehicle.heading_theta)) < 0.5
        #         if same_direction:
        #             self.is_on_lane = True

        # if not self.is_on_lane:
        #     map = ego_vehicle.navigation.map
        #     found = False
        #     for lane_id, lane_info in map.road_network.graph.items():
        #         lane_ = lane_info.lane
        #         lane_long, lane_lat = lane_.local_coordinates(ego_vehicle.position)
        #         if on_lane(*distance(lane_long, lane_lat, lane_), lane_):
        #             lane_heading = lane_.heading_theta_at(long)
        #             # Roughly 30 degrees permitted
        #             same_direction = abs(wrap_to_pi(lane_heading) - wrap_to_pi(ego_vehicle.heading_theta)) < 0.5
        #             if not same_direction:
        #                 continue
        #             found = True
        #             break
        #     if found:
        #         self.current_map_lane = lane_
        #         self.is_on_lane = True

        # Update ckpt index
        long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position, only_in_lane_point=True)
        if self._target_checkpoints_index[0] != self._target_checkpoints_index[1]:  # on last road
            # arrive to second checkpoint
            if lat < self.reference_trajectory.width:
                idx = max(int(long / self.DISCRETE_LEN) + 1, 0)
                idx = min(idx, len(self.checkpoints) - 1)
                self._target_checkpoints_index = [idx]
                if idx + 1 == len(self.checkpoints):
                    self._target_checkpoints_index.append(idx)
                else:
                    self._target_checkpoints_index.append(idx + 1)
        try:
            ckpt_1 = self.checkpoints[self._target_checkpoints_index[0]]
            ckpt_2 = self.checkpoints[self._target_checkpoints_index[1]]
        except:
            print(self.engine.global_seed)
            raise ValueError("target_ckpt".format(self._target_checkpoints_index))
        # target_road_1 is the road segment the vehicle is driving on.
        self._navi_info.fill(0.0)
        half = self.navigation_info_dim // 2
        self._navi_info[:half], lanes_heading1 = self._get_info_for_checkpoint(ckpt_1, ego_vehicle, long)

        self._navi_info[half:], lanes_heading2, = self._get_info_for_checkpoint(ckpt_2, ego_vehicle, long)

        if self._show_navi_info:
            # Whether to visualize little boxes in the scene denoting the checkpoints
            pos_of_goal = ckpt_1
            self._goal_node_path.setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], 1.8))
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)
            self.navi_arrow_dir = [lanes_heading1, lanes_heading2]
            dest_pos = self._dest_node_path.getPos()
            self._draw_line_to_dest(start_position=ego_vehicle.position, end_position=(dest_pos[0], -dest_pos[1]))
            navi_pos = self._goal_node_path.getPos()
            self._draw_line_to_navi(start_position=ego_vehicle.position, end_position=(navi_pos[0], -navi_pos[1]))

    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * 2

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    def _get_info_for_checkpoint(self, checkpoint, ego_vehicle, longitude):

        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        dir_vec = checkpoint - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )  # project to ego vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(clip((ckpt_in_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(clip((ckpt_in_rhs / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        lanes_heading = self.reference_trajectory.heading_theta_at(longitude)

        # TODO(LQY) Try to include the current lane's information into the navigation information
        bendradius = 0.0
        dir = 0.0
        angle = 0.0

        # # Dim 3: The bending radius of current lane
        navi_information.append(clip(bendradius, 0.0, 1.0))
        #
        # # Dim 4: The bending direction of current lane (+1 for clockwise, -1 for counterclockwise)
        navi_information.append(clip((dir + 1) / 2, 0.0, 1.0))
        #
        # # Dim 5: The angular difference between the heading in lane ending position and
        # # the heading in lane starting position
        navi_information.append(
            clip((np.rad2deg(angle) / BlockParameterSpace.CURVE[Parameter.angle].max + 1) / 2, 0.0, 1.0)
        )
        return navi_information, lanes_heading

    def destroy(self):
        self.map = None
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None
        super(TrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.map = None
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None


WaymoTrajectoryNavigation = TrajectoryNavigation
NuPlanTrajectoryNavigation = TrajectoryNavigation
