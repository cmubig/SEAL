class MetaDriveType:
    """
    Following waymo style, this class defines a set of strings used to denote different types of objects.
    Those types are used within MetaDrive and might mismatch to the strings used in other dataset.

    NOTE: when add new keys, make sure class method works well for them
    """

    # ===== Lane, Road =====
    LANE_SURFACE_STREET = "LANE_SURFACE_STREET"
    LANE_UNKNOWN = "LANE_UNKNOWN"
    LANE_FREEWAY = "LANE_FREEWAY"
    LANE_BIKE_LANE = "LANE_BIKE_LANE"

    # ===== Lane Line =====
    LINE_UNKNOWN = "UNKNOWN_LINE"
    LINE_BROKEN_SINGLE_WHITE = "ROAD_LINE_BROKEN_SINGLE_WHITE"
    LINE_SOLID_SINGLE_WHITE = "ROAD_LINE_SOLID_SINGLE_WHITE"
    LINE_SOLID_DOUBLE_WHITE = "ROAD_LINE_SOLID_DOUBLE_WHITE"
    LINE_BROKEN_SINGLE_YELLOW = "ROAD_LINE_BROKEN_SINGLE_YELLOW"
    LINE_BROKEN_DOUBLE_YELLOW = "ROAD_LINE_BROKEN_DOUBLE_YELLOW"
    LINE_SOLID_SINGLE_YELLOW = "ROAD_LINE_SOLID_SINGLE_YELLOW"
    LINE_SOLID_DOUBLE_YELLOW = "ROAD_LINE_SOLID_DOUBLE_YELLOW"
    LINE_PASSING_DOUBLE_YELLOW = "ROAD_LINE_PASSING_DOUBLE_YELLOW"

    # ===== Edge/Boundary/SideWalk/Region =====
    BOUNDARY_UNKNOWN = "UNKNOWN"
    BOUNDARY_LINE = "ROAD_EDGE_BOUNDARY"
    BOUNDARY_MEDIAN = "ROAD_EDGE_MEDIAN"
    STOP_SIGN = "STOP_SIGN"
    CROSSWALK = "CROSSWALK"
    SPEED_BUMP = "SPEED_BUMP"
    DRIVEWAY = "DRIVEWAY"

    # ===== Traffic Light =====
    LANE_STATE_UNKNOWN = "LANE_STATE_UNKNOWN"
    LANE_STATE_ARROW_STOP = "LANE_STATE_ARROW_STOP"
    LANE_STATE_ARROW_CAUTION = "LANE_STATE_ARROW_CAUTION"
    LANE_STATE_ARROW_GO = "LANE_STATE_ARROW_GO"
    LANE_STATE_STOP = "LANE_STATE_STOP"
    LANE_STATE_CAUTION = "LANE_STATE_CAUTION"
    LANE_STATE_GO = "LANE_STATE_GO"
    LANE_STATE_FLASHING_STOP = "LANE_STATE_FLASHING_STOP"
    LANE_STATE_FLASHING_CAUTION = "LANE_STATE_FLASHING_CAUTION"

    # the light states above will be converted to the following 4 types
    LIGHT_GREEN = "TRAFFIC_LIGHT_GREEN"
    LIGHT_RED = "TRAFFIC_LIGHT_RED"
    LIGHT_YELLOW = "TRAFFIC_LIGHT_YELLOW"
    LIGHT_UNKNOWN = "TRAFFIC_LIGHT_UNKNOWN"

    # ===== Agent type =====
    UNSET = "UNSET"
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    CYCLIST = "CYCLIST"
    OTHER = "OTHER"

    # ===== Object type =====
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
    TRAFFIC_BARRIER = "TRAFFIC_BARRIER"
    TRAFFIC_CONE = "TRAFFIC_CONE"
    TRAFFIC_OBJECT = "TRAFFIC_OBJECT"
    GROUND = "GROUND"
    INVISIBLE_WALL = "INVISIBLE_WALL"
    BUILDING = "BUILDING"

    # ===== Coordinate system =====
    COORDINATE_METADRIVE = "metadrive"
    COORDINATE_WAYMO = "waymo"

    # deprecated
    # LIGHT_ENUM_TO_STR = {
    #     0: LANE_STATE_UNKNOWN,
    #     1: LANE_STATE_ARROW_STOP,
    #     2: LANE_STATE_ARROW_CAUTION,
    #     3: LANE_STATE_ARROW_GO,
    #     4: LANE_STATE_STOP,
    #     5: LANE_STATE_CAUTION,
    #     6: LANE_STATE_GO,
    #     7: LANE_STATE_FLASHING_STOP,
    #     8: LANE_STATE_FLASHING_CAUTION
    # }

    @classmethod
    def is_traffic_object(cls, type):
        return type in [cls.TRAFFIC_CONE, cls.TRAFFIC_BARRIER, cls.TRAFFIC_OBJECT]

    @classmethod
    def has_type(cls, type_string: str):
        return type_string in cls.__dict__

    @classmethod
    def from_waymo(cls, waymo_type_string: str):
        assert cls.__dict__[waymo_type_string]
        return waymo_type_string

    @classmethod
    def from_nuplan(cls, waymo_type_string: str):
        # TODO: WIP
        return ""

    @classmethod
    def is_lane(cls, type):
        return type in [cls.LANE_SURFACE_STREET, cls.LANE_FREEWAY, cls.LANE_BIKE_LANE]

    @classmethod
    def is_road_line(cls, line):
        """
        This function relates to is_road_edge. We will have different processing when treating a line that
        is in the boundary or not.
        """
        return line in [
            cls.LINE_UNKNOWN, cls.LINE_BROKEN_SINGLE_WHITE, cls.LINE_SOLID_SINGLE_WHITE, cls.LINE_SOLID_DOUBLE_WHITE,
            cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW
        ]

    @classmethod
    def is_yellow_line(cls, line):
        return line in [
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW
        ]

    @classmethod
    def is_white_line(cls, line):
        return MetaDriveType.is_road_line(line) and not MetaDriveType.is_yellow_line(line)

    @classmethod
    def is_broken_line(cls, line):
        return line in [cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_SINGLE_WHITE]

    @classmethod
    def is_road_edge(cls, edge):
        """
        This function relates to is_road_line.
        """
        return edge in [cls.BOUNDARY_UNKNOWN, cls.BOUNDARY_LINE, cls.BOUNDARY_MEDIAN]

    @classmethod
    def is_sidewalk(cls, edge):
        return edge == cls.BOUNDARY_LINE

    @classmethod
    def is_vehicle(cls, type):
        return type == cls.VEHICLE

    @classmethod
    def is_traffic_light_in_yellow(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_YELLOW

    @classmethod
    def is_traffic_light_in_green(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_GREEN

    @classmethod
    def is_traffic_light_in_red(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_RED

    @classmethod
    def is_traffic_light_unknown(cls, light):
        return cls.simplify_light_status(light) == cls.LIGHT_UNKNOWN

    @classmethod
    def parse_light_status(cls, status: str, simplifying=True):
        """
        Parse light status from ENUM to STR
        """
        # if data_source == "waymo":
        #     status = cls.LIGHT_ENUM_TO_STR[status]
        if simplifying:
            return cls.simplify_light_status(status)
        else:
            return status

    @classmethod
    def simplify_light_status(cls, status: str):
        """
        Convert status to red/yellow/green/unknown
        """
        if status in [cls.LANE_STATE_UNKNOWN, cls.LANE_STATE_FLASHING_STOP, cls.LIGHT_UNKNOWN, None]:
            return cls.LIGHT_UNKNOWN
        elif status in [cls.LANE_STATE_ARROW_STOP, cls.LANE_STATE_STOP, cls.LIGHT_RED]:
            return cls.LIGHT_RED
        elif status in [cls.LANE_STATE_ARROW_CAUTION, cls.LANE_STATE_CAUTION, cls.LANE_STATE_FLASHING_CAUTION,
                        cls.LIGHT_YELLOW]:
            return cls.LIGHT_YELLOW
        elif status in [cls.LANE_STATE_ARROW_GO, cls.LANE_STATE_GO, cls.LIGHT_GREEN]:
            return cls.LIGHT_GREEN
        else:
            raise ValueError("Status: {} is not MetaDriveType".format(status))
