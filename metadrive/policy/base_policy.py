import copy
import logging
import uuid
from metadrive.constants import CamMask
import gym
import numpy as np
from panda3d.core import NodePath

from metadrive.base_class.configurable import Configurable
from metadrive.base_class.randomizable import Randomizable
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import get_engine


class BasePolicy(Randomizable, Configurable):
    DEBUG_MARK_COLOR = (255, 255, 255, 255)
    DEBUG_MARK_MODEL = None
    SYNC_DEBUG_MARK_POS_TASK_NAME = "policy_mark"

    def __init__(self, control_object, random_seed=None, config=None):
        Randomizable.__init__(self, random_seed)
        Configurable.__init__(self, config)
        # self.engine = get_engine()
        self.control_object = control_object
        self.action_info = dict()
        self._debug_mark = None
        self._mark_update_task_name = None
        self.show_policy_mark()

    def _get_task_name(self):
        return self.SYNC_DEBUG_MARK_POS_TASK_NAME + str(uuid.uuid4())

    def act(self, *args, **kwargs):
        """
        Return action [], policy implement information (dict) can be written in self.action_info, which will be
        retrieved automatically
        """
        pass

    def _sync_debug_mark(self, task):
        assert self._debug_mark is not None
        assert self._mark_update_task_name is not None
        pos = self.control_object.origin.getPos()
        height = getattr(self.control_object, "HEIGHT") if hasattr(self.control_object, "HEIGHT") else 2
        self._debug_mark.setPos(pos[0], pos[1], height + 0.5)
        return task.cont

    def get_action_info(self):
        """
        Get current action info for env.step() retrieve
        """
        return copy.deepcopy(self.action_info)

    def reset(self):
        self.action_info.clear()

    def destroy(self):
        if self._debug_mark is not None:
            self.engine.taskMgr.remove(self._mark_update_task_name)
            self._debug_mark.removeNode()
            self._debug_mark = None
            self._mark_update_task_name = None
        super(BasePolicy, self).destroy()
        self.control_object = None
        logging.debug("{} is released".format(self.__class__.__name__))

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.name

    @property
    def engine(self):
        return get_engine()

    @classmethod
    def get_input_space(cls):
        """
        It defines the input space of this class of policy
        """
        # logging.info(
        #     "No input space set for this policy! If you are querying an action space, "
        #     "the agent policy may not take any external input from env.step() and thus the env.action_space is None"
        # )
        return gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)

    def get_state(self):
        return self.get_action_info()

    @property
    def episode_step(self):
        return self.engine.episode_step

    def show_policy_mark(self):
        if not self.engine.global_config["show_policy_mark"]:
            return
        if self.DEBUG_MARK_MODEL is None:
            self.DEBUG_MARK_MODEL = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            self.DEBUG_MARK_MODEL.setScale(0.5)
            self.DEBUG_MARK_MODEL.show(CamMask.MainCam)
            self.DEBUG_MARK_MODEL.hprInterval(5, (360, 0, 0)).loop()
        self._debug_mark = NodePath(self.name)
        self.DEBUG_MARK_MODEL.instanceTo(self._debug_mark)

        # texture = AssetLoader.loader.loadTexture(AssetLoader.file_path("textures", "height_map.png"))
        # texture.set_format(Texture.F_srgb)
        # self._debug_mark.setTexture(texture)
        r, g, b, a = self.DEBUG_MARK_COLOR
        self._debug_mark.setColor(r / 255, g / 255, b / 255, a / 255)

        self._debug_mark.reparentTo(self.engine.origin)
        pos = self.control_object.origin.getPos()
        height = getattr(self.control_object, "HEIGHT") if hasattr(self.control_object, "HEIGHT") else 2
        self._debug_mark.setPos(pos[0], pos[1], height + 0.5)
        self._mark_update_task_name = self._get_task_name()
        self.engine.taskMgr.add(self._sync_debug_mark, self._mark_update_task_name)
