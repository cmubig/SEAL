from metadrive.component.vehicle_module.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.engine.engine_utils import engine_initialized, get_global_config
from direct.filter.CommonFilters import CommonFilters


class RGBCamera(BaseCamera):
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = CamMask.RgbCam
    PBR_ADAPT = False

    def __init__(self):
        assert engine_initialized(), "You should initialize engine before adding camera to vehicle"
        config = get_global_config()["vehicle_config"]["rgb_camera"]
        self.BUFFER_W, self.BUFFER_H = config[0], config[1]
        cuda = True if get_global_config()["vehicle_config"]["image_source"] == "rgb_camera" else False
        super(RGBCamera, self).__init__(True, cuda)
        cam = self.get_cam()
        lens = self.get_lens()
        cam.lookAt(2.4, 0, 1.3)

        lens.setFov(60)
        lens.setAspectRatio(2.0)
