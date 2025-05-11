import torch
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster

def height_scan_my(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    # 由于sensor.data.ray_hits_w只能返回无穷大，会报错，所以只返回sensor.data.pos_w来代替
    # return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1)