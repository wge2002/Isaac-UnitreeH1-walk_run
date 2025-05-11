# 此处的python文件主要运用于gym中task的注册与登记

import gymnasium as gym

# from . import agents, flat_env_cfg, rough_env_cfg
from .UniTreeH1 import flat_env_cfg, rough_env_cfg
from .agents import rsl_rl_cfg

##
# Register Gym environments.
##

gym.register(
    # 指的是在崎岖、不平坦或高复杂度的环境中进行的任务或实验。
    id="RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeH1RoughEnvCfg,  # 环境的设置
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeH1RoughPPORunnerCfg,  # 算法代理的设置
    },
)

gym.register(
    # 指的是在平坦、无障碍或低复杂度的环境中进行的任务或实验。
    id="RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeH1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeH1FlatPPORunnerCfg,
    },
)