# 这个python文件的作用就是在已经有训练完的模型基础上，运行它。
# cd D:\Nvidia\Omniverse\library\IsaacLab
# isaaclab.bat -p source\standalone\learn_walk_my\scripts\play.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0
# isaaclab.bat -p source\standalone\learn_walk_my\scripts\play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0', help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
# 先注释掉
# import robot_lab.tasks  # noqa: F401

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
# 将策略导出为Torch JIT文件。在这里，“策略”通常指的是一个用于决策或预测的机器学习模型。
# 将策略导出为Torch ONNX文件
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

import sys
home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将 {home} 目录添加到模块搜索路径中
sys.path.append(home_dir)
# 使得系统识别到task中的init文件，完成gym.register的注册。
import task


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # make a smaller scene for play
    # env_cfg.scene.num_envs = 1
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1, 1)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()