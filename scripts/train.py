# 此python文件主要是各种任务的train的入口，包括简单的命令行设置，以及agent和environment的初始化等等。

# 运行测试
# cd D:\Nvidia\Omniverse\library\IsaacLab
# python source\standalone\learn_walk_my\scripts\train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0 --headless # 别人的
# isaaclab.bat -p source\standalone\learn_walk_my\scripts\train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0 --headless
# python scripts/train.py --task Isaac-Cartpole-v0 --num_envs 256 先用官方的测试是否可以运行
# 粗糙地面步行
# isaaclab.bat -p source\standalone\learn_walk_my\scripts\train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 --headless

import argparse
import sys
import os
home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将 {home} 目录添加到模块搜索路径中
sys.path.append(home_dir)
from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0', help="Name of the task.")
parser.add_argument("--seed", type=int, default=2408, help="Seed used for the environment")
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
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
# 我在此先注释掉
# import exts.robot_lab.robot_lab.tasks  # noqa: F401

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import task

# 这些设置用于调整CUDA和cuDNN（NVIDIA的深度学习加速库）的行为，以优化性能和结果的可重复性。
torch.backends.cuda.matmul.allow_tf32 = True # 控制CUDA中的矩阵乘法（matmul）操作是否允许使用TensorFloat-32（TF32）格式。它在保持与FP32相似的精度的同时，提高了计算性能。
torch.backends.cudnn.allow_tf32 = True # 这个设置是针对cuDNN库中的操作。cuDNN是NVIDIA提供的用于深度神经网络的高性能GPU加速库。将此设置为True允许cuDNN中的某些操作使用TF32格式，以提高训练速度.
torch.backends.cudnn.deterministic = False # 这个设置控制cuDNN操作是否保证结果的可重复性。将其设置为False（默认值）时，cuDNN将选择可能更快但可能引入随机性的算法
torch.backends.cudnn.benchmark = False # 这个设置影响cuDNN自动调优算法的行为。
# 当设置为True时，cuDNN会在程序开始时自动选择一个最佳的卷积算法，以优化性能。
# 这个选择过程可能需要一些时间，但一旦完成，它将固定所选的算法，直到下次程序启动。
# 然而，如果输入数据的大小或形状在运行时发生变化，这可能不是最佳选择。
# 将此设置为False（默认值）时，cuDNN不会进行这种自动调优，而是会立即选择一个算法，
# 这可能会稍微降低性能，但避免了自动调优的开销，并且对于大小可变的输入数据更为灵活。


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    # 在创建新的log_dir之前保存恢复路径
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()