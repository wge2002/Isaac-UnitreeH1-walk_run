from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

# 类型检查，是否是ManagerBasedRLEnv类
if TYPE_CHECKING: 
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    使用L2内核奖励脚迈出的长步。
    此函数奖励代理执行超过阈值的步骤。这有助于确保机器人将脚抬离地面并采取行动。奖励按双脚在空中停留的时间总和计算。
    如果命令很小（即代理不应该迈出一步），则奖励为零。
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # 接触传感器
    # compute the reward
    # compute_first_contact函数
    # 一个布尔张量，表示在最后一个时间段内建立联系的物体：attr:`dt`秒。注意：布尔类型
    # 形状为（N，B），其中N是传感器的数量，B是每个传感器中的物体数量。
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    # 这行代码确保只有在命令的幅度（在xy轴上）大于0.1时，才给予奖励。
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    奖励两足动物用脚迈出的长步。
    此函数奖励智能体达到指定阈值的步骤，并一次保持一只脚在空中。
    如果命令很小（即代理不应该迈出一步），则奖励为零。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    # 使用torch.where函数根据in_contact的值选择contact_time（如果接触）或air_time（如果未接触）。
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # 表示在每个时间步中，是否只有一个身体部分在接触地面。这是通过检查in_contact中每行（即每个时间步）的和是否等于1来实现的。
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # 使用torch.clamp将奖励限制在一个最大阈值threshold内。
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding  惩罚脚部滑行"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # net_forces_w_history：Shape is (N, T, B, 3)
    contacts = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel.
    在机器人的重力对齐坐标系中跟踪线性速度命令 在x和y轴上 并使用指数核来计算奖励。
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel.
    在机器人的重力对齐坐标系中跟踪角速度命令(在x和y轴上),并使用指数核来计算奖励。
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def foot_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward foot_contact 奖励脚的接触"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # compute_first_contact在上面有解释
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2-kernel.  使用L2内核惩罚资产距离目标的高度。
    高度惩罚
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    base_height = torch.mean(asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2], dim=1)
    return torch.square(base_height - target_height)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power
    奖励关节的力量
    """ 
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    # 角速度和扭矩的逐元素乘法计算了每个关节上由于扭矩作用而产生的瞬时功率。
    # 功率是力和速度（或扭矩和角速度）的点积，它表示单位时间内完成的功。
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still_when_zero_command(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command.
    当没有命令时，惩罚偏离默认位置的关节位置。
    # 没有命令时稳定在本位
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return torch.sum(torch.abs(diff_angle), dim=1) * command

def arm_swing_coordination(
    env: ManagerBasedRLEnv,
    command_name: str,
    arm_joint_cfg: SceneEntityCfg,
    leg_sensor_cfg: SceneEntityCfg,
    velocity_scale: float = 2.0,
    coordination_gain: float = 0.5,
    std: float = 0.25
) -> torch.Tensor:
    """Reward coordinated arm swinging during locomotion using exponential kernel.
    
    该函数通过以下方式奖励摆臂动作：
    1. 检测腿部接触状态确定步态相位
    2. 计算手臂关节速度的协调性（异侧手臂同步运动）
    3. 结合速度幅度和协调性计算奖励
    
    Args:
        velocity_scale: 手臂速度的缩放因子，用于调整速度影响的敏感度
        coordination_gain: 协调性奖励的增益系数
        std: 指数核的标准差参数
    """
    # 提取必要组件
    asset: Articulation = env.scene[arm_joint_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[leg_sensor_cfg.name]
    # print("ContactSensorData attributes:", dir(contact_sensor.data))
    # 获取腿部接触状态（假设前两个身体是左右腿）
    # 获取左右腿的当前接触时间（>0 表示接触）
    contact_time = contact_sensor.data.current_contact_time[:, leg_sensor_cfg.body_ids]
    left_in_contact = contact_time[:, 0] > 0.0  # 左腿接触状态
    right_in_contact = contact_time[:, 1] > 0.0  # 右腿接触状态
    
    # 获取手臂关节速度（假设前两个关节是左右肩关节）
    arm_vel = asset.data.joint_vel[:, arm_joint_cfg.joint_ids]
    # Shape: [num_envs, 2] 其中第二维是[left_arm, right_arm]
    
    # 计算协调性指标（异侧运动协调）
    # 当左腿离地时（摆动期），右臂应该向前摆动（正速度）
    # 当右腿离地时，左臂应该向前摆动
    phase_coordination = (
        (left_in_contact * arm_vel[:, 1]) +  # 左触地时右臂摆动
        (right_in_contact * arm_vel[:, 0])    # 右触地时左臂摆动
    )
    
    # 计算速度幅度奖励（鼓励适当幅度摆动）
    velocity_magnitude = torch.norm(arm_vel, dim=1) * velocity_scale
    
    # 综合协调性和速度幅度
    swing_reward = torch.exp(-(phase_coordination - velocity_magnitude)**2 / std**2)
    
    # 增加双臂反相位运动奖励（交叉协调）
    cross_coordination = -arm_vel[:, 0] * arm_vel[:, 1]  # 速度方向相反时值为正
    total_reward = swing_reward + coordination_gain * cross_coordination
    
    # 仅在有效移动时应用奖励
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return total_reward * moving


def simple_arm_swing_reward(
        
    env: ManagerBasedRLEnv,
    arm_joint_cfg: SceneEntityCfg,
    velocity_scale: float = 0.1,
) -> torch.Tensor:
    """简单的手臂摆动奖励，仅鼓励关节运动
    
    Args:
        velocity_scale: 速度奖励的缩放因子 (默认0.1)
    """
    # 获取手臂关节速度 [num_envs, num_joints]
    arm_vel = env.scene[arm_joint_cfg.name].data.joint_vel[:, arm_joint_cfg.joint_ids]
    
    # 计算各关节速度绝对值之和作为奖励
    reward = torch.sum(torch.abs(arm_vel), dim=1) * velocity_scale
    
    return reward



def penalize_torso_twist(
    env: ManagerBasedRLEnv,
    torso_joint_cfg: SceneEntityCfg,
    velocity_scale: float = 0.5,
    max_penalty: float = 2.0
) -> torch.Tensor:
    """修正版腰部扭动惩罚（适配实际数据结构）
    
    关键修正：
    - 使用joint_vel而不是joint_ang_vel（更可靠）
    - 正确处理2D张量结构
    """
    # 获取腰部关节数据
    torso = env.scene[torso_joint_cfg.name]
    
    # 提取目标关节的线速度 [num_envs, num_joints]
    joint_vel = torso.data.joint_vel[:, torso_joint_cfg.joint_ids]
    
    # 计算速度绝对值 [num_envs, num_joints]
    vel_magnitude = torch.abs(joint_vel)
    
    # 取所有目标关节的最大值 [num_envs]
    max_twist = torch.max(vel_magnitude, dim=1)[0]
    
    # 计算惩罚项（线性版本更稳定）
    penalty = velocity_scale * max_twist
    
    # 应用惩罚限制
    penalty = torch.clamp(penalty, max=max_penalty)
    
    return penalty  # 返回正惩罚值

