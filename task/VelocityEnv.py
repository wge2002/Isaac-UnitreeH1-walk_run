from __future__ import annotations

import math
from dataclasses import MISSING

# import exts.robot_lab.robot_lab.tasks.locomotion.velocity.mdp as mdp
from task import mdp

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip,导入不平整地面资产


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), # 定义了一个时间范围，用于重新采样速度命令。
        rel_standing_envs=0.02, # 与环境静态部分的相对站立环境因子
        rel_heading_envs=1.0, # 表示与环境方向相关的某个因子的值，用于调整或限制机器人朝向的命令。
        heading_command=True, # 是否允许发送朝向命令
        heading_control_stiffness=0.5, # 表示朝向控制的刚度或灵敏度，这是一个介于0和1之间的值，用于调整机器人朝向调整的反应速度和精度。
        debug_vis=True, # 指示是否启用调试可视化
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ), # 用于定义速度命令的线性速度（x和y方向）和角速度（z轴方向）以及朝向（heading）的允许范围
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip=None
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-100.0, 100.0),
            # scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            # clip=(-100.0, 100.0),
            # scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            # clip=(-100.0, 100.0),
            # scale=1.0,
        )
        joint_pos = ObsTerm(
            # “relative”（相对的）
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            # clip=(-100.0, 100.0),
            # scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            # clip=(-100.0, 100.0),
            # scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            # clip=(-100.0, 100.0),
            # scale=1.0,
        )
        height_scan = ObsTerm(
            # func=mdp.height_scan,
            func=mdp.height_scan_my,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
            # scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class AMPCfg(ObsGroup):
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    AMP: AMPCfg = AMPCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8), # 静态摩擦系数
            "dynamic_friction_range": (0.6, 0.6), # 动态摩擦系数
            "restitution_range": (0.0, 0.0), # 恢复系数（也称为弹性系数),决定了物体碰撞后的能量损失
            "num_buckets": 64, # The number of materials is specified by ``num_buckets``.
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0), # 指定了质量随机化的范围 distribution 分布
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        # force 力 作用在物体上的推或拉的作用力，直接改变物体的平动状态。 平
        # torque 力矩 力和力臂的乘积，描述作用在物体上的力产生的旋转效应，直接改变物体的转动状态。 旋
        func=mdp.apply_external_force_torque, # torque 扭矩
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, # uniform 均匀分布
        mode="reset",
        params={
            # yaw 实体绕垂直轴（通常是Z轴）的旋转角度（即偏航角）的重置范围，单位是弧度
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, 
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                # Roll（翻滚）：绕物体的纵轴（通常是x轴）旋转的角速度。在飞行器中，这会导致机身的横滚运动，即机身向左或向右倾斜。
                "roll": (-0.5, 0.5),
                # Pitch（俯仰）：绕物体的横轴（通常是y轴）旋转的角速度。在飞行器中，这会导致机身的爬升或下滑运动，即机身向上或向下倾斜。
                "pitch": (-0.5, 0.5),
                # Yaw（偏航）：绕物体的垂直轴（通常是z轴）旋转的角速度。在飞行器中，这会导致机身围绕垂直轴线的旋转，影响航向角和转弯半径。
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        # 通过按给定范围缩放默认位置和速度来重置机器人关节。
        func=mdp.reset_joints_by_scale,  # reset_joints_by_offset
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        # 随机化机器人的执行器增益，在这里主要是 重置了关节执行器的刚度和阻尼数值
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), # ".*"表示所有
            "stiffness_distribution_params": (0.8, 1.2), # 刚度（stiffness）
            "damping_distribution_params": (0.8, 1.2), # 阻尼（damping）
            "operation": "scale", # 倍化处理，缩放
            "distribution": "log_uniform", # 将对数均匀分布用于生成随机增益值
        },
    )

    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.75, 1.25), # friction 摩擦系数
            # armature通常指的是电机的转子部分
            "armature_distribution_params": (0.75, 1.25), # armature 电枢
            # 关节角度下限和上限，关节角度的限制通常是以角度（如度或弧度）为单位
            "lower_limit_distribution_params": (0.75, 1.25),
            "upper_limit_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # interval
    push_robot = EventTerm(
        # 模拟开始后10到15秒之间首次触发，并且之后周期性地在该时间范围内随机间隔触发的事件
        # 设置速度，将速度值推给机器人
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# 奖励，强化学习重点部分
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # General
    # UNUESD is_alive
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0)

    # Root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "target_height": 0.0},
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )

    # Joint penaltie
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0)
    # UNUESD joint_vel_l1
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        # 主要作用是动态地给对象的实例（在这个上下文中是 self）添加一个新的属性或更新一个已存在的属性。
        # set attribute（属性）
        setattr(self, attr_name, rew_term)

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # UNUESD action_l2

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "threshold": 1.0},
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Others
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "threshold": 0.5,
        },
    )

    foot_contact = RewTerm(
        func=mdp.foot_contact,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    base_height_rough_l2 = RewTerm(
        func=mdp.base_height_rough_l2,
        weight=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.0,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )

    joint_power = RewTerm(func=mdp.joint_power, weight=0)

    stand_still_when_zero_command = RewTerm(
        func=mdp.stand_still_when_zero_command,
        weight=0,
        params={"command_name": "base_velocity"},
    )

    # 在奖励配置中
    arm_move_with_ankle = RewTerm(
        func=mdp.arm_swing_coordination,
        weight=0.01,
        params={
            "velocity_scale":0.5,
            "coordination_gain":0.8,
            "std":0.3,
            "arm_joint_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder.*"]),
            "leg_sensor_cfg":SceneEntityCfg("contact_forces", body_names=[".*ankle.*"]),
            "command_name": "base_velocity",
        },
    )

    # 在奖励配置中
    arm_move = RewTerm(
        func=mdp.simple_arm_swing_reward,  # 指向简化版函数
        weight=0.005,  # 建议更小的权重（因奖励计算更直接）
        params={
            "arm_joint_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*shoulder.*"]  # 匹配所有肩关节
                # joint_names=[".*shoulder_pitch"]  # 匹配所有肩关节
            ),
            "velocity_scale": 1  # 可调节参数（原1.5可能过大）
        }
    )    

    torso_penalty = RewTerm(
        func=mdp.penalize_torso_twist,
        weight=-0.01,  # 负权重
        params={
            "torso_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=["torso"]  # 根据实际URDF调整
            ),
            "velocity_scale": 0.5,
            "max_penalty": 1.5
        }
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=4)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        # If the weight of rewards is 0, set rewards to None
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


if __name__ == "__main__":
    # run the environment
    print('test the environment')