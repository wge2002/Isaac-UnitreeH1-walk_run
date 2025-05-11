
from task import mdp
from task.VelocityEnv import LocomotionVelocityRoughEnvCfg

from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
# use cloud assets
from omni.isaac.lab_assets import H1_MINIMAL_CFG  # isort: skip


@configclass
class UnitreeH1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    _run_disable_zero_weight_rewards = True

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree-h1
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # ------------------------------Observations------------------------------
        # 这里，base_lin_vel与height_scan这两行都是注释掉了。但是会有报错，不注释后报错消失。
        # 运行play的checkpoint时候，flat注释掉，rough不要注释。否则电脑想要的tensor的shape和checkpoint的shape不一样。
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        # ------------------------------Actions------------------------------

        # ------------------------------Events------------------------------
        # self.events.reset_base_amp = None
        # self.events.reset_robot_joints_amp = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_parameters = None

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = -200

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.05 # 智能体在xy平面上的角速度越大，它将受到越多的惩罚
        self.rewards.flat_orientation_l2.weight = -1.0 # 鼓励智能体保持物体的平坦姿态。如果物体的姿态偏离平坦状态，它将受到惩罚。
        self.rewards.base_height_l2.weight = 0
        self.rewards.body_lin_acc_l2.weight = 0

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = 0
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.create_joint_deviation_l1_rewterm( # 用于惩罚指定关节（髋关节的偏航和滚动、肩关节和肘关节）的偏离程度
            "joint_deviation_other_l1", -0.2, [".*_hip_yaw", ".*_hip_roll", ".*_shoulder_.*", ".*_elbow"]
        )
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_torso_l1", -0.1, ["torso"])
        self.rewards.joint_pos_limits.weight = -1.0 # 这意味着如果关节位置超出了预设的限制，智能体将受到较大的惩罚。
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = [".*_ankle"]
        self.rewards.joint_vel_limits.weight = 0

        # Action penalties
        self.rewards.applied_torque_limits.weight = 0
        self.rewards.action_rate_l2.weight = -0.005 # 意味着智能体在连续动作之间的变化率（即动作速率）越大，它将受到越多的惩罚。
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = 0

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0 # 智能体越接近目标线性速度，将获得越多的奖励
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # Others
        self.rewards.feet_air_time.weight = 0.25 # 奖励一次一只脚在空中
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*ankle_link"]
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.foot_contact.weight = 0
        self.rewards.base_height_rough_l2.weight = 0
        self.rewards.feet_slide.weight = -0.25 # 惩罚脚步的滑行
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*ankle_link"]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [".*ankle_link"]
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_when_zero_command.weight = 0

        # arm move
        self.rewards.arm_move.weight = 0.01
        self.rewards.arm_move_with_ankle.weight = 0
        self.rewards.torso_penalty.weight = -0.01
        # If the weight of rewards is 0, set rewards to None
        if self._run_disable_zero_weight_rewards:
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [".*torso_link"]

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
