from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation

from dataclasses import MISSING

from omni.isaac.lab.controllers import DifferentialIKControllerCfg  # noqa: F401
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    关节运动的基类。
    此动作项使用仿射变换（缩放和偏移）对原始动作进行预处理。这些变换可以配置为应用于关节的一个子集。
    从数学上讲，动作项定义为：
    ..数学：
    \text{action}=\text{offset}+\text{scaling}\times\text{input action}
    其中
    math:`\text{action}`是发送到关节驱动关节的动作
    math:`\text{offset}`是应用于输入动作的偏移
    math:`\text{1scaling}`是施加到输入动作的缩放
    math:1\text{input-action}`是用户的输入动作。
    基于上述，这种动作转换确保了输入和输出动作处于相同的单位和维度。
    然后，该动作项的子类可以将输出动作映射到关节关节的特定所需命令（例如位置、速度等）。
    """

    cfg: JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action.  offset 偏移？"""
    _clip: dict[str, tuple] | None = None
    """The clip applied to the input action.  clip 裁切?"""

    def __init__(self, cfg: JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        # print(self._joint_ids, self._joint_names)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse clip
        if cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = cfg.clip
            else:
                raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are dict.")
            

        # parse scale
        if isinstance(cfg.scale, (float, int)): # 如果cfg.scale是浮点数（float）或整数（int）
            self._scale = float(cfg.scale) # 将其转换为浮点数并赋值给self._scale
        elif isinstance(cfg.scale, dict): # 如果cfg.scale是字典（dict）
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            # 根据_joint_names（关节名称）解析字典，获取匹配的索引列表（index_list）和值列表（value_list）。
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            # 最后，将这些值按索引列表分配到self._scale张量的对应位置。
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        

        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # clip actions，裁切动作
        if self._clip is not None:
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self._clip, self._joint_names)
            for index in range(len(index_list)):
                min_value, max_value = value_list[index]
                self._raw_actions[:, index_list[index]].clip_(min_value, max_value)
        
        # apply the affine transformations,使用公式
        self._processed_actions = self._raw_actions * self._scale + self._offset  # offset和scale在init()当中处理的

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0 # 将所有raw_actions都设置为0


@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0.
       动作的偏移因子(正则表达式的浮点数或字典)。默认值为0.0。
    """
    clip: dict[str, tuple] | None = None
    """The clipping range for action. Defaults to None, in which case no clipping is applied.
       动作的剪切范围。默认为“无”，在这种情况下，不应用剪裁。
    """


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class JointVelocityActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = JointVelocityAction

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the articulation asset as offset.
    Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class JointEffortActionCfg(JointActionCfg):
    """Configuration for the joint effort action term.

    See :class:`JointEffortAction` for more details.
    """

    class_type: type[ActionTerm] = JointEffortAction



