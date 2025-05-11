
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UnitreeH1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100 # 模型保存的间隔迭代次数。即每100次迭代后，当前的模型将被保存。
    experiment_name = "h1_rough" # 实验的名称，用于区分不同的实验或训练过程。这通常用于日志记录和模型保存时的命名。
    empirical_normalization = False # 是否使用经验归一化。经验归一化是一种数据预处理技术，可以在训练过程中动态调整输入数据的统计量（如均值和标准差），以加速收敛和提高稳定性。
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, # 初始噪声标准差，用于在训练初期增加探索性。
        actor_hidden_dims=[512, 256, 128], # （Actor Network，即策略网络）的隐藏层维度。这里使用了一个三层网络，每层的神经元数逐渐减少。
        critic_hidden_dims=[512, 256, 128], # （Critic Network，即值函数网络）的隐藏层维度
        activation="elu", # 激活函数，这里使用的是"elu"，一种常用于深度学习的非线性激活函数。
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # 值损失系数，用于控制值损失在总损失中的权重。
        use_clipped_value_loss=True, # 是否使用裁剪值损失。PPO通过裁剪策略更新前后的值函数差异来限制策略更新幅度，以提高训练的稳定性。
        clip_param=0.2, # 裁剪参数的值，用于控制值函数差异裁剪的程度
        entropy_coef=0.01, # 熵系数，用于鼓励探索。较高的熵系数会使策略更加随机。
        num_learning_epochs=5, # 每个迭代中优化器的更新次数。
        num_mini_batches=4, #  将收集的数据分成多少个mini-batch进行训练。
        learning_rate=1.0e-3, # 学习率，控制参数更新的步长。
        schedule="adaptive", # 学习率调度器，这里设置为"adaptive"，意味着学习率可能会根据训练过程中的某些指标进行自适应调整。
        gamma=0.99, # 折扣因子，用于计算未来奖励的当前价值。
        lam=0.95, # GAE（Generalized Advantage Estimation）中的lambda参数，用于平衡值函数误差和优势函数误差。
        desired_kl=0.01, # 期望的KL散度值，用于PPO的KL散度惩罚项，以控制策略更新的幅度。
        max_grad_norm=1.0, # 梯度裁剪的最大范数，用于防止梯度爆炸。
    )


@configclass
class UnitreeH1FlatPPORunnerCfg(UnitreeH1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1000
        self.experiment_name = "h1_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]