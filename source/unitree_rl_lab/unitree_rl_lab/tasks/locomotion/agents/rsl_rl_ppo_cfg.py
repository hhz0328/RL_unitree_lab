# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
# 算法执行器注册位置

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """基础PPO运行器配置类。
    
    这个类定义了PPO（Proximal Policy Optimization）算法的所有配置参数，
    包括训练参数、网络结构、算法超参数等。
    
    继承自RslRlOnPolicyRunnerCfg，专门用于腿式机器人的强化学习训练。
    """
    
    # ==================== 训练参数 ====================
    num_steps_per_env = 24          # 每个环境收集的步数
    max_iterations = 50000          # 最大训练迭代次数
    save_interval = 100             # 模型保存间隔（每100次迭代保存一次）
    experiment_name = ""            # 实验名称（与任务名称相同）
    empirical_normalization = False # 是否使用经验归一化
    
    # ==================== 策略网络配置 ====================
    policy = RslRlPpoActorCriticCfg(
        # 动作噪声标准差：用于探索的初始噪声大小
        # 较大的值会增加探索，较小的值会减少探索
        init_noise_std=1.0,
        
        # Actor网络隐藏层维度：策略网络的结构
        # [512, 256, 128] 表示3层隐藏层，神经元数分别为512、256、128
        # 较大的网络容量可以学习更复杂的策略
        actor_hidden_dims=[512, 256, 128],
        
        # Critic网络隐藏层维度：价值网络的结构
        # 通常与Actor网络结构相同，用于估计状态价值
        critic_hidden_dims=[512, 256, 128],
        
        # 激活函数：ELU（Exponential Linear Unit）
        # ELU在负值区域有非零梯度，有助于缓解梯度消失问题
        activation="elu",
    )
    
    # ==================== PPO算法配置 ====================
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值损失系数：价值函数损失的权重
        # 控制价值学习和策略学习的平衡
        value_loss_coef=1.0,
        
        # 是否使用裁剪的价值损失：防止价值函数更新过大
        # 提高训练稳定性
        use_clipped_value_loss=True,
        
        # 策略裁剪参数：PPO的核心超参数
        # 限制新旧策略的差异，防止策略更新过大
        # 典型值范围：0.1-0.3
        clip_param=0.2,
        
        # 熵系数：鼓励探索的权重
        # 较大的值会增加探索，较小的值会减少探索
        # 典型值范围：0.001-0.1
        entropy_coef=0.01,
        
        # 学习轮数：每次更新时对同一批数据进行多少次学习
        # 较多的轮数可以提高样本利用率，但可能导致过拟合
        num_learning_epochs=5,
        
        # 小批量数量：将数据分成多少个小批量进行学习
        # 影响梯度更新的频率和稳定性
        num_mini_batches=4,
        
        # 学习率：策略和价值网络的更新步长
        # 较大的学习率会加快收敛但可能不稳定
        # 较小的学习率会更稳定但收敛较慢
        learning_rate=1.0e-3,
        
        # 学习率调度：自适应学习率调整
        # "adaptive"表示根据KL散度自动调整学习率
        schedule="adaptive",
        
        # 折扣因子：未来奖励的衰减系数
        # 接近1表示重视长期奖励，接近0表示重视短期奖励
        # 典型值：0.95-0.99
        gamma=0.99,
        
        # GAE参数：广义优势估计的λ参数
        # 控制偏差和方差的权衡
        # 典型值：0.9-0.99
        lam=0.95,
        
        # 期望KL散度：目标策略更新幅度
        # 用于自适应学习率调整
        # 典型值：0.01-0.02
        desired_kl=0.01,
        
        # 最大梯度范数：梯度裁剪的阈值
        # 防止梯度爆炸，提高训练稳定性
        max_grad_norm=1.0,
    )
