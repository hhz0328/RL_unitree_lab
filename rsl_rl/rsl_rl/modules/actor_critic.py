# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
Actor-Critic 网络模块

该模块实现了标准的 Actor-Critic 架构，用于强化学习中的策略梯度方法。
主要功能包括：
- 策略网络 (Actor)：生成动作分布
- 价值网络 (Critic)：评估状态价值
- 动作噪声管理：支持标量和对数标准差
- 动作分布计算：基于正态分布

核心组件：
- Actor MLP：多层感知机策略网络
- Critic MLP：多层感知机价值网络
- 动作分布：基于均值和标准差的正态分布
- 噪声参数：可学习的动作标准差

应用场景：
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- 其他基于策略梯度的强化学习算法
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络类
    
    该类实现了标准的 Actor-Critic 架构，包含策略网络和价值网络。
    Actor 网络负责生成动作分布，Critic 网络负责评估状态价值。
    
    主要特性：
    - 支持自定义网络架构（隐藏层维度）
    - 可配置的激活函数
    - 可学习的动作噪声参数
    - 基于正态分布的动作采样
    - 支持训练和推理两种模式
    """
    is_recurrent = False  # 标记为非循环网络

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        """
        初始化 Actor-Critic 网络
        
        Args:
            num_actor_obs: Actor 网络的观察值维度
            num_critic_obs: Critic 网络的观察值维度
            num_actions: 动作空间维度
            actor_hidden_dims: Actor 网络隐藏层维度列表
            critic_hidden_dims: Critic 网络隐藏层维度列表
            activation: 激活函数类型
            init_noise_std: 初始动作噪声标准差
            noise_std_type: 噪声标准差类型 ("scalar" 或 "log")
            **kwargs: 其他参数（会被忽略）
        """
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # 解析激活函数
        activation = resolve_nn_activation(activation)

        # 设置网络输入维度
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # 构建Policy网络 (Actor)
        actor_layers = []
        # 输入层
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        # 隐藏层
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                # 输出层：生成动作均值
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                # 隐藏层：线性层 + 激活函数
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # 构建Value function网络 (Critic)
        critic_layers = []
        # 输入层
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        # 隐藏层
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                # 输出层：生成状态价值（标量）
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                # 隐藏层：线性层 + 激活函数
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # 打印网络结构信息
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # 配置Action动作噪声参数
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            # 标量标准差：直接学习标准差值
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            # 对数标准差：学习对数值，确保标准差为正
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # 动作分布（在 update_distribution 中填充）
        self.distribution = None
        # 禁用参数验证以提高速度
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        """
        使用正交初始化权重（目前未使用）
        
        Args:
            sequential: 顺序网络模块
            scales: 每层的缩放因子列表
        """
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """
        重置网络状态（非循环网络，无需重置）
        
        Args:
            dones: 完成标志（在此类中不使用）
        """
        pass

    def forward(self):
        """
        前向传播（未实现，使用具体的方法如 act 或 evaluate）
        
        Raises:
            NotImplementedError: 直接调用 forward 方法未实现
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        """动作分布的均值"""
        return self.distribution.mean

    @property
    def action_std(self):
        """动作分布的标准差"""
        return self.distribution.stddev

    @property
    def entropy(self):
        """动作分布的熵（用于探索）"""
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """
        更新动作分布
        
        Args:
            observations: 观察值张量
            
        Raises:
            ValueError: 未知的标准差类型
        """
        # 计算动作均值
        mean = self.actor(observations)
        # 计算动作标准差
        if self.noise_std_type == "scalar":
            # 标量标准差：直接扩展
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            # 对数标准差：取指数后扩展
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # 创建正态分布
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """
        生成动作（训练模式）
        
        Args:
            observations: 观察值张量
            **kwargs: 其他参数
            
        Returns:
            采样的动作张量
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """
        计算动作的对数概率
        
        Args:
            actions: 动作张量
            
        Returns:
            动作的对数概率张量
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """
        生成动作（推理模式）
        
        Args:
            observations: 观察值张量
            
        Returns:
            动作均值张量（无噪声）
        """
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """
        评估状态价值
        
        Args:
            critic_observations: Critic 网络的观察值张量
            **kwargs: 其他参数
            
        Returns:
            状态价值张量
        """
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """
        加载 Actor-Critic 模型的参数
        
        Args:
            state_dict (dict): 模型的状态字典
            strict (bool): 是否严格匹配状态字典的键
            
        Returns:
            bool: 是否恢复之前的训练。此标志被 OnPolicyRunner 的 load() 函数使用，
                 用于确定如何加载其他参数（例如，用于策略蒸馏）
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
