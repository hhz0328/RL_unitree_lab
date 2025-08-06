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
循环 Actor-Critic 网络模块

该模块实现了基于循环神经网络的 Actor-Critic 架构，用于处理序列数据和部分可观察环境。
主要功能包括：
- 循环策略网络 (Actor)：处理序列观察值并生成动作分布
- 循环价值网络 (Critic)：处理序列观察值并评估状态价值
- 记忆组件：LSTM 或 GRU 循环神经网络
- 隐藏状态管理：动态重置和状态传递

核心组件：
- Actor 记忆网络：处理策略相关的序列数据
- Critic 记忆网络：处理价值评估相关的序列数据
- 基础 Actor-Critic：继承自标准 Actor-Critic 网络

应用场景：
- 部分可观察马尔可夫决策过程 (POMDP)
- 需要记忆历史信息的强化学习任务
- 序列决策问题
- 具有时间依赖性的环境
"""

from __future__ import annotations

import warnings

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class ActorCriticRecurrent(ActorCritic):
    """
    循环 Actor-Critic 网络类
    
    该类继承自标准的 ActorCritic 类，添加了循环神经网络组件来处理序列数据。
    通过记忆网络处理观察值序列，然后传递给基础的 Actor-Critic 网络进行动作生成和价值评估。
    
    主要特性：
    - 继承标准 Actor-Critic 的所有功能
    - 添加循环神经网络记忆组件
    - 支持 LSTM 和 GRU 两种循环网络类型
    - 独立的 Actor 和 Critic 记忆网络
    - 隐藏状态的动态管理
    """
    is_recurrent = True  # 标记为循环网络

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        """
        初始化循环 Actor-Critic 网络
        
        Args:
            num_actor_obs: Actor 网络的观察值维度
            num_critic_obs: Critic 网络的观察值维度
            num_actions: 动作空间维度
            actor_hidden_dims: Actor 网络隐藏层维度列表
            critic_hidden_dims: Critic 网络隐藏层维度列表
            activation: 激活函数类型
            rnn_type: 循环神经网络类型 ("lstm" 或 "gru")
            rnn_hidden_dim: RNN 隐藏状态维度
            rnn_num_layers: RNN 层数
            init_noise_std: 初始动作噪声标准差
            **kwargs: 其他参数（会被忽略）
        """
        # 处理已弃用的参数名
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # 仅在新参数为默认值时覆盖
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        # 检查并忽略未预期的参数
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        # 初始化父类（Actor-Critic），使用 RNN 隐藏维度作为输入维度
        super().__init__(
            num_actor_obs=rnn_hidden_dim,      # Actor 输入维度为 RNN 隐藏维度
            num_critic_obs=rnn_hidden_dim,     # Critic 输入维度为 RNN 隐藏维度
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        # 解析激活函数
        activation = resolve_nn_activation(activation)

        # 创建 Actor 和 Critic 的记忆网络
        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        # 打印记忆网络结构信息
        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        """
        重置记忆网络的隐藏状态
        
        Args:
            dones: 完成标志张量，指示哪些环境已完成
        """
        # 重置 Actor 和 Critic 记忆网络的隐藏状态
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        """
        生成动作（训练模式）
        
        Args:
            observations: 观察值张量
            masks: 轨迹掩码（用于批量模式）
            hidden_states: 外部提供的隐藏状态（用于批量模式）
            
        Returns:
            采样的动作张量
        """
        # 通过 Actor 记忆网络处理观察值
        input_a = self.memory_a(observations, masks, hidden_states)
        # 调用父类的 act 方法生成动作
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        """
        生成动作（推理模式）
        
        Args:
            observations: 观察值张量
            
        Returns:
            动作均值张量（无噪声）
        """
        # 通过 Actor 记忆网络处理观察值
        input_a = self.memory_a(observations)
        # 调用父类的 act_inference 方法生成动作
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """
        评估状态价值
        
        Args:
            critic_observations: Critic 网络的观察值张量
            masks: 轨迹掩码（用于批量模式）
            hidden_states: 外部提供的隐藏状态（用于批量模式）
            
        Returns:
            状态价值张量
        """
        # 通过 Critic 记忆网络处理观察值
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        # 调用父类的 evaluate 方法评估价值
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        """
        获取记忆网络的隐藏状态
        
        Returns:
            tuple: (Actor 隐藏状态, Critic 隐藏状态)
        """
        return self.memory_a.hidden_states, self.memory_c.hidden_states
