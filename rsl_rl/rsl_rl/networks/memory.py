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
记忆网络模块

该模块实现了用于强化学习的循环神经网络记忆组件，支持：
- LSTM 和 GRU 两种循环神经网络类型
- 批量模式训练和推理模式
- 隐藏状态的动态重置和分离
- 支持多环境并行训练

主要功能：
- 序列数据处理和记忆保持
- 隐藏状态管理（重置、分离）
- 批量训练和单步推理的切换
- 支持 LSTM 的双隐藏状态（h, c）

应用场景：
- 部分可观察马尔可夫决策过程 (POMDP)
- 需要记忆历史信息的强化学习任务
- 序列决策问题
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories


class Memory(torch.nn.Module):
    """
    记忆网络类
    
    该类实现了基于循环神经网络的记忆组件，用于强化学习中的序列数据处理。
    支持 LSTM 和 GRU 两种循环神经网络类型，能够处理批量训练和单步推理两种模式。
    
    主要特性：
    - 支持 LSTM 和 GRU 循环神经网络
    - 自动管理隐藏状态
    - 支持批量模式训练和推理模式切换
    - 动态隐藏状态重置和分离
    """
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        """
        初始化记忆网络
        
        Args:
            input_size: 输入特征维度
            type: 循环神经网络类型 ("lstm" 或 "gru")
            num_layers: RNN 层数
            hidden_size: 隐藏状态维度
        """
        super().__init__()
        # RNN
        # 根据类型选择循环神经网络
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        # 初始化隐藏状态为 None
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        """
        前向传播
        
        Args:
            input: 输入张量
            masks: 轨迹掩码（用于批量模式）
            hidden_states: 外部提供的隐藏状态（用于批量模式）
            
        Returns:
            处理后的输出张量
            
        Raises:
            ValueError: 批量模式下未提供隐藏状态
        """
        # 判断是否为批量模式（通过 masks 参数判断）
        batch_mode = masks is not None
        if batch_mode:
            # 批量模式：需要保存的隐藏状态（用于策略更新）
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            # 使用外部提供的隐藏状态进行前向传播
            out, _ = self.rnn(input, hidden_states)
            # 使用掩码去除填充的轨迹
            out = unpad_trajectories(out, masks)
        else:
            # 推理/蒸馏模式：使用上一步的隐藏状态
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        """
        重置隐藏状态
        
        Args:
            dones: 完成标志张量，指示哪些环境已完成
            hidden_states: 自定义隐藏状态（可选）
            
        Note:
            - 如果 dones 为 None，重置所有隐藏状态
            - 如果 dones 不为 None，只重置已完成环境的隐藏状态
            - 支持 LSTM 的双隐藏状态（h, c）和 GRU 的单隐藏状态
        """
        if dones is None:  # 重置所有隐藏状态
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # 重置已完成环境的隐藏状态
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # LSTM 的情况（元组）
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:  # GRU 的情况（单个张量）
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        """
        分离隐藏状态的梯度
        
        Args:
            dones: 完成标志张量，指示哪些环境已完成
            
        Note:
            - 如果 dones 为 None，分离所有隐藏状态的梯度
            - 如果 dones 不为 None，只分离已完成环境的隐藏状态梯度
            - 用于防止梯度在已完成环境中传播，提高训练效率
        """
        if self.hidden_states is not None:
            if dones is None:  # 分离所有隐藏状态的梯度
                if isinstance(self.hidden_states, tuple):  # LSTM 的情况（元组）
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:  # GRU 的情况（单个张量）
                    self.hidden_states = self.hidden_states.detach()
            else:  # 分离已完成环境的隐藏状态梯度
                if isinstance(self.hidden_states, tuple):  # LSTM 的情况（元组）
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:  # GRU 的情况（单个张量）
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()
