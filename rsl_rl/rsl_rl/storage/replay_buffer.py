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
回放缓冲区 (Replay Buffer) 模块

本模块实现了固定大小的回放缓冲区，用于存储经验元组。
回放缓冲区是强化学习中的重要组件，特别是在离策略算法中，
用于打破样本间的相关性并提高训练效率。

主要功能：
- 固定大小的循环缓冲区
- 存储状态和下一状态对
- 随机采样生成器
- 自动覆盖旧数据

核心特性：
- 循环覆盖：当缓冲区满时，新数据会覆盖最旧的数据
- 随机采样：支持随机采样用于训练
- 设备管理：支持CPU和GPU设备
- 批量插入：支持批量插入多个状态对

适用场景：
- 离策略强化学习算法
- 经验回放
- 数据缓冲和采样
- AMP训练中的专家数据存储
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    固定大小的回放缓冲区类
    
    该类实现了固定大小的循环缓冲区，用于存储经验元组。
    当缓冲区满时，新数据会覆盖最旧的数据，实现循环覆盖。
    
    属性:
        states: 当前状态张量
        next_states: 下一状态张量
        buffer_size: 缓冲区大小
        device: 计算设备
        step: 当前插入位置
        num_samples: 当前样本数量
    """

    def __init__(self, obs_dim, buffer_size, device):
        """
        初始化回放缓冲区
        
        Args:
            obs_dim: 观察值维度
            buffer_size: 缓冲区最大大小
            device: 计算设备（CPU或GPU）
        """
        # 初始化状态存储张量
        self.states = torch.zeros(buffer_size, obs_dim).to(device)      # 当前状态张量
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device) # 下一状态张量
        self.buffer_size = buffer_size  # 缓冲区大小
        self.device = device            # 计算设备

        # 初始化索引和计数器
        self.step = 0          # 当前插入位置（循环索引）
        self.num_samples = 0   # 当前样本数量

    def insert(self, states, next_states):
        """
        插入新的状态对到缓冲区
        
        该方法将新的状态和下一状态对插入到缓冲区中。
        如果缓冲区已满，新数据会覆盖最旧的数据。
        
        Args:
            states: 当前状态张量，形状为 [num_states, obs_dim]
            next_states: 下一状态张量，形状为 [num_states, obs_dim]
        """
        # 获取要插入的状态数量
        num_states = states.shape[0]
        
        # 计算插入的起始和结束索引
        start_idx = self.step
        end_idx = self.step + num_states
        
        # 检查是否需要循环覆盖
        if end_idx > self.buffer_size:
            # 需要循环覆盖：数据跨越缓冲区边界
            # 填充缓冲区末尾部分
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
            # 填充缓冲区开头部分
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            # 不需要循环覆盖：数据完全在缓冲区范围内
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        # 更新样本数量和插入位置
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size  # 循环索引

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        前馈生成器
        
        生成随机采样的mini-batch数据，用于训练。
        每次调用都会随机采样指定数量的状态对。
        
        Args:
            num_mini_batch: mini-batch的数量
            mini_batch_size: 每个mini-batch的大小
            
        Yields:
            tuple: (states, next_states) 状态对张量
        """
        for _ in range(num_mini_batch):
            # 随机采样索引
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            # 返回采样的状态对
            yield (self.states[sample_idxs].to(self.device), self.next_states[sample_idxs].to(self.device))
