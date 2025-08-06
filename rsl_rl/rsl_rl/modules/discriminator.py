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
判别器网络模块

该模块实现了用于对抗性运动先验 (AMP) 的判别器神经网络，用于奖励预测和运动质量评估。
主要功能包括：
- 运动质量判别：区分专家运动和生成运动
- AMP 奖励计算：基于判别器输出计算对抗性奖励
- 梯度惩罚：正则化判别器训练
- 奖励插值：结合任务奖励和 AMP 奖励

核心组件：
- 主干网络 (Trunk)：多层感知机处理输入特征
- AMP 线性层：生成判别器输出
- 梯度惩罚计算：防止判别器过拟合
- 奖励预测：计算最终的 AMP 奖励

应用场景：
- AMP (Adversarial Motion Priors) 训练
- 运动模仿学习
- 自然运动生成
- 运动质量评估
"""

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """
    对抗性运动先验 (AMP) 判别器神经网络
    
    该类实现了用于 AMP 奖励预测的判别器网络，能够区分专家运动和生成运动的质量，
    并基于判别结果计算对抗性奖励来指导策略学习。
    
    Args:
        input_dim (int): 输入特征向量维度（连接的状态和下一状态）
        amp_reward_coef (float): AMP 奖励的缩放系数
        hidden_layer_sizes (list[int]): MLP 主干网络的隐藏层大小
        device (torch.device): 运行模型的设备（CPU 或 GPU）
        task_reward_lerp (float, optional): AMP 奖励和任务奖励之间的插值因子，默认为 0.0（仅 AMP 奖励）
    
    Attributes:
        trunk (nn.Sequential): 处理输入特征的 MLP 层
        amp_linear (nn.Linear): 生成判别器输出的最终线性层
        task_reward_lerp (float): 结合奖励的插值因子
    """

    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        """
        初始化判别器网络
        
        Args:
            input_dim: 输入特征维度
            amp_reward_coef: AMP 奖励系数
            hidden_layer_sizes: 隐藏层维度列表
            device: 计算设备
            task_reward_lerp: 任务奖励插值因子
        """
        super().__init__()

        # 设置设备和输入维度
        self.device = device
        self.input_dim = input_dim

        # 设置 AMP 奖励系数
        self.amp_reward_coef = amp_reward_coef
        
        # 构建主干网络 (MLP)
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))  # 线性层
            amp_layers.append(nn.ReLU())                           # ReLU 激活函数
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        
        # 构建 AMP 线性输出层
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        # 设置为训练模式
        self.trunk.train()
        self.amp_linear.train()

        # 设置任务奖励插值因子
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        """
        判别器网络的前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            torch.Tensor: 判别器输出逻辑值，形状为 (batch_size, 1)
        """
        # 通过主干网络处理输入
        h = self.trunk(x)
        # 通过 AMP 线性层生成判别器输出
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """
        计算专家数据的梯度惩罚，用于正则化判别器
        
        Args:
            expert_state (torch.Tensor): 专家状态批次
            expert_next_state (torch.Tensor): 专家下一状态批次
            lambda_ (float, optional): 梯度惩罚系数，默认为 10
            
        Returns:
            torch.Tensor: 标量梯度惩罚损失
        """
        # 连接专家状态和下一状态
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        # 计算判别器输出
        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        
        # 计算梯度
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # 强制梯度范数接近 0（Wasserstein GAN 的梯度惩罚）
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """
        预测 AMP 奖励，给定当前状态和下一状态，可选择与任务奖励插值
        
        Args:
            state (torch.Tensor): 当前状态张量
            next_state (torch.Tensor): 下一状态张量
            task_reward (torch.Tensor): 任务特定奖励张量
            normalizer (optional): 标准化器对象，用于在预测前标准化输入状态
            
        Returns:
            tuple:
                - reward (torch.Tensor): 预测的 AMP 奖励（可选择插值），形状为 (batch_size,)
                - d (torch.Tensor): 原始判别器输出逻辑值，形状为 (batch_size, 1)
        """
        with torch.no_grad():
            # 设置为评估模式
            self.eval()
            
            # 如果提供了标准化器，对输入状态进行标准化
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            # 计算判别器输出
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            
            # 计算 AMP 奖励：基于判别器输出和期望值 1 的差异
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            
            # 如果设置了任务奖励插值，进行奖励插值
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            
            # 恢复训练模式
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        """
        在判别器奖励和任务奖励之间进行线性插值
        
        Args:
            disc_r (torch.Tensor): 判别器奖励
            task_r (torch.Tensor): 任务奖励
            
        Returns:
            torch.Tensor: 插值后的奖励
        """
        # 线性插值：r = (1 - lerp) * disc_r + lerp * task_r
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
