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

#  Copyright (c) 2020 Preferred Networks, Inc.

"""
标准化器模块

该模块实现了用于强化学习的各种标准化器，用于稳定训练和提高学习效率。
主要功能包括：
- 经验标准化：基于经验值标准化均值和方差
- 折扣变化标准化：基于折扣奖励的奖励标准化
- 折扣平均：计算折扣奖励的平均值

核心组件：
- EmpiricalNormalization：经验标准化器
- EmpiricalDiscountedVariationNormalization：折扣变化标准化器
- DiscountedAverage：折扣平均计算器

应用场景：
- 观察值标准化：稳定神经网络训练
- 奖励标准化：处理非平稳奖励函数
- 价值函数学习：提高学习速度和稳定性
- 策略梯度方法：减少方差，提高收敛性
"""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """
    经验标准化器
    
    基于经验值标准化输入数据的均值和方差，用于稳定神经网络训练。
    通过在线更新均值和方差统计量，实现数据的标准化处理。
    
    主要特性：
    - 在线学习：动态更新统计量
    - 数值稳定：添加小量防止除零
    - 可逆操作：支持标准化和反标准化
    - 批量处理：支持批量数据标准化
    """

    def __init__(self, shape, eps=1e-2, until=None):
        """
        初始化经验标准化器
        
        Args:
            shape (int or tuple of int): 输入值的形状（除批次轴外）
            eps (float): 用于数值稳定性的小量，默认为 1e-2
            until (int or None): 如果指定此参数，模块将学习输入值直到批次大小总和超过此值
        """
        super().__init__()
        # 设置数值稳定性参数
        self.eps = eps
        # 设置学习截止条件
        self.until = until
        
        # 注册缓冲区：均值和方差统计量
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))  # 均值
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))    # 方差
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))    # 标准差
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))  # 样本计数

    @property
    def mean(self):
        """获取当前均值"""
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        """获取当前标准差"""
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """
        基于经验值标准化输入数据的均值和方差
        
        Args:
            x (ndarray or Variable): 输入值
            
        Returns:
            ndarray or Variable: 标准化后的输出值
        """
        # 训练模式下更新统计量
        if self.training:
            self.update(x)
        # 标准化：减去均值，除以标准差
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """
        学习输入值而不计算其输出值
        
        使用在线更新算法更新均值和方差统计量
        """
        # 检查是否达到学习截止条件
        if self.until is not None and self.count >= self.until:
            return

        # 获取当前批次大小并更新计数
        count_x = x.shape[0]
        self.count += count_x
        # 计算学习率（新样本权重）
        rate = count_x / self.count

        # 计算当前批次的统计量
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)  # 方差
        mean_x = torch.mean(x, dim=0, keepdim=True)                 # 均值
        
        # 更新均值：增量更新
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        
        # 更新方差：考虑均值变化的增量更新
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        
        # 更新标准差
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        """
        反标准化操作
        
        Args:
            y: 标准化后的值
            
        Returns:
            反标准化后的原始值
        """
        # 反标准化：乘以标准差，加上均值
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """
    经验折扣变化标准化器
    
    来自 Pathak 的 PPO 大规模研究的奖励标准化方法。
    
    由于奖励函数是非平稳的，标准化奖励的尺度有助于价值函数快速学习。
    通过将奖励除以折扣奖励总和的运行标准差估计来实现标准化。
    
    主要特性：
    - 处理非平稳奖励：适应奖励函数的变化
    - 折扣奖励标准化：基于折扣奖励的统计量
    - 在线更新：动态调整标准化参数
    - 提高学习速度：稳定价值函数学习
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        """
        初始化经验折扣变化标准化器
        
        Args:
            shape: 输入形状
            eps: 数值稳定性参数
            gamma: 折扣因子
            until: 学习截止条件
        """
        super().__init__()

        # 创建经验标准化器和折扣平均计算器
        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        """
        前向传播：标准化奖励
        
        Args:
            rew: 输入奖励
            
        Returns:
            标准化后的奖励
        """
        if self.training:
            # 更新折扣奖励
            avg = self.disc_avg.update(rew)

            # 基于折扣奖励更新统计量
            self.emp_norm.update(avg)

        # 如果标准差大于 0，进行标准化；否则返回原始奖励
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    """
    折扣平均计算器
    
    计算奖励的折扣平均值，用于奖励标准化。
    
    折扣平均值定义为：
    
    .. math::
        \\bar{R}_t = \\gamma \\bar{R}_{t-1} + r_t
    
    其中 γ 是折扣因子，r_t 是当前奖励。
    
    Args:
        gamma (float): 折扣因子
    """

    def __init__(self, gamma):
        """
        初始化折扣平均计算器
        
        Args:
            gamma: 折扣因子
        """
        self.avg = None  # 折扣平均值
        self.gamma = gamma  # 折扣因子

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """
        更新折扣平均值
        
        Args:
            rew: 当前奖励张量
            
        Returns:
            更新后的折扣平均值
        """
        if self.avg is None:
            # 初始化：第一个奖励
            self.avg = rew
        else:
            # 更新：折扣平均值 = γ * 上一时刻平均值 + 当前奖励
            self.avg = self.avg * self.gamma + rew
        return self.avg
