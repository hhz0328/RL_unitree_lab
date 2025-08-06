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
随机网络蒸馏 (Random Network Distillation, RND) 模块

本模块实现了随机网络蒸馏算法，用于强化学习中的内在探索。
RND通过训练一个预测器网络来预测目标网络的输出，从而产生内在奖励来鼓励探索。

主要功能：
- 构建目标网络和预测器网络
- 计算内在奖励（预测误差）
- 支持状态和奖励的归一化
- 支持权重调度策略

参考文献：
Burda, Yuri, et al. "Exploration by random network distillation." 
arXiv preprint arXiv:1810.12894 (2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.normalizer import (
    EmpiricalDiscountedVariationNormalization,
    EmpiricalNormalization,
)
from rsl_rl.utils import resolve_nn_activation


class RandomNetworkDistillation(nn.Module):
    """
    随机网络蒸馏 (RND) 实现类
    
    该模块包含两个神经网络：
    - 目标网络 (Target Network): 固定权重的随机网络，用于生成目标嵌入
    - 预测器网络 (Predictor Network): 可训练的网络，用于预测目标网络的输出
    
    内在奖励通过两个网络输出之间的欧几里得距离计算。
    
    参考文献：
        Burda, Yuri, et al. "Exploration by random network distillation." 
        arXiv preprint arXiv:1810.12894 (2018).
    """

    def __init__(
        self,
        num_states: int,
        num_outputs: int,
        predictor_hidden_dims: list[int],
        target_hidden_dims: list[int],
        activation: str = "elu",
        weight: float = 0.0,
        state_normalization: bool = False,
        reward_normalization: bool = False,
        device: str = "cpu",
        weight_schedule: dict | None = None,
    ):
        """
        初始化 RND 模块
        
        如果 state_normalization 为 True，则使用经验归一化层对输入状态进行归一化。
        如果 reward_normalization 为 True，则使用经验折扣变化归一化层对内在奖励进行归一化。
        
        注意：
            如果预测器和目标网络的隐藏维度配置中为 -1，则使用状态数量作为隐藏维度。
        
        Args:
            num_states: 预测器和目标网络的输入状态数量
            num_outputs: 预测器和目标网络的输出维度（嵌入大小）
            predictor_hidden_dims: 预测器网络的隐藏层维度列表
            target_hidden_dims: 目标网络的隐藏层维度列表
            activation: 激活函数，默认为 "elu"
            weight: 内在奖励的缩放因子，默认为 0.0
            state_normalization: 是否对输入状态进行归一化，默认为 False
            reward_normalization: 是否对内在奖励进行归一化，默认为 False
            device: 使用的设备，默认为 "cpu"
            weight_schedule: RND权重参数的调度类型，默认为 None（使用常数权重）
                字典格式，包含以下键：
                - "mode": 权重调度类型
                    - "constant": 常数权重调度
                    - "step": 阶梯权重调度
                    - "linear": 线性权重调度
                
                对于 "step" 调度，需要以下参数：
                - "final_step": 权重达到最终值的步数
                - "final_value": 权重的最终值
                
                对于 "linear" 调度，需要以下参数：
                - "initial_step": 权重开始变化的步数
                - "final_step": 权重达到最终值的步数
                - "final_value": 权重的最终值
        """
        # 初始化父类
        super().__init__()

        # 存储参数
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.initial_weight = weight
        self.device = device
        self.state_normalization = state_normalization
        self.reward_normalization = reward_normalization

        # 输入状态归一化
        if state_normalization:
            # 使用经验归一化层，形状为状态维度，更新次数限制为1e8
            self.state_normalizer = EmpiricalNormalization(shape=[self.num_states], until=1.0e8).to(self.device)
        else:
            # 不使用归一化，直接返回输入
            self.state_normalizer = torch.nn.Identity()
        
        # 内在奖励归一化
        if reward_normalization:
            # 使用经验折扣变化归一化层，形状为空（标量），更新次数限制为1e8
            self.reward_normalizer = EmpiricalDiscountedVariationNormalization(shape=[], until=1.0e8).to(self.device)
        else:
            # 不使用归一化，直接返回输入
            self.reward_normalizer = torch.nn.Identity()

        # 更新计数器
        self.update_counter = 0

        # 解析权重调度器
        if weight_schedule is not None:
            self.weight_scheduler_params = weight_schedule
            # 根据调度模式获取对应的调度函数
            self.weight_scheduler = getattr(self, f"_{weight_schedule['mode']}_weight_schedule")
        else:
            self.weight_scheduler = None
        
        # 创建网络架构
        # 构建预测器网络
        self.predictor = self._build_mlp(num_states, predictor_hidden_dims, num_outputs, activation).to(self.device)
        # 构建目标网络
        self.target = self._build_mlp(num_states, target_hidden_dims, num_outputs, activation).to(self.device)

        # 设置目标网络为不可训练模式
        self.target.eval()

    def get_intrinsic_reward(self, rnd_state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算内在奖励
        
        注意：计数器按每次学习迭代的环境步数更新
        
        Args:
            rnd_state: 用于RND的状态张量
            
        Returns:
            tuple: (内在奖励张量, 归一化后的RND状态张量)
        """
        # 更新计数器（每次学习迭代的环境步数）
        self.update_counter += 1
        
        # 对RND状态进行归一化
        rnd_state = self.state_normalizer(rnd_state)
        
        # 从目标网络和预测器网络获取RND状态的嵌入
        target_embedding = self.target(rnd_state).detach()      # 目标网络嵌入（固定）
        predictor_embedding = self.predictor(rnd_state).detach() # 预测器网络嵌入（固定）
        
        # 计算内在奖励（两个嵌入之间的欧几里得距离）
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        
        # 对内在奖励进行归一化
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)

        # 检查权重调度
        if self.weight_scheduler is not None:
            # 使用调度器计算当前权重
            self.weight = self.weight_scheduler(step=self.update_counter, **self.weight_scheduler_params)
        else:
            # 使用初始权重
            self.weight = self.initial_weight
        
        # 缩放内在奖励
        intrinsic_reward *= self.weight

        return intrinsic_reward, rnd_state

    def forward(self, *args, **kwargs):
        """
        前向传播方法（未实现）
        
        抛出异常，因为RND模块不使用标准的forward方法，
        而是使用get_intrinsic_reward方法来计算内在奖励。
        """
        raise RuntimeError("Forward method is not implemented. Use get_intrinsic_reward instead.")

    def train(self, mode: bool = True):
        """
        设置模块为训练模式
        
        Args:
            mode: 是否为训练模式，默认为True
            
        Returns:
            self: 返回模块自身以支持链式调用
        """
        # 设置预测器网络为训练模式
        self.predictor.train(mode)
        
        # 如果启用状态归一化，设置状态归一化器为训练模式
        if self.state_normalization:
            self.state_normalizer.train(mode)
        
        # 如果启用奖励归一化，设置奖励归一化器为训练模式
        if self.reward_normalization:
            self.reward_normalizer.train(mode)
        
        return self

    def eval(self):
        """
        设置模块为评估模式
        
        Returns:
            self: 返回模块自身以支持链式调用
        """
        return self.train(False)

    """
    私有方法
    """

    @staticmethod
    def _build_mlp(input_dims: int, hidden_dims: list[int], output_dims: int, activation_name: str = "elu"):
        """
        构建多层感知机网络
        
        用于构建目标网络和预测器网络
        
        Args:
            input_dims: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dims: 输出维度
            activation_name: 激活函数名称，默认为"elu"
            
        Returns:
            nn.Sequential: 构建的神经网络
        """
        network_layers = []
        
        # 解析隐藏层维度
        # 如果维度为-1，则使用观察值数量
        hidden_dims = [input_dims if dim == -1 else dim for dim in hidden_dims]
        
        # 解析激活函数
        activation = resolve_nn_activation(activation_name)
        
        # 第一层
        network_layers.append(nn.Linear(input_dims, hidden_dims[0]))
        network_layers.append(activation)
        
        # 后续层
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                # 最后一层（输出层）
                network_layers.append(nn.Linear(hidden_dims[layer_index], output_dims))
            else:
                # 隐藏层
                network_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                network_layers.append(activation)
        
        return nn.Sequential(*network_layers)

    """
    不同的权重调度策略
    """

    def _constant_weight_schedule(self, step: int, **kwargs):
        """
        常数权重调度
        
        权重始终保持为初始值
        
        Args:
            step: 当前步数（未使用）
            **kwargs: 其他参数（未使用）
            
        Returns:
            float: 初始权重值
        """
        return self.initial_weight

    def _step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs):
        """
        阶梯权重调度
        
        在指定步数之前使用初始权重，之后使用最终权重
        
        Args:
            step: 当前步数
            final_step: 权重变化的步数
            final_value: 最终权重值
            **kwargs: 其他参数（未使用）
            
        Returns:
            float: 当前步数对应的权重值
        """
        return self.initial_weight if step < final_step else final_value

    def _linear_weight_schedule(self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs):
        """
        线性权重调度
        
        在指定步数范围内线性插值权重值
        
        Args:
            step: 当前步数
            initial_step: 权重开始变化的步数
            final_step: 权重达到最终值的步数
            final_value: 最终权重值
            **kwargs: 其他参数（未使用）
            
        Returns:
            float: 当前步数对应的权重值
        """
        if step < initial_step:
            # 在初始步数之前，使用初始权重
            return self.initial_weight
        elif step > final_step:
            # 在最终步数之后，使用最终权重
            return final_value
        else:
            # 在指定范围内，线性插值权重
            return self.initial_weight + (final_value - self.initial_weight) * (step - initial_step) / (
                final_step - initial_step
            )
