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
学生-教师网络模块

本模块实现了学生-教师架构，用于强化学习中的策略蒸馏。
该架构包含两个神经网络：
- 学生网络 (Student Network): 需要训练的网络，学习模仿教师网络的行为
- 教师网络 (Teacher Network): 预训练的网络，提供专家行为作为学习目标

主要功能：
- 构建学生网络和教师网络
- 支持动作噪声和分布管理
- 提供动作生成和评估接口
- 支持从不同来源加载网络参数

适用场景：
- 策略蒸馏 (Policy Distillation)
- 知识迁移 (Knowledge Transfer)
- 从专家策略学习 (Learning from Expert Policies)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class StudentTeacher(nn.Module):
    """
    学生-教师网络类
    
    该类实现了学生-教师架构，用于策略蒸馏。学生网络学习模仿教师网络的行为，
    通过最小化两者输出之间的差异来实现知识迁移。
    
    属性:
        is_recurrent: 标识该类为非循环网络类型
        loaded_teacher: 标识教师网络是否已加载参数
    """
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        """
        初始化学生-教师网络
        
        Args:
            num_student_obs: 学生网络的观察值维度
            num_teacher_obs: 教师网络的观察值维度
            num_actions: 动作空间维度
            student_hidden_dims: 学生网络的隐藏层维度列表，默认为[256, 256, 256]
            teacher_hidden_dims: 教师网络的隐藏层维度列表，默认为[256, 256, 256]
            activation: 激活函数，默认为"elu"
            init_noise_std: 动作噪声的标准差，默认为0.1
            **kwargs: 其他参数（将被忽略）
        """
        # 检查是否有未预期的参数
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        
        # 初始化父类
        super().__init__()
        
        # 解析激活函数
        activation = resolve_nn_activation(activation)
        
        # 教师网络加载标志
        self.loaded_teacher = False  # 标识教师网络是否已加载参数

        # 设置网络输入维度
        mlp_input_dim_s = num_student_obs  # 学生网络输入维度
        mlp_input_dim_t = num_teacher_obs  # 教师网络输入维度

        # 构建学生网络
        student_layers = []
        # 第一层
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        
        # 后续层
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                # 最后一层（输出层）
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                # 隐藏层
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        
        # 创建学生网络
        self.student = nn.Sequential(*student_layers)

        # 构建教师网络
        teacher_layers = []
        # 第一层
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        
        # 后续层
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                # 最后一层（输出层）
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                # 隐藏层
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        
        # 创建教师网络并设置为评估模式
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()  # 教师网络始终为评估模式

        # 打印网络信息
        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # 动作噪声参数
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))  # 可学习的动作标准差
        self.distribution = None  # 动作分布
        
        # 禁用参数验证以提高速度
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        """
        重置方法（非循环网络，无需实现）
        
        Args:
            dones: 完成标志（未使用）
            hidden_states: 隐藏状态（未使用）
        """
        pass

    def forward(self):
        """
        前向传播方法（未实现）
        
        抛出异常，因为该类不使用标准的forward方法
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        """
        动作均值属性
        
        Returns:
            torch.Tensor: 当前动作分布的均值
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """
        动作标准差属性
        
        Returns:
            torch.Tensor: 当前动作分布的标准差
        """
        return self.distribution.stddev

    @property
    def entropy(self):
        """
        动作熵属性
        
        Returns:
            torch.Tensor: 当前动作分布的熵
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """
        更新动作分布
        
        Args:
            observations: 观察值张量
        """
        # 通过学生网络计算动作均值
        mean = self.student(observations)
        # 扩展标准差以匹配均值形状
        std = self.std.expand_as(mean)
        # 创建正态分布
        self.distribution = Normal(mean, std)

    def act(self, observations):
        """
        生成动作（训练模式）
        
        Args:
            observations: 观察值张量
            
        Returns:
            torch.Tensor: 从动作分布中采样的动作
        """
        # 更新动作分布
        self.update_distribution(observations)
        # 从分布中采样动作
        return self.distribution.sample()

    def act_inference(self, observations):
        """
        生成动作（推理模式）
        
        Args:
            observations: 观察值张量
            
        Returns:
            torch.Tensor: 动作均值（无噪声）
        """
        # 直接返回学生网络的输出（动作均值）
        actions_mean = self.student(observations)
        return actions_mean

    def evaluate(self, teacher_observations):
        """
        评估教师网络输出
        
        Args:
            teacher_observations: 教师网络的观察值张量
            
        Returns:
            torch.Tensor: 教师网络的输出
        """
        # 使用无梯度模式评估教师网络
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """
        加载学生网络和教师网络的参数
        
        该方法支持从两种不同的来源加载参数：
        1. 从强化学习训练中加载（包含"actor"键）
        2. 从策略蒸馏训练中加载（包含"student"键）
        
        Args:
            state_dict (dict): 模型的状态字典
            strict (bool): 是否严格匹配键值，默认为True
            
        Returns:
            bool: 是否从策略蒸馏训练中恢复。该标志用于OnPolicyRunner的load()函数
                 来确定如何加载其他参数。
        """
        # 检查状态字典是否包含强化学习训练的actor参数
        if any("actor" in key for key in state_dict.keys()):  # 从强化学习训练加载参数
            # 重命名键以匹配教师网络，并移除critic参数
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    # 将"actor."替换为空字符串，得到教师网络的参数名
                    teacher_state_dict[key.replace("actor.", "")] = value
            
            # 加载教师网络参数
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            
            # 如果教师网络是循环网络，还需要加载循环记忆（暂未实现）
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            
            # 设置成功加载参数的标志
            self.loaded_teacher = True
            self.teacher.eval()  # 确保教师网络为评估模式
            return False  # 表示不是从策略蒸馏训练恢复
            
        elif any("student" in key for key in state_dict.keys()):  # 从策略蒸馏训练加载参数
            # 加载完整的状态字典（包含学生和教师网络）
            super().load_state_dict(state_dict, strict=strict)
            
            # 设置成功加载参数的标志
            self.loaded_teacher = True
            self.teacher.eval()  # 确保教师网络为评估模式
            return True  # 表示是从策略蒸馏训练恢复
            
        else:
            # 状态字典既不包含actor也不包含student参数
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        """
        获取隐藏状态（非循环网络，返回None）
        
        Returns:
            None: 非循环网络没有隐藏状态
        """
        return None

    def detach_hidden_states(self, dones=None):
        """
        分离隐藏状态（非循环网络，无需实现）
        
        Args:
            dones: 完成标志（未使用）
        """
        pass
