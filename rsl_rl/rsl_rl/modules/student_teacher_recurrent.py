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
学生-教师循环神经网络模块

本模块实现了带有循环神经网络(RNN)的学生-教师架构，用于强化学习中的策略蒸馏。
该模块扩展了基础的StudentTeacher类，通过添加记忆组件来处理序列观察值。

主要功能：
- 支持LSTM和GRU类型的循环神经网络
- 可配置学生网络和教师网络是否使用RNN
- 处理序列观察值的隐藏状态管理
- 支持批量模式和推理模式

适用场景：
- 需要记忆的强化学习任务
- 部分可观察马尔可夫决策过程(POMDP)
- 策略蒸馏中的序列数据处理
"""

from __future__ import annotations

import warnings

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class StudentTeacherRecurrent(StudentTeacher):
    """
    学生-教师循环神经网络类
    
    该类扩展了基础的StudentTeacher类，通过添加循环神经网络组件来处理序列观察值。
    支持学生网络和教师网络独立配置是否使用RNN。
    
    属性:
        is_recurrent: 标识该类为循环网络类型
    """
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=0.1,
        teacher_recurrent=False,
        **kwargs,
    ):
        """
        初始化学生-教师循环神经网络
        
        Args:
            num_student_obs: 学生网络的观察值维度
            num_teacher_obs: 教师网络的观察值维度
            num_actions: 动作空间维度
            student_hidden_dims: 学生网络的隐藏层维度列表，默认为[256, 256, 256]
            teacher_hidden_dims: 教师网络的隐藏层维度列表，默认为[256, 256, 256]
            activation: 激活函数，默认为"elu"
            rnn_type: RNN类型，可选"lstm"或"gru"，默认为"lstm"
            rnn_hidden_dim: RNN隐藏层维度，默认为256
            rnn_num_layers: RNN层数，默认为1
            init_noise_std: 动作噪声的标准差，默认为0.1
            teacher_recurrent: 教师网络是否使用RNN，默认为False
            **kwargs: 其他参数（将被忽略）
        """
        # 处理已弃用的参数
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            # 只有当新参数使用默认值时才覆盖
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        
        # 检查是否有未预期的参数
        if kwargs:
            print(
                "StudentTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        # 存储教师网络是否使用RNN的标志
        self.teacher_recurrent = teacher_recurrent

        # 调用父类初始化
        # 如果教师网络使用RNN，则教师观察值维度为RNN隐藏维度，否则为原始维度
        super().__init__(
            num_student_obs=rnn_hidden_dim,  # 学生网络输入维度为RNN隐藏维度
            num_teacher_obs=rnn_hidden_dim if teacher_recurrent else num_teacher_obs,  # 教师网络输入维度
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        # 解析激活函数
        activation = resolve_nn_activation(activation)

        # 创建学生网络的记忆组件
        self.memory_s = Memory(
            num_student_obs, 
            type=rnn_type, 
            num_layers=rnn_num_layers, 
            hidden_size=rnn_hidden_dim
        )
        
        # 如果教师网络使用RNN，创建教师网络的记忆组件
        if self.teacher_recurrent:
            self.memory_t = Memory(
                num_teacher_obs, 
                type=rnn_type, 
                num_layers=rnn_num_layers, 
                hidden_size=rnn_hidden_dim
            )

        # 打印网络信息
        print(f"Student RNN: {self.memory_s}")
        if self.teacher_recurrent:
            print(f"Teacher RNN: {self.memory_t}")

    def reset(self, dones=None, hidden_states=None):
        """
        重置记忆组件的隐藏状态
        
        Args:
            dones: 完成标志张量，用于重置特定环境的隐藏状态
            hidden_states: 隐藏状态元组 (student_hidden_states, teacher_hidden_states)
        """
        # 如果没有提供隐藏状态，使用None
        if hidden_states is None:
            hidden_states = (None, None)
        
        # 重置学生网络的隐藏状态
        self.memory_s.reset(dones, hidden_states[0])
        
        # 如果教师网络使用RNN，重置教师网络的隐藏状态
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])

    def act(self, observations):
        """
        生成动作（训练模式）
        
        Args:
            observations: 观察值张量
            
        Returns:
            tuple: (动作张量, 动作对数概率张量, 熵张量, 学生网络输出张量)
        """
        # 通过学生网络的记忆组件处理观察值
        input_s = self.memory_s(observations)
        # 调用父类的act方法，传入处理后的观察值
        return super().act(input_s.squeeze(0))

    def act_inference(self, observations):
        """
        生成动作（推理模式）
        
        Args:
            observations: 观察值张量
            
        Returns:
            torch.Tensor: 动作张量（无噪声）
        """
        # 通过学生网络的记忆组件处理观察值
        input_s = self.memory_s(observations)
        # 调用父类的act_inference方法，传入处理后的观察值
        return super().act_inference(input_s.squeeze(0))

    def evaluate(self, teacher_observations):
        """
        评估教师网络输出
        
        Args:
            teacher_observations: 教师网络的观察值张量
            
        Returns:
            torch.Tensor: 教师网络输出张量
        """
        # 如果教师网络使用RNN，通过记忆组件处理观察值
        if self.teacher_recurrent:
            teacher_observations = self.memory_t(teacher_observations)
        
        # 调用父类的evaluate方法，传入处理后的观察值
        return super().evaluate(teacher_observations.squeeze(0))

    def get_hidden_states(self):
        """
        获取当前隐藏状态
        
        Returns:
            tuple: (学生网络隐藏状态, 教师网络隐藏状态)
                    如果教师网络不使用RNN，则教师网络隐藏状态为None
        """
        if self.teacher_recurrent:
            # 如果教师网络使用RNN，返回两个网络的隐藏状态
            return self.memory_s.hidden_states, self.memory_t.hidden_states
        else:
            # 如果教师网络不使用RNN，只返回学生网络的隐藏状态
            return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        """
        分离隐藏状态（用于梯度计算）
        
        Args:
            dones: 完成标志张量，用于分离特定环境的隐藏状态
        """
        # 分离学生网络的隐藏状态
        self.memory_s.detach_hidden_states(dones)
        
        # 如果教师网络使用RNN，分离教师网络的隐藏状态
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)
