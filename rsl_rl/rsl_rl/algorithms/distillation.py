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
策略蒸馏 (Policy Distillation) 算法模块

本模块实现了策略蒸馏算法，用于训练学生模型来模仿教师模型的行为。
策略蒸馏是一种知识迁移技术，通过让学生网络学习教师网络的输出来实现知识传递。

主要功能：
- 行为克隆 (Behavior Cloning)
- 支持MSE和Huber损失函数
- 支持循环神经网络 (RNN)
- 支持多GPU分布式训练
- 梯度累积和批量更新

核心组件：
- 学生-教师网络 (StudentTeacher/StudentTeacherRecurrent)
- 轨迹存储 (RolloutStorage)
- 优化器和损失函数
- 分布式训练支持

适用场景：
- 从专家策略学习
- 模型压缩和加速
- 知识迁移
- 多模态学习
"""

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage


class Distillation:
    """
    策略蒸馏算法类
    
    该类实现了策略蒸馏算法，通过行为克隆训练学生模型来模仿教师模型。
    学生网络学习教师网络的输出，实现知识迁移。
    
    属性:
        policy: 学生-教师网络模型
        storage: 轨迹存储组件
        optimizer: 优化器
        loss_fn: 损失函数
    """

    policy: StudentTeacher | StudentTeacherRecurrent
    """学生-教师网络模型"""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        loss_type="mse",
        device="cpu",
        # 分布式训练参数
        multi_gpu_cfg: dict | None = None,
    ):
        """
        初始化策略蒸馏算法
        
        Args:
            policy: 学生-教师网络模型
            num_learning_epochs: 学习轮数，默认为1
            gradient_length: 梯度累积长度，默认为15
            learning_rate: 学习率，默认为1e-3
            loss_type: 损失函数类型，可选"mse"或"huber"，默认为"mse"
            device: 计算设备，默认为"cpu"
            multi_gpu_cfg: 多GPU配置字典，默认为None
        """
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        
        # 多GPU参数
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]  # 全局GPU排名
            self.gpu_world_size = multi_gpu_cfg["world_size"]    # 总GPU数量
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND组件（占位符，待移除）
        self.rnd = None  # TODO: remove when runner has a proper base class

        # 蒸馏组件
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # 稍后初始化
        # 只优化学生网络参数
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()  # 转换存储
        self.last_hidden_states = None  # 最后的隐藏状态

        # 蒸馏参数
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate

        # 初始化损失函数
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss  # 均方误差损失
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss  # Huber损失
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        # 更新计数器
        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        """
        初始化存储组件
        
        Args:
            training_type: 训练类型
            num_envs: 环境数量
            num_transitions_per_env: 每个环境的转换数量
            student_obs_shape: 学生网络观察值形状
            teacher_obs_shape: 教师网络观察值形状
            actions_shape: 动作形状
        """
        # 创建轨迹存储
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,  # RND状态形状（蒸馏不使用RND）
            self.device,
        )

    def act(self, obs, teacher_obs):
        """
        生成动作
        
        Args:
            obs: 学生网络观察值
            teacher_obs: 教师网络观察值
            
        Returns:
            torch.Tensor: 学生网络生成的动作
        """
        # 计算动作
        self.transition.actions = self.policy.act(obs).detach()  # 学生网络动作
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()  # 教师网络动作
        # 记录观察值
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        """
        处理环境步骤
        
        Args:
            rewards: 奖励
            dones: 完成标志
            infos: 信息字典
        """
        # 记录奖励和完成标志
        self.transition.rewards = rewards
        self.transition.dones = dones
        # 记录转换
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        """
        更新学生网络
        
        通过行为克隆损失训练学生网络来模仿教师网络的行为。
        
        Returns:
            dict: 包含行为克隆损失的字典
        """
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        # 遍历学习轮数
        for epoch in range(self.num_learning_epochs):
            # 重置隐藏状态
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            # 遍历轨迹数据
            for obs, _, _, privileged_actions, dones in self.storage.generator():
                # 推理学生网络以计算梯度
                actions = self.policy.act_inference(obs)

                # 行为克隆损失
                # 让学生网络的动作接近教师网络的动作
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # 累积总损失
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # 梯度步骤
                if cnt % self.gradient_length == 0:
                    # 清零梯度
                    self.optimizer.zero_grad()
                    # 反向传播
                    loss.backward()
                    # 多GPU梯度同步
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    # 更新参数
                    self.optimizer.step()
                    # 分离隐藏状态
                    self.policy.detach_hidden_states()
                    # 重置损失
                    loss = 0

                # 重置完成标志
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        # 计算平均行为损失
        mean_behavior_loss /= cnt
        # 清除存储
        self.storage.clear()
        # 保存最后的隐藏状态
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # 构建损失字典
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    """
    辅助函数
    """

    def broadcast_parameters(self):
        """
        将模型参数广播到所有GPU
        
        用于分布式训练中同步模型参数
        """
        # 获取当前GPU上的模型参数
        model_params = [self.policy.state_dict()]
        # 广播模型参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 从源GPU加载模型参数到所有GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """
        从所有GPU收集梯度并平均
        
        该函数在反向传播后调用，以在所有GPU上同步梯度。
        """
        # 创建张量来存储梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        
        # 在所有GPU上平均梯度
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        
        # 用减少的梯度更新所有参数的梯度
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # 从共享缓冲区复制数据
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新下一个参数的偏移量
                offset += numel
