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
轨迹存储 (Rollout Storage) 模块

本模块实现了轨迹存储系统，用于存储和管理强化学习训练过程中的轨迹数据。
轨迹存储是强化学习算法中的核心组件，用于收集、组织和提供训练所需的数据。

主要功能：
- 轨迹数据收集和存储
- 支持强化学习和策略蒸馏两种训练模式
- 支持前馈网络和循环网络
- 自动计算回报和优势函数
- 提供多种数据生成器

核心特性：
- 多环境并行：支持多个环境同时收集数据
- 灵活存储：根据训练类型动态分配存储空间
- 自动计算：自动计算GAE回报和优势函数
- 循环网络支持：支持RNN的隐藏状态管理
- RND支持：支持随机网络蒸馏的状态存储

适用场景：
- 强化学习训练（PPO、AMP等）
- 策略蒸馏训练
- 多环境并行训练
- 循环神经网络训练
"""

from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    """
    轨迹存储类
    
    该类实现了完整的轨迹存储系统，支持强化学习和策略蒸馏两种训练模式。
    提供灵活的数据存储、自动计算和批量生成功能。
    
    属性:
        observations: 观察值张量
        privileged_observations: 特权观察值张量
        rewards: 奖励张量
        actions: 动作张量
        dones: 完成标志张量
        values: 价值函数张量（RL模式）
        returns: 回报张量（RL模式）
        advantages: 优势函数张量（RL模式）
    """

    class Transition:
        """
        转换数据类
        
        用于临时存储单个时间步的转换数据，包括观察值、动作、奖励等。
        """
        def __init__(self):
            # 核心数据
            self.observations = None              # 观察值
            self.privileged_observations = None   # 特权观察值
            self.actions = None                   # 动作
            self.privileged_actions = None        # 特权动作（蒸馏模式）
            self.rewards = None                   # 奖励
            self.dones = None                     # 完成标志
            
            # 强化学习相关
            self.values = None                    # 价值函数输出
            self.actions_log_prob = None          # 动作对数概率
            self.action_mean = None               # 动作均值
            self.action_sigma = None              # 动作标准差
            
            # 其他
            self.hidden_states = None             # 隐藏状态（RNN）
            self.rnd_state = None                 # RND状态

        def clear(self):
            """清空所有数据"""
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        """
        初始化轨迹存储
        
        Args:
            training_type: 训练类型（"rl"或"distillation"）
            num_envs: 环境数量
            num_transitions_per_env: 每个环境的转换数量
            obs_shape: 观察值形状
            privileged_obs_shape: 特权观察值形状
            actions_shape: 动作形状
            rnd_state_shape: RND状态形状，默认为None
            device: 计算设备，默认为"cpu"
        """
        # 存储输入参数
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        # 核心数据存储
        # 形状: [时间步, 环境数, *观察值形状]
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        
        # 特权观察值（可选）
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        
        # 其他核心数据
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # 策略蒸馏模式专用数据
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # 强化学习模式专用数据
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # RND状态存储（可选）
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # RNN网络隐藏状态存储
        self.saved_hidden_states_a = None  # Actor隐藏状态
        self.saved_hidden_states_c = None  # Critic隐藏状态

        # 转换计数器
        self.step = 0

    def add_transitions(self, transition: Transition):
        """
        添加转换数据
        
        将单个时间步的转换数据添加到存储中。
        
        Args:
            transition: 包含转换数据的Transition对象
        """
        # 检查缓冲区是否溢出
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # 复制核心数据
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # 策略蒸馏模式数据
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # 强化学习模式数据
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # RND状态数据
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # 保存RNN隐藏状态
        self._save_hidden_states(transition.hidden_states)

        # 增加步数计数器
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """
        保存RNN隐藏状态
        
        Args:
            hidden_states: 隐藏状态元组 (actor_hidden_states, critic_hidden_states)
        """
        if hidden_states is None or hidden_states == (None, None):
            return
        
        # 将GRU隐藏状态转换为元组格式以匹配LSTM格式
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        
        # 如果需要则初始化隐藏状态存储
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        
        # 复制隐藏状态
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        """清空存储，重置步数计数器"""
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        """
        计算回报和优势函数
        
        使用广义优势估计(GAE)计算每个时间步的回报和优势函数。
        
        Args:
            last_values: 最后一步的价值函数输出
            gamma: 折扣因子
            lam: GAE参数
            normalize_advantage: 是否归一化优势函数
        """
        advantage = 0
        
        # 从后往前计算GAE
        for step in reversed(range(self.num_transitions_per_env)):
            # 如果是最后一步，使用提供的价值函数输出
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            
            # 非终止状态标志：1表示非终止，0表示终止
            next_is_not_terminal = 1.0 - self.dones[step].float()
            
            # TD误差：r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            
            # 优势函数：A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            
            # 回报：R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # 计算优势函数
        self.advantages = self.returns - self.values
        
        # 归一化优势函数（如果启用）
        # 这是为了防止重复归一化（例如，如果使用每小批量归一化）
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def generator(self):
        """
        策略蒸馏数据生成器
        
        为策略蒸馏训练提供数据生成器。
        
        Yields:
            tuple: (observations, privileged_observations, actions, privileged_actions, dones)
        """
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            # 如果没有特权观察值，使用普通观察值
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[i], self.dones[i]

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        前馈网络小批量生成器
        
        为强化学习训练提供随机小批量数据生成器。
        
        Args:
            num_mini_batches: 小批量数量
            num_epochs: 训练轮数，默认为8
            
        Yields:
            tuple: 包含所有训练所需数据的元组
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        
        # 计算批量大小
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        
        # 生成随机索引
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # 展平核心数据
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # PPO相关数据
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # RND数据
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        # 生成小批量数据
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # 选择小批量索引
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # 创建小批量数据
                # -- 核心数据
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- PPO数据
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- RND数据
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # 返回小批量数据
                yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None, rnd_state_batch

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        循环网络小批量生成器
        
        为循环神经网络训练提供轨迹数据生成器。
        处理轨迹的填充和掩码，以支持变长序列。
        
        Args:
            num_mini_batches: 小批量数量
            num_epochs: 训练轮数，默认为8
            
        Yields:
            tuple: 包含所有训练所需数据的元组，包括隐藏状态和掩码
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        
        # 分割和填充轨迹
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        # RND轨迹处理
        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                # 计算轨迹边界
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # 准备轨迹数据
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                # 准备其他数据
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # 处理隐藏状态
                # 重塑为 [num_envs, time, num layers, hidden dim]（原始形状: [time, num_layers, num_envs, hidden_dim]）
                # 然后只取完成后的时间步（展平num envs和time维度），
                # 取一批轨迹，最后重塑回 [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                
                # 为GRU移除元组
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                # 返回轨迹数据
                yield obs_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch, rnd_state_batch

                first_traj = last_traj