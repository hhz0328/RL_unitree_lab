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
近端策略优化 (Proximal Policy Optimization, PPO) 算法模块

本模块实现了PPO算法，这是一种重要的策略梯度方法，通过限制策略更新的幅度来确保训练稳定性。
PPO通过裁剪目标函数来防止策略更新过大，同时保持样本效率。

主要功能：
- 标准PPO算法实现
- 支持随机网络蒸馏(RND)用于内在探索
- 支持对称性数据增强和镜像损失
- 支持多GPU分布式训练
- 自适应学习率调整
- 支持循环神经网络(RNN)

核心组件：
- Actor-Critic策略网络
- 轨迹存储 (RolloutStorage)
- RND模块 (可选)
- 对称性增强模块 (可选)
- 优化器和损失函数

适用场景：
- 连续控制任务
- 机器人控制
- 游戏AI
- 需要稳定训练的强化学习任务

参考文献：
- PPO: https://arxiv.org/abs/1707.06347
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:
    """
    近端策略优化 (PPO) 算法类
    
    该类实现了PPO算法，通过裁剪目标函数来限制策略更新幅度，
    确保训练稳定性并提高样本效率。
    
    属性:
        policy: Actor-Critic策略网络
        storage: 轨迹存储组件
        optimizer: 优化器
        rnd: 随机网络蒸馏模块（可选）
        symmetry: 对称性增强配置（可选）
    """

    policy: ActorCritic
    """Actor-Critic策略网络"""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND参数
        rnd_cfg: dict | None = None,
        # 对称性参数
        symmetry_cfg: dict | None = None,
        # 分布式训练参数
        multi_gpu_cfg: dict | None = None,
    ):
        """
        初始化PPO算法
        
        Args:
            policy: Actor-Critic策略网络
            num_learning_epochs: 学习轮数，默认为1
            num_mini_batches: 小批量数量，默认为1
            clip_param: PPO裁剪参数，默认为0.2
            gamma: 折扣因子，默认为0.998
            lam: GAE参数，默认为0.95
            value_loss_coef: 价值损失系数，默认为1.0
            entropy_coef: 熵损失系数，默认为0.0
            learning_rate: 学习率，默认为1e-3
            max_grad_norm: 最大梯度范数，默认为1.0
            use_clipped_value_loss: 是否使用裁剪价值损失，默认为True
            schedule: 学习率调度类型，默认为"fixed"
            desired_kl: 期望KL散度，默认为0.01
            device: 计算设备，默认为"cpu"
            normalize_advantage_per_mini_batch: 是否按小批量归一化优势，默认为False
            rnd_cfg: RND配置字典，默认为None
            symmetry_cfg: 对称性配置字典，默认为None
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

        # RND组件
        if rnd_cfg is not None:
            # 创建RND模块
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # 创建RND优化器
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # 对称性组件
        if symmetry_cfg is not None:
            # 检查是否启用对称性
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # 如果没有使用对称性，打印提示信息
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            
            # 如果函数是字符串，则解析为函数
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            
            # 检查配置有效性
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            
            # 存储对称性配置
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO组件
        self.policy = policy
        self.policy.to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 创建轨迹存储
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO参数
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        """
        初始化存储组件
        
        Args:
            training_type: 训练类型
            num_envs: 环境数量
            num_transitions_per_env: 每个环境的转换数量
            actor_obs_shape: Actor观察值形状
            critic_obs_shape: Critic观察值形状
            actions_shape: 动作形状
        """
        # 为RND创建内存
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        
        # 创建轨迹存储
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        """
        生成动作
        
        Args:
            obs: Actor观察值
            critic_obs: Critic观察值
            
        Returns:
            torch.Tensor: 生成的动作
        """
        # 如果是循环网络，获取隐藏状态
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        # 计算动作和价值
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # 在环境步骤之前记录观察值
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
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
        # 注意：这里克隆是因为稍后我们会基于超时进行奖励引导
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # 计算内在奖励并添加到外在奖励
        if self.rnd:
            # 从infos中获取好奇心门控/观察值
            rnd_state = infos["observations"]["rnd_state"]
            # 计算内在奖励
            # 注意：如果使用归一化，rnd_state是归一化后的门控状态
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # 将内在奖励添加到外在奖励
            self.transition.rewards += self.intrinsic_rewards
            # 记录好奇心门控
            self.transition.rnd_state = rnd_state.clone()

        # 基于超时的奖励引导
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # 记录转换
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        """
        计算回报
        
        Args:
            last_critic_obs: 最后的Critic观察值
        """
        # 计算最后一步的价值
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        """
        更新策略
        
        执行PPO的核心更新步骤，包括代理损失计算、价值函数损失计算、
        以及各种增强功能的损失计算。
        
        Returns:
            dict: 包含各种损失的字典
        """
        # 初始化损失统计
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        
        # RND损失
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        
        # 对称性损失
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # 小批量生成器
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 遍历小批量
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # 每个样本的增强数量
            # 我们从1开始，如果使用对称性增强则增加
            num_aug = 1
            # 原始批量大小
            original_batch_size = obs_batch.shape[0]

            # 检查是否应该按小批量归一化优势
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 执行对称性增强
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # 使用对称性进行数据增强
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # 返回形状: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # 计算每个样本的增强数量
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # 重复批量的其余部分
                # -- Actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- Critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # 重新计算当前批量转换的动作对数概率和熵
            # 注意：我们需要这样做是因为我们用新参数更新了策略
            # -- Actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- Critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- 熵
            # 我们只保留第一个增强（原始）的熵
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL散度计算
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 在所有GPU上减少KL散度
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # 更新学习率
                    # 只在主进程上执行此适应
                    # TODO: 这是否需要？如果KL散度在所有GPU上"相同"，
                    #       那么学习率在所有GPU上也应该相同。
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 为所有GPU更新学习率
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # 为所有参数组更新学习率
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 代理损失（PPO的核心）
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 价值函数损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总损失
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # 对称性损失
            if self.symmetry:
                # 获取对称动作
                # 如果我们之前进行了增强，则不需要再次增强
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # 计算每个样本的增强数量
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actor为对称增强观察值预测的动作
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # 计算对称增强的动作
                # 注意：我们假设第一个增强是原始的。
                #   我们不使用之前的action_batch，因为该动作是从分布中采样的。
                #   但是，对称性损失是使用分布的均值计算的。
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # 计算损失（我们跳过第一个增强，因为它是原始的）
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # 将损失添加到总损失
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # 随机网络蒸馏损失
            if self.rnd:
                # 预测嵌入和目标
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # 计算均方误差损失
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # 计算梯度
            # -- 对于PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- 对于RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # 从所有GPU收集梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 应用梯度
            # -- 对于PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- 对于RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # 存储损失
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND损失
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- 对称性损失
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # 计算平均损失
        # -- 对于PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- 对于RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- 对于对称性
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- 清除存储
        self.storage.clear()

        # 构建损失字典
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

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
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        
        # 广播模型参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        
        # 从源GPU加载模型参数到所有GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """
        从所有GPU收集梯度并平均
        
        该函数在反向传播后调用，以在所有GPU上同步梯度。
        """
        # 创建张量来存储梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # 在所有GPU上平均梯度
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 获取所有参数
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # 用减少的梯度更新所有参数的梯度
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # 从共享缓冲区复制数据
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新下一个参数的偏移量
                offset += numel
