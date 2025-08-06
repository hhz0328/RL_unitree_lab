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
AMP 在线策略训练器模块

该模块实现了基于 AMP (Adversarial Motion Priors) 的在线策略强化学习训练器。
主要功能包括：
- AMP-PPO 算法的完整训练流程
- 对抗性运动先验学习
- 多 GPU 分布式训练支持
- 经验回放和策略更新
- 模型保存和加载
- 训练日志记录和可视化
- 策略蒸馏支持
- RND (Random Network Distillation) 支持

核心组件：
- 策略网络 (Actor-Critic)
- 判别器网络 (Discriminator)
- AMP 数据加载器
- 观察值标准化器
- 经验存储缓冲区
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import AMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    Discriminator,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import AMPLoader, Normalizer, store_code_state


class AmpOnPolicyRunner:
    """
    AMP 在线策略训练器类
    
    该类实现了基于 AMP 的在线策略强化学习训练流程，支持：
    - AMP-PPO 算法训练
    - 对抗性运动先验学习
    - 多 GPU 分布式训练
    - 策略蒸馏
    - RND 内在动机学习
    - 经验标准化
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """
        初始化 AMP 在线策略训练器
        
        Args:
            env: 向量化环境实例
            train_cfg: 训练配置字典
            log_dir: 日志目录路径
            device: 计算设备 (cpu/cuda)
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]  # 算法配置
        self.policy_cfg = train_cfg["policy"]   # 策略配置
        self.device = device
        self.env = env

        # 检查并配置多 GPU 训练
        self._configure_multi_gpu()

        # 根据算法类型确定训练类型
        if self.alg_cfg["class_name"] in ["PPO", "AMPPPO"]:
            self.training_type = "rl"  # 强化学习
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"  # 策略蒸馏
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # 解析观察值维度
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # 解析特权观察值类型
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # Actor-Critic 强化学习，如 PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # 策略蒸馏
            else:
                self.privileged_obs_type = None

        # 解析特权观察值维度
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # 评估策略类并创建实例
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 解析 RND 门控状态维度
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # 检查是否存在 RND 门控状态
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # 获取 RND 门控状态维度
            num_rnd_state = rnd_state.shape[1]
            # 将 RND 门控状态添加到配置中
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # 根据时间步长缩放 RND 权重（类似于 legged_gym 环境中的奖励缩放）
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # 如果使用对称性，则传递环境配置对象
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # 这被对称性函数用于处理不同的观察值项
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # 初始化 AMP 数据加载器
        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.step_dt,  # 帧间时间间隔
            preload_transitions=True,              # 预加载转换
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],  # 预加载转换数量
            motion_files=train_cfg["amp_motion_files"],                        # 运动文件
        )
        # 创建 AMP 观察值标准化器
        amp_normalizer = Normalizer(amp_data.observation_dim)
        # 创建判别器网络
        discriminator = Discriminator(
            amp_data.observation_dim * 2,           # 输入维度（当前状态 + 下一状态）
            train_cfg["amp_reward_coef"],           # 奖励系数
            train_cfg["amp_discr_hidden_dims"],     # 隐藏层维度
            device,
            train_cfg["amp_task_reward_lerp"],      # 任务奖励插值
        ).to(self.device)
        # 最小标准差
        min_std = torch.zeros(len(train_cfg["min_normalized_std"]), device=self.device, requires_grad=False)

        # 初始化算法
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: AMPPPO = alg_class(
            policy,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # 存储训练配置
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 每个环境的步数
        self.save_interval = self.cfg["save_interval"]          # 保存间隔
        self.empirical_normalization = self.cfg["empirical_normalization"]  # 经验标准化
        
        # 初始化观察值标准化器
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            # 不使用标准化
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # 初始化存储和模型
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # 决定是否禁用日志记录
        # 只从 rank 0 进程（主进程）记录日志
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        
        # 日志记录相关
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """
        执行学习训练循环
        
        Args:
            num_learning_iterations: 学习迭代次数
            init_at_random_ep_len: 是否在随机情节长度处初始化（用于探索）
        """
        # 初始化日志记录器
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # 启动 Tensorboard 或 Neptune & Tensorboard 摘要记录器，默认：Tensorboard
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # 检查教师模型是否已加载（策略蒸馏）
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # 随机化初始情节长度（用于探索）
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 开始学习
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        amp_obs = self.env.get_amp_obs_for_expert_trans()
        obs, privileged_obs, amp_obs = obs.to(self.device), privileged_obs.to(self.device), amp_obs.to(self.device)
        self.train_mode()  # 切换到训练模式（例如 dropout）

        # 记录保持
        ep_infos = []
        rewbuffer = deque(maxlen=100)  # 奖励缓冲区
        lenbuffer = deque(maxlen=100)  # 长度缓冲区
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 为记录外在和内在奖励创建缓冲区
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)  # 外在奖励缓冲区
            irewbuffer = deque(maxlen=100)  # 内在奖励缓冲区
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 确保所有参数同步（分布式训练）
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: 是否需要同步经验标准化器？
            #   目前：不需要，因为它们都应该"渐近地"收敛到相同的值

        # 开始训练
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # 经验收集阶段
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 采样动作
                    actions = self.alg.act(obs, privileged_obs, amp_obs)
                    # 环境步进
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_obs_for_expert_trans()
                    # 移动到设备
                    obs, rewards, dones, next_amp_obs = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_amp_obs.to(self.device),
                    )
                    # 执行标准化
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # 处理终止状态转换
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids = self.env.reset_env_ids
                    terminal_amp_states = self.env.get_amp_obs_for_expert_trans()[reset_env_ids]
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    # 使用判别器预测 AMP 奖励
                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                    )[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    # 提取内在奖励（仅用于记录）
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # 记录保持
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # 更新奖励
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # 更新情节长度
                        cur_episode_length += 1
                        # 清除已完成情节的数据
                        # -- 通用
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- 内在和外在奖励
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # 计算回报
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # 更新策略
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # 记录信息
            if self.log_dir is not None and not self.disable_logs:
                # 记录信息
                self.log(locals())
                # 保存模型
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清除情节信息
            ep_infos.clear()
            # 保存代码状态
            if it == start_iter and not self.disable_logs:
                # 获取所有差异文件
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 如果可能，将它们存储到 wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束后保存最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """
        记录训练日志
        
        Args:
            locs: 局部变量字典
            width: 日志输出宽度
            pad: 填充宽度
        """
        """
        记录训练日志
        
        Args:
            locs: 局部变量字典
            width: 日志输出宽度
            pad: 填充宽度
        """
        # 计算收集大小
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # 更新总时间步和时间
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- 情节信息
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 处理标量和零维张量信息
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # 记录到日志记录器和终端
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # 分别记录内在和外在奖励
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # 其他所有内容
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb 不支持非整数 x 轴记录
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- 情节信息
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        """
        保存模型
        
        Args:
            path: 保存路径
            infos: 额外信息
        """
        """
        保存模型
        
        Args:
            path: 保存路径
            infos: 额外信息
        """
        # -- 保存模型
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- 如果使用 RND，保存 RND 模型
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- 如果使用经验标准化，保存观察值标准化器
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # 保存模型
        torch.save(saved_dict, path)

        # 将模型上传到外部日志服务
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """
        加载模型
        
        Args:
            path: 模型路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            加载的额外信息
        """
        """
        加载模型
        
        Args:
            path: 模型路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            加载的额外信息
        """
        loaded_dict = torch.load(path, weights_only=False)
        # -- 加载模型
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        # -- 如果使用 RND，加载 RND 模型
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- 如果使用经验标准化，加载观察值标准化器
        if self.empirical_normalization:
            if resumed_training:
                # 如果恢复之前的训练，Actor/Student 标准化器为 Actor/Student 加载
                # Critic/Teacher 标准化器为 Critic/Teacher 加载
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 如果训练没有恢复但加载了模型，这次运行必须是在 RL 训练之后的蒸馏训练
                # 因此 Actor 标准化器为教师模型加载。学生的标准化器不加载，
                # 因为观察空间可能与之前的 RL 训练不同
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- 如果使用，加载优化器
        if load_optimizer and resumed_training:
            # -- 算法优化器
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- 如果使用，RND 优化器
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- 加载当前学习迭代
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """
        获取推理策略
        
        Args:
            device: 计算设备
            
        Returns:
            推理策略函数
        """
        """
        获取推理策略
        
        Args:
            device: 计算设备
            
        Returns:
            推理策略函数
        """
        self.eval_mode()  # 切换到评估模式（例如 dropout）
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        """切换到训练模式"""
        # -- PPO
        self.alg.policy.train()
        self.alg.discriminator.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- 标准化
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """切换到评估模式"""
        # -- PPO
        self.alg.policy.eval()
        self.alg.discriminator.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- 标准化
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """
        添加 Git 仓库到日志记录
        
        Args:
            repo_file_path: 仓库文件路径
        """
        """
        添加 Git 仓库到日志记录
        
        Args:
            repo_file_path: 仓库文件路径
        """
        self.git_status_repos.append(repo_file_path)

    """
    辅助函数
    """

    def _configure_multi_gpu(self):
        """配置多 GPU 训练"""
        # 检查是否启用分布式训练
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # 如果不是分布式训练，将本地和全局 rank 设置为 0 并返回
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # 获取 rank 和 world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # 创建配置字典
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # 主进程的 rank
            "local_rank": self.gpu_local_rank,    # 当前进程的 rank
            "world_size": self.gpu_world_size,    # 总进程数
        }

        # 检查用户是否为本地 rank 指定了设备
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # 验证多 GPU 配置
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # 初始化 torch 分布式
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # 将设备设置为本地 rank
        torch.cuda.set_device(self.gpu_local_rank)
