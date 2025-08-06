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
AMP运动加载器显示模块

本模块实现了用于显示和可视化的AMP运动数据加载器。
该模块专门用于加载和处理运动捕捉数据，支持时间插值、批量采样和预加载功能。

主要功能：
- 加载JSON格式的运动捕捉数据
- 支持时间插值和帧混合
- 加权轨迹采样
- 批量数据处理
- 预加载优化

核心特性：
- 时间插值：在帧之间进行平滑插值
- 轨迹权重：支持不同轨迹的加权采样
- 批量处理：支持批量数据生成
- 预加载：可选的预加载功能以提高性能

数据结构：
- 关节位置：26个关节的位置信息
- 关节速度：26个关节的速度信息
- 根状态：6个根状态参数（位置和方向）

适用场景：
- AMP训练中的专家数据加载
- 运动可视化
- 运动数据分析和处理
- 机器人运动规划
"""

import glob
import json

import numpy as np
import torch


class AMPLoaderDisplay:
    """
    AMP运动加载器显示类
    
    该类专门用于加载和处理运动捕捉数据，支持时间插值、批量采样和预加载功能。
    主要用于AMP训练中的专家数据加载和运动可视化。
    
    属性:
        trajectories: 轨迹数据列表
        trajectory_names: 轨迹名称列表
        trajectory_weights: 轨迹权重列表
        trajectory_lens: 轨迹长度列表
        device: 计算设备
    """

    # 数据维度常量
    JOINT_POS_SIZE = 26      # 关节位置维度
    JOINT_VEL_SIZE = 26      # 关节速度维度

    # 数据索引常量
    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    ROOT_STATES_NUM = 6      # 根状态数量
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """
        初始化AMP运动加载器
        
        专家数据集提供来自运动捕捉数据集的AMP观察值。
        
        Args:
            device: 计算设备（CPU或GPU）
            time_between_frames: 帧之间的时间间隔（秒）
            data_dir: 数据目录路径
            preload_transitions: 是否预加载转换数据
            num_preload_transitions: 预加载的转换数量
            motion_files: 运动文件路径列表
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # 为每个轨迹存储的值
        self.trajectories = []              # 轨迹数据
        self.trajectories_full = []         # 完整轨迹数据
        self.trajectory_names = []          # 轨迹名称
        self.trajectory_idxs = []           # 轨迹索引
        self.trajectory_lens = []           # 轨迹长度（秒）
        self.trajectory_weights = []        # 轨迹权重
        self.trajectory_frame_durations = [] # 轨迹帧持续时间
        self.trajectory_num_frames = []     # 轨迹帧数

        # 加载每个运动文件
        for i, motion_file in enumerate(motion_files):
            # 提取轨迹名称
            self.trajectory_names.append(motion_file.split(".")[0])
            
            # 读取JSON文件
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # 移除前7个观察值维度（根位置和根方向）
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, : AMPLoaderDisplay.JOINT_VEL_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : AMPLoaderDisplay.JOINT_VEL_END_IDX], dtype=torch.float32, device=device
                    )
                )
                
                # 存储轨迹信息
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # 归一化轨迹权重
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # 预加载转换数据
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print("Preloading {num_preload_transitions} transitions")
            # 批量采样轨迹索引和时间
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            # 预加载当前帧和下一帧
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        # 合并所有完整轨迹
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """
        通过加权采样获取轨迹索引
        
        Returns:
            int: 采样的轨迹索引
        """
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """
        批量采样轨迹索引
        
        Args:
            size: 采样数量
            
        Returns:
            np.ndarray: 采样的轨迹索引数组
        """
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """
        为轨迹采样随机时间
        
        Args:
            traj_idx: 轨迹索引
            
        Returns:
            float: 采样的时间点
        """
        # 计算时间偏移量
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        # 确保时间在有效范围内
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """
        为多个轨迹批量采样随机时间
        
        Args:
            traj_idxs: 轨迹索引数组
            
        Returns:
            np.ndarray: 采样的时间点数组
        """
        # 计算时间偏移量
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        # 批量采样时间
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        # 确保时间非负
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        """
        线性插值（球面线性插值）
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            blend: 混合系数 [0, 1]
            
        Returns:
            torch.Tensor: 插值结果
        """
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """
        获取指定轨迹的AMP观察值
        
        Args:
            traj_idx: 轨迹索引
            
        Returns:
            torch.Tensor: 轨迹数据
        """
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """
        获取指定轨迹在指定时间的帧
        
        Args:
            traj_idx: 轨迹索引
            time: 时间点
            
        Returns:
            torch.Tensor: 插值后的帧数据
        """
        # 计算时间比例
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        
        # 计算相邻帧索引
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        # 插值得到结果
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """
        批量获取指定轨迹在指定时间的帧
        
        Args:
            traj_idxs: 轨迹索引数组
            times: 时间点数组
            
        Returns:
            torch.Tensor: 批量插值后的帧数据
        """
        # 计算时间比例
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        
        # 计算相邻帧索引
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        
        # 初始化结果张量
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        
        # 为每个轨迹获取帧数据
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        
        # 计算混合系数并插值
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """
        获取指定轨迹在指定时间的完整帧
        
        Args:
            traj_idx: 轨迹索引
            time: 时间点
            
        Returns:
            torch.Tensor: 完整帧数据
        """
        # 计算时间比例
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        
        # 计算相邻帧索引
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        
        # 混合帧姿态
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """
        批量获取指定轨迹在指定时间的完整帧
        
        Args:
            traj_idxs: 轨迹索引数组
            times: 时间点数组
            
        Returns:
            torch.Tensor: 批量完整帧数据
        """
        # 计算时间比例
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        
        # 计算相邻帧索引
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        
        # 初始化AMP帧数据张量
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs),
            AMPLoaderDisplay.JOINT_VEL_END_IDX - AMPLoaderDisplay.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs),
            AMPLoaderDisplay.JOINT_VEL_END_IDX - AMPLoaderDisplay.JOINT_POSE_START_IDX,
            device=self.device,
        )
        
        # 为每个轨迹获取AMP帧数据
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX
            ]
        
        # 计算混合系数并插值
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """
        获取随机帧
        
        Returns:
            torch.Tensor: 随机采样的帧数据
        """
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """
        获取随机完整帧
        
        Returns:
            torch.Tensor: 随机采样的完整帧数据
        """
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        """
        批量获取完整帧
        
        Args:
            num_frames: 帧数量
            
        Returns:
            torch.Tensor: 批量完整帧数据
        """
        if self.preload_transitions:
            # 使用预加载的数据
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            # 实时采样
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """
        在两个帧之间进行线性插值，包括方向
        
        Args:
            frame0: 第一个帧，对应blend=0
            frame1: 第二个帧，对应blend=1
            blend: [0,1]之间的浮点数，指定两个帧之间的插值
            
        Returns:
            torch.Tensor: 两个帧的插值结果
        """
        # 提取关节姿态和速度
        joints0, joints1 = AMPLoaderDisplay.get_joint_pose(frame0), AMPLoaderDisplay.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoaderDisplay.get_joint_vel(frame0), AMPLoaderDisplay.get_joint_vel(frame1)

        # 插值关节姿态和速度
        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        # 连接结果
        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        生成AMP转换的批量数据
        
        Args:
            num_mini_batch: 小批量数量
            mini_batch_size: 小批量大小
            
        Yields:
            tuple: (当前状态, 下一状态) 张量对
        """
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                # 使用预加载的数据
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX]
                s_next = self.preloaded_s_next[
                    idxs, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX
                ]
            else:
                # 实时生成数据
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            
            yield s, s_next

    @property
    def observation_dim(self):
        """
        AMP观察值的维度
        
        Returns:
            int: 观察值维度
        """
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        """
        运动数量
        
        Returns:
            int: 运动数量
        """
        return len(self.trajectory_names)

    @staticmethod
    def get_joint_pose(pose):
        """
        从姿态数据中提取关节姿态
        
        Args:
            pose: 姿态数据
            
        Returns:
            torch.Tensor: 关节姿态数据
        """
        return pose[AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        """
        从批量姿态数据中提取关节姿态
        
        Args:
            poses: 批量姿态数据
            
        Returns:
            torch.Tensor: 批量关节姿态数据
        """
        return poses[:, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        """
        从姿态数据中提取关节速度
        
        Args:
            pose: 姿态数据
            
        Returns:
            torch.Tensor: 关节速度数据
        """
        return pose[AMPLoaderDisplay.JOINT_VEL_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        """
        从批量姿态数据中提取关节速度
        
        Args:
            poses: 批量姿态数据
            
        Returns:
            torch.Tensor: 批量关节速度数据
        """
        return poses[:, AMPLoaderDisplay.JOINT_VEL_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX]
