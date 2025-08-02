from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """线速度命令等级课程学习函数。
    
    这个函数实现了线速度命令的渐进式学习，根据机器人的表现动态调整速度要求。
    当机器人能够很好地跟踪当前速度命令时，会逐渐增加速度的难度。
    
    Args:
        env: 强化学习环境实例
        env_ids: 环境ID序列
        reward_term_name: 奖励项名称，默认为"track_lin_vel_xy"
    
    Returns:
        torch.Tensor: 当前线速度命令范围的上限值
    """
    # 获取速度命令管理器
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges          # 当前速度命令范围
    limit_ranges = command_term.cfg.limit_ranges  # 速度限制范围

    # 获取奖励项配置
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # 计算平均奖励（归一化到回合长度）
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 每个回合结束时检查是否需要调整难度
    if env.common_step_counter % env.max_episode_length == 0:
        # 如果奖励超过阈值的80%，说明机器人表现良好，可以增加难度
        if reward > reward_term.weight * 0.8:
            # 定义速度范围的增量：同时增加正负速度范围
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            
            # 调整X方向线速度范围，并确保不超过限制
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],  # 下限
                limit_ranges.lin_vel_x[1],  # 上限
            ).tolist()
            
            # 调整Y方向线速度范围，并确保不超过限制
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],  # 下限
                limit_ranges.lin_vel_y[1],  # 上限
            ).tolist()

    # 返回当前X方向线速度范围的上限值
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """角速度命令等级课程学习函数。
    
    这个函数实现了角速度命令的渐进式学习，根据机器人的表现动态调整转向速度要求。
    当机器人能够很好地跟踪当前角速度命令时，会逐渐增加转向速度的难度。
    
    Args:
        env: 强化学习环境实例
        env_ids: 环境ID序列
        reward_term_name: 奖励项名称，默认为"track_ang_vel_z"
    
    Returns:
        torch.Tensor: 当前角速度命令范围的上限值
    """
    # 获取速度命令管理器
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges          # 当前速度命令范围
    limit_ranges = command_term.cfg.limit_ranges  # 速度限制范围

    # 获取奖励项配置
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # 计算平均奖励（归一化到回合长度）
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 每个回合结束时检查是否需要调整难度
    if env.common_step_counter % env.max_episode_length == 0:
        # 如果奖励超过阈值的80%，说明机器人表现良好，可以增加难度
        if reward > reward_term.weight * 0.8:
            # 定义角速度范围的增量：同时增加正负角速度范围
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            
            # 调整Z轴角速度范围，并确保不超过限制
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],  # 下限
                limit_ranges.ang_vel_z[1],  # 上限
            ).tolist()

    # 返回当前Z轴角速度范围的上限值
    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
