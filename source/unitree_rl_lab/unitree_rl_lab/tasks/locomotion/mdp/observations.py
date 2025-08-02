from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """计算步态相位函数。
    
    这个函数用于计算机器人的步态相位，为强化学习提供时序信息。
    步态相位是一个周期性信号，用于帮助机器人学习自然的步态模式。
    
    步态相位的计算基于时间，使用正弦和余弦函数生成周期性信号：
    - 相位0：sin(0) = 0, cos(0) = 1
    - 相位π/2：sin(π/2) = 1, cos(π/2) = 0  
    - 相位π：sin(π) = 0, cos(π) = -1
    - 相位3π/2：sin(3π/2) = -1, cos(3π/2) = 0
    
    Args:
        env: 强化学习环境实例
        period: 步态周期（秒），例如0.8秒表示一个完整的步态周期
    
    Returns:
        torch.Tensor: 形状为(num_envs, 2)的张量，包含每个环境的步态相位
                     第一列是sin(phase)，第二列是cos(phase)
    """
    # 初始化回合长度缓冲区（如果不存在）
    # 这个缓冲区用于跟踪每个环境的当前回合步数
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # 计算全局相位：将时间转换为0到1之间的相位值
    # episode_length_buf * env.step_dt: 当前回合的时间（秒）
    # % period: 取模运算，确保相位在周期内循环
    # / period: 归一化到0-1范围
    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    # 创建相位张量：形状为(num_envs, 2)
    # 第一列存储sin(phase)，第二列存储cos(phase)
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    
    # 计算sin(2π * global_phase)：提供相位的正弦分量
    # 2π确保一个完整的周期
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    
    # 计算cos(2π * global_phase)：提供相位的余弦分量
    # 余弦分量与正弦分量相位差90度
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    
    return phase
