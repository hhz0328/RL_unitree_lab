from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """均匀等级速度命令配置类。
    
    这个类继承自UniformVelocityCommandCfg，专门用于足式机器人的速度控制。
    它添加了limit_ranges字段，用于定义速度命令的限制范围。
    
    主要功能：
    1. 定义机器人需要跟踪的目标速度
    2. 设置速度命令的生成范围
    3. 提供速度限制，确保命令在合理范围内
    """
    
    # 速度限制范围：定义速度命令的上限和下限
    # 这个字段用于确保生成的速度命令不会超出机器人的能力范围
    # 例如：如果机器人的最大速度是1.0 m/s，那么limit_ranges.lin_vel_x应该设置为(-1.0, 1.0)
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
