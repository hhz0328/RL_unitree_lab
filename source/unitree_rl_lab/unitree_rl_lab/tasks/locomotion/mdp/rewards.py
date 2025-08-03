from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
关节惩罚函数。
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚机器人关节消耗的能量。
    
    计算每个关节的功率消耗：功率 = 角速度 × 力矩
    总能量消耗是所有关节功率的绝对值之和。
    
    Args:
        env: 强化学习环境实例
        asset_cfg: 资产配置，指定要计算的机器人
    
    Returns:
        torch.Tensor: 每个环境的能量消耗惩罚值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节角速度和力矩
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]      # 关节角速度
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids] # 关节力矩
    
    # 计算功率：|角速度| × |力矩|
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """鼓励机器人在没有移动命令时保持静止。
    
    当速度命令很小时，惩罚关节偏离默认位置。
    这有助于机器人学习在静止时保持稳定的姿态。
    
    Args:
        env: 强化学习环境实例
        command_name: 速度命令名称
        asset_cfg: 资产配置
    
    Returns:
        torch.Tensor: 静止惩罚值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算关节位置偏离默认位置的惩罚
    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    
    # 获取速度命令的范数
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    
    # 只有当速度命令很小时才应用惩罚
    return reward * (cmd_norm < 0.1)


"""
机器人姿态奖励函数。
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励机器人将重力方向与期望方向对齐（使用L2平方核）。
    
    通过计算投影重力与期望重力向量的余弦距离来评估姿态。
    
    Args:
        env: 强化学习环境实例
        desired_gravity: 期望的重力方向向量
        asset_cfg: 资产配置
    
    Returns:
        torch.Tensor: 姿态对齐奖励值
    """
    # 提取使用的量（启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    # 计算余弦距离
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)
    # 从[-1, 1]映射到[0, 1]
    normalized = 0.5 * cos_dist + 0.5
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚Z轴基座线速度（使用L2平方核）。
    
    鼓励机器人保持直立，防止上升或下降。
    
    Args:
        env: 强化学习环境实例
        asset_cfg: 资产配置
    
    Returns:
        torch.Tensor: 向上运动惩罚值
    """
    # 提取使用的量（启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算与垂直方向的偏差
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """惩罚关节位置偏离默认位置。
    
    当机器人移动时，允许更大的关节偏差；
    当机器人静止时，要求更严格的关节位置。
    
    Args:
        env: 强化学习环境实例
        asset_cfg: 资产配置
        stand_still_scale: 静止时的惩罚缩放因子
        velocity_threshold: 速度阈值
    
    Returns:
        torch.Tensor: 关节位置惩罚值
    """
    # 提取使用的量（启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取速度命令和身体速度
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    
    # 计算关节位置偏差
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    
    # 根据运动状态调整惩罚
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
脚部奖励函数。
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚脚部绊倒。
    
    检测脚部是否撞到垂直表面（如墙壁），这通常表示步态不当。
    
    Args:
        env: 强化学习环境实例
        sensor_cfg: 传感器配置
    
    Returns:
        torch.Tensor: 脚部绊倒惩罚值
    """
    # 提取使用的量（启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取Z方向和XY方向的接触力
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])  # Z方向力
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)  # XY方向力
    
    # 惩罚脚部撞到垂直表面（XY力大于Z力的4倍）
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """奖励摆动脚部达到指定的离地高度。
    
    鼓励脚部在摆动时保持合适的离地高度，避免拖地。
    
    Args:
        env: 强化学习环境实例
        command_name: 命令名称
        asset_cfg: 资产配置
        target_height: 目标离地高度
        tanh_mult: tanh函数倍数
    
    Returns:
        torch.Tensor: 脚部高度奖励值
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算脚部相对于身体的位置
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    
    # 计算脚部相对于身体的速度
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    
    # 将脚部位置和速度转换到身体坐标系
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    
    # 计算脚部高度误差
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    
    # 计算脚部水平速度的tanh值
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    
    # 计算奖励
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    
    # 只在有移动命令时应用奖励
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    
    # 根据重力方向调整奖励
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """奖励摆动脚部达到指定的离地高度（简化版本）。
    
    使用指数函数计算奖励，鼓励脚部保持合适的离地高度。
    
    Args:
        env: 强化学习环境实例
        asset_cfg: 资产配置
        target_height: 目标离地高度
        std: 标准差
        tanh_mult: tanh函数倍数
    
    Returns:
        torch.Tensor: 脚部离地高度奖励值
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算脚部高度误差
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    
    # 计算脚部水平速度的tanh值
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    
    # 计算奖励
    reward = foot_z_target_error * foot_velocity_tanh
    
    # 使用指数函数计算最终奖励
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚脚部过于接近。
    
    防止脚部在运动过程中过于接近，这可能导致碰撞或不稳定。
    
    Args:
        env: 强化学习环境实例
        threshold: 距离阈值
        asset_cfg: 资产配置
    
    Returns:
        torch.Tensor: 脚部距离惩罚值
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取脚部位置
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # 计算脚部间距离
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    
    # 如果距离小于阈值，返回惩罚值
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """当命令为零时奖励脚部接触。
    
    鼓励机器人在没有移动命令时保持脚部接触地面，保持稳定。
    
    Args:
        env: 强化学习环境实例
        sensor_cfg: 传感器配置
        command_name: 命令名称
    
    Returns:
        torch.Tensor: 脚部接触奖励值
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 检查是否有接触
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # 获取命令范数
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    
    # 计算接触奖励
    reward = torch.sum(is_contact, dim=-1).float()
    
    # 只在命令很小时应用奖励
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚各脚部在空中/地面时间上的方差。
    
    鼓励各脚部保持一致的步态模式，避免步态不协调。
    
    Args:
        env: 强化学习环境实例
        sensor_cfg: 传感器配置
    
    Returns:
        torch.Tensor: 空中时间方差惩罚值
    """
    # 提取使用的量（启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 检查是否启用了空中时间跟踪
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    
    # 计算奖励
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]      # 上次空中时间
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]  # 上次接触时间
    
    # 计算空中时间和接触时间的方差
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)


"""
脚部步态奖励函数。
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """脚部步态奖励。
    
    根据步态相位奖励脚部的接触模式，鼓励自然的步态。
    
    Args:
        env: 强化学习环境实例
        period: 步态周期
        offset: 各脚部的相位偏移
        sensor_cfg: 传感器配置
        threshold: 相位阈值
        command_name: 命令名称
    
    Returns:
        torch.Tensor: 步态奖励值
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 检查是否有接触
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # 计算全局相位
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    
    # 计算各脚部的相位
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    # 计算步态奖励
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        # 判断是否应该处于支撑相
        is_stance = leg_phase[:, i] < threshold
        # 奖励正确的接触模式（支撑相时接触，摆动相时不接触）
        reward += ~(is_stance ^ is_contact[:, i])

    # 只在有移动命令时应用奖励
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    
    return reward


"""
其他奖励函数。
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """关节镜像奖励。
    
    鼓励对称关节保持镜像关系，例如左右腿的对应关节。
    
    Args:
        env: 强化学习环境实例
        asset_cfg: 资产配置
        mirror_joints: 镜像关节对列表
    
    Returns:
        torch.Tensor: 关节镜像奖励值
    """
    # 提取使用的量（启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 缓存关节位置（如果不存在）
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # 为所有关节对缓存关节位置
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 遍历所有关节对
    for joint_pair in env.joint_mirror_joints_cache:
        # 计算每对关节的差异并添加到总奖励中
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    
    # 归一化奖励（防止某些环境中关节对多影响权重）
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    
    return reward
