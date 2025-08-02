# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL命令行参数处理模块。

这个模块负责处理与RSL-RL强化学习框架相关的命令行参数，
包括实验配置、模型加载、日志记录等功能。
"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """向解析器添加RSL-RL相关的命令行参数。

    这个函数为argparse解析器添加了一个专门的参数组，
    包含所有与RSL-RL训练相关的配置选项。

    Args:
        parser: 要添加参数的解析器对象。
    """
    # 创建一个新的参数组
    arg_group = parser.add_argument_group("rsl_rl", description="RSL-RL智能体的参数。")
    
    # -- 实验相关参数
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, 
        help="存储日志的实验文件夹名称。"
    )
    arg_group.add_argument(
        "--run_name", type=str, default=None, 
        help="日志目录的运行名称后缀。"
    )
    
    # -- 模型加载参数
    arg_group.add_argument(
        "--resume", action="store_true", default=False, 
        help="是否从检查点恢复训练。"
    )
    arg_group.add_argument(
        "--load_run", type=str, default=None, 
        help="要恢复的训练运行文件夹名称。"
    )
    arg_group.add_argument(
        "--checkpoint", type=str, default=None, 
        help="要恢复的检查点文件。"
    )
    
    # -- 日志记录参数
    arg_group.add_argument(
        "--logger", type=str, default=None, 
        choices={"wandb", "tensorboard", "neptune"}, 
        help="要使用的日志记录模块。"
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, 
        help="使用wandb或neptune时的日志项目名称。"
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """根据输入解析RSL-RL智能体的配置。

    这个函数从配置注册表中加载默认配置，然后根据命令行参数进行更新。

    Args:
        task_name: 环境名称。
        args_cli: 命令行参数。

    Returns:
        基于输入解析的RSL-RL智能体配置。
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # 加载默认配置
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    
    # 如果实验名称为空，则根据任务名称生成默认名称
    if rslrl_cfg.experiment_name == "":
        rslrl_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    
    # 使用命令行参数更新配置
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """根据命令行参数更新RSL-RL智能体配置。

    这个函数将命令行参数的值覆盖到默认配置中，
    实现配置的动态调整。

    Args:
        agent_cfg: RSL-RL智能体的配置对象。
        args_cli: 命令行参数。

    Returns:
        根据输入更新的RSL-RL智能体配置。
    """
    # 使用命令行参数覆盖默认配置
    
    # 处理随机种子
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # 如果种子为-1，则随机生成一个种子
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    
    # 处理模型恢复相关参数
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    
    # 处理运行名称
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    
    # 处理日志记录器
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    
    # 为wandb和neptune设置项目名称
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    # 如果实验名称为空，根据任务名称生成
    if agent_cfg.experiment_name == "":
        task_name = args_cli.task
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")

    return agent_cfg
