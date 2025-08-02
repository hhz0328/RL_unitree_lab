# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree机器人配置文件。

这个文件定义了各种Unitree机器人在Isaac Sim中的配置，
包括物理属性、关节参数、执行器设置等。

参考: https://github.com/unitreerobotics/unitree_ros
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg  # noqa: F401
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Unitree机器人关节配置类。
    
    继承自ArticulationCfg，添加了joint_sdk_names字段，
    用于映射关节名称到SDK中的名称。
    """

    joint_sdk_names: list[str] | None = None  # SDK中的关节名称列表


# 机器人模型文件路径
# UNITREE_MODEL_DIR = MISSING
UNITREE_MODEL_DIR = "/home/hhz/WU/2025/g1/rl_lab/unitree_rl_lab/unitree_model"

# ==================== Go2四足机器人配置 ====================
UNITREE_GO2_CFG = UnitreeArticulationCfg(
    # 生成配置：指定USD模型文件和物理属性
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",  # Go2机器人的USD模型文件
        activate_contact_sensors=True,  # 启用接触传感器
        # 刚体属性配置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 不禁用重力
            retain_accelerations=False,  # 不保留加速度
            linear_damping=0.0,  # 线性阻尼
            angular_damping=0.0,  # 角阻尼
            max_linear_velocity=100.0,  # 最大线速度
            max_angular_velocity=100.0,  # 最大角速度
            max_depenetration_velocity=1.0,  # 最大分离速度
        ),
        # 关节属性配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # 启用自碰撞检测
            solver_position_iteration_count=4,  # 位置求解器迭代次数
            solver_velocity_iteration_count=0  # 速度求解器迭代次数
        ),
    ),
    # 初始状态配置
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # 初始位置 (x, y, z)
        # 关节初始位置（弧度）
        joint_pos={
            ".*R_hip_joint": -0.1,  # 右髋关节
            ".*L_hip_joint": 0.1,   # 左髋关节
            "F[L,R]_thigh_joint": 0.8,  # 前腿大腿关节
            "R[L,R]_thigh_joint": 1.0,  # 后腿大腿关节
            ".*_calf_joint": -1.5,  # 小腿关节
        },
        joint_vel={".*": 0.0},  # 所有关节初始速度为0
    ),
    soft_joint_pos_limit_factor=0.9,  # 软关节位置限制因子
    # 执行器配置
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],  # 腿部关节
            effort_limit_sim=23.5,  # 力矩限制
            velocity_limit_sim=30.0,  # 速度限制
            stiffness=25.0,  # 刚度
            damping=0.5,  # 阻尼
            friction=0.01,  # 摩擦
            armature=0.005,  # 电枢
        ),
    },
    # fmt: off
    # SDK中的关节名称映射（用于实际机器人控制）
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",      # 右前腿
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",      # 左前腿
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",      # 右后腿
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"       # 左后腿
    ],
    # fmt: on
)

# ==================== Go2W轮式四足机器人配置 ====================
UNITREE_GO2W_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2W/usd/go2w.usd",  # Go2W机器人的USD模型文件
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),  # 稍微高一点，因为有轮子
        joint_pos={
            ".*L_hip_joint": 0.0,  # 左髋关节
            ".*R_hip_joint": -0.0,  # 右髋关节
            "F.*_thigh_joint": 0.8,  # 前腿大腿关节
            "R.*_thigh_joint": 0.8,  # 后腿大腿关节
            ".*_calf_joint": -1.5,  # 小腿关节
            ".*_foot_joint": 0.0,  # 轮子关节
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿部执行器（与Go2相同）
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            armature=0.005,
        ),
        # 轮子执行器（Go2W特有）
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],  # 轮子关节
            effort_limit_sim=23.7,
            velocity_limit_sim=30.1,
            stiffness=0.0,  # 轮子不需要刚度
            damping=0.5,
            friction=0.0,  # 轮子摩擦为0
        ),
    },
    # fmt: off
    # SDK关节名称映射（包含轮子）
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",      # 右前腿
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",      # 左前腿
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",      # 右后腿
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",      # 左后腿
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"  # 四个轮子
    ],
    # fmt: on
)

# ==================== H1人形机器人配置 ====================
UNITREE_H1_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/H1/h1/usd/h1.usd",  # H1机器人的USD模型文件
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,  # H1速度限制更高
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 人形机器人禁用自碰撞
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),  # 人形机器人站立高度
        joint_pos={
            ".*_hip_pitch_joint": -0.1,  # 髋关节俯仰
            ".*_knee_joint": 0.3,        # 膝关节
            ".*_ankle_joint": -0.2,      # 踝关节
            ".*_shoulder_pitch_joint": 0.20,  # 肩关节俯仰
            ".*_elbow_joint": 0.32,      # 肘关节
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿部执行器
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",    # 髋关节偏航
                ".*_hip_roll_joint",   # 髋关节横滚
                ".*_hip_pitch_joint",  # 髋关节俯仰
                ".*_knee_joint",       # 膝关节
                "torso_joint",         # 躯干关节
            ],
            # 不同关节的力矩限制
            effort_limit_sim={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,  # 膝关节力矩最大
                "torso_joint": 200.0,
            },
            # 不同关节的速度限制
            velocity_limit_sim={
                ".*_hip_yaw_joint": 23.0,
                ".*_hip_roll_joint": 23.0,
                ".*_hip_pitch_joint": 23.0,
                ".*_knee_joint": 14.0,  # 膝关节速度较慢
                "torso_joint": 23.0,
            },
            # 不同关节的刚度
            stiffness={
                ".*_hip_.*_joint": 150.0,
                ".*_knee_joint": 200.0,  # 膝关节刚度最高
                "torso_joint": 300.0,    # 躯干刚度最高
            },
            # 不同关节的阻尼
            damping={
                ".*_hip_.*_joint": 2.0,
                ".*_knee_joint": 4.0,    # 膝关节阻尼较大
                "torso_joint": 6.0,      # 躯干阻尼最大
            },
        ),
        # 脚部执行器
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],  # 踝关节
            effort_limit_sim=40.0,
            velocity_limit_sim=9.0,
            stiffness=40.0,
            damping=2.0,
        ),
        # 手臂执行器
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",  # 肩关节俯仰
                ".*_shoulder_roll_joint",   # 肩关节横滚
                ".*_shoulder_yaw_joint",    # 肩关节偏航
                ".*_elbow_joint",           # 肘关节
            ],
            # 手臂关节的力矩限制
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,
            },
            # 手臂关节的速度限制
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            # 手臂关节的刚度
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
            },
            # 手臂关节的阻尼
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
        ),
    },
    # SDK关节名称映射（29个关节）
    joint_sdk_names=[
        "right_hip_roll_joint",      # 右髋横滚
        "right_hip_pitch_joint",     # 右髋俯仰
        "right_knee_joint",          # 右膝
        "left_hip_roll_joint",       # 左髋横滚
        "left_hip_pitch_joint",      # 左髋俯仰
        "left_knee_joint",           # 左膝
        "torso_joint",               # 躯干
        "left_hip_yaw_joint",        # 左髋偏航
        "right_hip_yaw_joint",       # 右髋偏航
        "",                          
        "left_ankle_joint",          # 左踝
        "right_ankle_joint",         # 右踝
        "right_shoulder_pitch_joint", # 右肩俯仰
        "right_shoulder_roll_joint",  # 右肩横滚
        "right_shoulder_yaw_joint",   # 右肩偏航
        "right_elbow_joint",          # 右肘
        "left_shoulder_pitch_joint",  # 左肩俯仰
        "left_shoulder_roll_joint",   # 左肩横滚
        "left_shoulder_yaw_joint",    # 左肩偏航
        "left_elbow_joint",           # 左肘
    ],
)

# ==================== G1 29自由度人形机器人配置 ====================
UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",  # G1机器人的USD模型文件
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # G1启用自碰撞检测
            solver_position_iteration_count=8,  # 更多迭代次数
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),  # G1的站立高度
        joint_pos={
            "left_hip_pitch_joint": -0.1,   # 左髋俯仰
            "right_hip_pitch_joint": -0.1,  # 右髋俯仰
            ".*_knee_joint": 0.3,           # 膝关节
            ".*_ankle_pitch_joint": -0.2,   # 踝关节俯仰
            ".*_shoulder_pitch_joint": 0.3, # 肩关节俯仰
            "left_shoulder_roll_joint": 0.25,   # 左肩横滚
            "right_shoulder_roll_joint": -0.25, # 右肩横滚
            ".*_elbow_joint": 0.97,         # 肘关节
            "left_wrist_roll_joint": 0.15,  # 左腕横滚
            "right_wrist_roll_joint": -0.15, # 右腕横滚
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿部执行器（包含腰部）
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",    # 髋关节横滚
                ".*_hip_yaw_joint",     # 髋关节偏航
                ".*_hip_pitch_joint",   # 髋关节俯仰
                ".*_knee_joint",        # 膝关节
                "waist_.*_joint",       # 腰部关节
            ],
            effort_limit_sim=300,       # 腿部力矩限制
            velocity_limit_sim=100.0,   # 腿部速度限制
            # 不同关节的刚度
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,     # 膝关节刚度较高
                "waist_.*_joint": 200.0,    # 腰部刚度最高
            },
            # 不同关节的阻尼
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,       # 膝关节阻尼较大
                "waist_.*_joint": 5.0,      # 腰部阻尼最大
            },
            # 不同关节的电枢
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_.*_joint": 0.01,
            },
        ),
        # 脚部执行器
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,        # 脚部力矩较小
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],  # 踝关节俯仰和横滚
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        # 手臂执行器（包含手腕）
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",  # 肩关节俯仰
                ".*_shoulder_roll_joint",   # 肩关节横滚
                ".*_shoulder_yaw_joint",    # 肩关节偏航
                ".*_elbow_joint",           # 肘关节
                ".*_wrist_roll_joint",      # 腕关节横滚
                ".*_wrist_pitch_joint",     # 腕关节俯仰
                ".*_wrist_yaw_joint",       # 腕关节偏航
            ],
            effort_limit_sim=300,       # 手臂力矩限制
            velocity_limit_sim=100.0,   # 手臂速度限制
            stiffness=40.0,             # 手臂刚度
            damping=10.0,               # 手臂阻尼
            # 不同关节的电枢
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    },
    # SDK关节名称映射（29个关节）
    joint_sdk_names=[
        # 左腿 (6个关节)
        "left_hip_pitch_joint",      # 左髋俯仰
        "left_hip_roll_joint",       # 左髋横滚
        "left_hip_yaw_joint",        # 左髋偏航
        "left_knee_joint",           # 左膝
        "left_ankle_pitch_joint",    # 左踝俯仰
        "left_ankle_roll_joint",     # 左踝横滚
        # 右腿 (6个关节)
        "right_hip_pitch_joint",     # 右髋俯仰
        "right_hip_roll_joint",      # 右髋横滚
        "right_hip_yaw_joint",       # 右髋偏航
        "right_knee_joint",          # 右膝
        "right_ankle_pitch_joint",   # 右踝俯仰
        "right_ankle_roll_joint",    # 右踝横滚
        # 腰部 (3个关节)
        "waist_yaw_joint",           # 腰部偏航
        "waist_roll_joint",          # 腰部横滚
        "waist_pitch_joint",         # 腰部俯仰
        # 左臂 (7个关节)
        "left_shoulder_pitch_joint", # 左肩俯仰
        "left_shoulder_roll_joint",  # 左肩横滚
        "left_shoulder_yaw_joint",   # 左肩偏航
        "left_elbow_joint",          # 左肘
        "left_wrist_roll_joint",     # 左腕横滚
        "left_wrist_pitch_joint",    # 左腕俯仰
        "left_wrist_yaw_joint",      # 左腕偏航
        # 右臂 (7个关节)
        "right_shoulder_pitch_joint", # 右肩俯仰
        "right_shoulder_roll_joint",  # 右肩横滚
        "right_shoulder_yaw_joint",   # 右肩偏航
        "right_elbow_joint",          # 右肘
        "right_wrist_roll_joint",     # 右腕横滚
        "right_wrist_pitch_joint",    # 右腕俯仰
        "right_wrist_yaw_joint",      # 右腕偏航
    ],
)


"""Unitree G1 23自由度人形机器人配置。"""
