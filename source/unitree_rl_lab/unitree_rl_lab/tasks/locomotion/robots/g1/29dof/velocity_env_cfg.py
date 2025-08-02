import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# ==================== 地形配置 ====================
# 鹅卵石道路地形生成器配置
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),              # 地形大小 (长, 宽)
    border_width=20.0,            # 边界宽度
    num_rows=9,                   # 地形行数
    num_cols=21,                  # 地形列数
    horizontal_scale=0.1,         # 水平缩放
    vertical_scale=0.005,         # 垂直缩放
    slope_threshold=0.75,         # 坡度阈值
    difficulty_range=(0.0, 1.0),  # 难度范围
    use_cache=False,              # 不使用缓存
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),  # 50%平坦地形
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """足式机器人的地形场景配置。"""

    # 地面地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # 使用鹅卵石道路地形生成器
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,  # 最大初始地形等级
        collision_group=-1,                  # 碰撞组
        # 物理材质配置
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",    # 摩擦组合模式
            restitution_combine_mode="multiply", # 弹性组合模式
            static_friction=1.0,                 # 静摩擦系数
            dynamic_friction=1.0,                # 动摩擦系数
        ),
        # 视觉材质配置
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,                    # 投影UVW
            texture_scale=(0.25, 0.25),          # 纹理缩放
        ),
        debug_vis=False,                         # 调试可视化
    )
    
    # 机器人配置
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 传感器配置
    # 高度扫描器：用于检测地形高度
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # 安装在躯干上
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 偏移位置
        attach_yaw_only=True,                    # 只附加偏航角
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 网格模式
        debug_vis=False,                         # 调试可视化
        mesh_prim_paths=["/World/ground"],       # 检测地面网格
    )
    
    # 接触力传感器：检测机器人各部分的接触
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",     # 检测所有机器人部件
        history_length=3,                        # 历史长度
        track_air_time=True                      # 跟踪空中时间
    )
    
    # 灯光配置
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,                     # 光照强度
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",  # 天空纹理
        ),
    )


@configclass
class EventCfg:
    """事件配置：定义训练过程中的各种随机化事件。"""

    # ==================== 启动时事件 ====================
    # 随机化物理材质：增加训练的鲁棒性
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",                         # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 应用到所有机器人部件
            "static_friction_range": (0.3, 1.0),    # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 1.0),   # 动摩擦系数范围
            "restitution_range": (0.0, 0.0),        # 弹性系数范围
            "num_buckets": 64,                      # 分桶数量
        },
    )

    # 随机化机器人质量：增加训练多样性
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",                         # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 应用到躯干
            "mass_distribution_params": (-1.0, 3.0),  # 质量分布参数
            "operation": "add",                     # 操作类型：添加
        },
    )

    # ==================== 重置时事件 ====================
    # 施加外部力和力矩：模拟外部干扰
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",                           # 重置时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 应用到躯干
            "force_range": (0.0, 0.0),             # 力范围（当前为0）
            "torque_range": (-0.0, 0.0),           # 力矩范围（当前为0）
        },
    )

    # 重置机器人基座状态：随机化初始位置和姿态
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",                           # 重置时执行
        params={
            "pose_range": {
                "x": (-0.5, 0.5),    # X位置范围
                "y": (-0.5, 0.5),    # Y位置范围
                "yaw": (-3.14, 3.14) # 偏航角范围
            },
            "velocity_range": {
                "x": (0.0, 0.0),     # X速度范围
                "y": (0.0, 0.0),     # Y速度范围
                "z": (0.0, 0.0),     # Z速度范围
                "roll": (0.0, 0.0),  # 横滚角速度范围
                "pitch": (0.0, 0.0), # 俯仰角速度范围
                "yaw": (0.0, 0.0),   # 偏航角速度范围
            },
        },
    )

    # 重置机器人关节：随机化关节状态
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",                           # 重置时执行
        params={
            "position_range": (1.0, 1.0),         # 位置范围（保持默认），只希望关节速度有扰动，而不希望关节位置乱动
            "velocity_range": (-1.0, 1.0),        # 速度范围
        },
    )

    # ==================== 间隔事件 ====================
    # 推动机器人：模拟外部推力干扰
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",                        # 间隔执行
        interval_range_s=(5.0, 5.0),            # 间隔时间范围
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # 推力速度范围
    )


@configclass
class CommandsCfg:
    """MDP命令规范：定义机器人需要执行的速度命令。"""

    # 基座速度命令：控制机器人的移动
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",                     # 目标资产名称
        resampling_time_range=(10.0, 10.0),     # 重新采样时间范围
        rel_standing_envs=0.02,                 # 站立环境比例
        rel_heading_envs=1.0,                   # 朝向环境比例
        heading_command=False,                  # 不使用朝向命令
        debug_vis=True,                         # 调试可视化
        # 速度范围
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),    # 线速度X范围
            lin_vel_y=(-0.1, 0.1),    # 线速度Y范围
            ang_vel_z=(-0.1, 0.1)     # 角速度Z范围
        ),
        # 限制范围
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),    # 线速度X限制范围
            lin_vel_y=(-0.3, 0.3),    # 线速度Y限制范围
            ang_vel_z=(-0.2, 0.2)     # 角速度Z限制范围
        ),
    )


@configclass
class ActionsCfg:
    """MDP动作规范：定义机器人的动作空间。"""

    # 关节位置动作：控制所有关节的位置
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",                     # 目标资产名称
        joint_names=[".*"],                     # 所有关节
        scale=0.25,                            # 动作缩放
        use_default_offset=True                # 使用默认偏移
    )


@configclass
class ObservationsCfg:
    """MDP观察规范：定义智能体能够观察到的信息。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        """策略网络的观察组。"""

        # 观察项（保持顺序）
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,              # 基座角速度
            scale=0.2,                          # 缩放因子
            noise=Unoise(n_min=-0.2, n_max=0.2) # 噪声
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,         # 投影重力
            noise=Unoise(n_min=-0.05, n_max=0.05) # 噪声
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,        # 速度命令
            params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,             # 相对关节位置
            noise=Unoise(n_min=-0.01, n_max=0.01) # 噪声
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,             # 相对关节速度
            scale=0.05,                         # 缩放因子
            noise=Unoise(n_min=-1.5, n_max=1.5) # 噪声
        )
        last_action = ObsTerm(func=mdp.last_action)  # 上一个动作
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})  # 步态相位（注释掉）

        def __post_init__(self):
            self.history_length = 5             # 历史长度
            self.enable_corruption = True       # 启用数据损坏
            self.concatenate_terms = True       # 连接观察项

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        """评论家网络的观察组。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 基座线速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)  # 基座角速度
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 投影重力
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,        # 速度命令
            params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)  # 相对关节位置
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)  # 相对关节速度
        last_action = ObsTerm(func=mdp.last_action)  # 上一个动作
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})  # 步态相位（
        # height_scanner = ObsTerm(func=mdp.height_scan,  # 高度扫描
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5             # 历史长度

    # privileged observations
    # 特权观察（评论家使用）
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """MDP奖励项：定义训练过程中的奖励函数。"""

    # ==================== 任务奖励 ====================
    # 跟踪线速度：鼓励机器人按照命令移动
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,                             # 权重
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 跟踪角速度：鼓励机器人按照命令转向
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5,                             # 权重
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # 存活奖励：鼓励机器人保持站立
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # ==================== 基座奖励 ====================
    # 惩罚Z轴线速度：防止机器人上升或下降
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 惩罚XY轴角速度：防止机器人倾斜
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 惩罚关节速度：鼓励平滑运动
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # 惩罚关节加速度：鼓励平滑运动
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 惩罚动作变化率：鼓励平滑控制
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # 惩罚关节位置超出限制
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # 惩罚能量消耗：鼓励节能
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # ==================== 关节偏差奖励 ====================
    # 惩罚手臂关节偏差：保持手臂在合理位置
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,                            # 权重
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",     # 肩关节
                    ".*_elbow_joint",           # 肘关节
                    ".*_wrist_.*",              # 腕关节
                ],
            )
        },
    )
    # 惩罚腰部关节偏差：保持腰部稳定
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,                              # 权重
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",                  # 腰部关节
                ],
            )
        },
    )
    # 惩罚腿部关节偏差：保持腿部稳定
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,                            # 权重
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # ==================== 机器人姿态奖励 ====================
    # 惩罚不平坦的姿态：鼓励保持直立
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # 惩罚基座高度偏差：保持合适的高度
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # ==================== 脚部奖励 ====================
    # 步态奖励：鼓励自然的步态
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,                             # 权重
        params={
            "period": 0.8,                      # 步态周期
            "offset": [0.0, 0.5],               # 偏移
            "threshold": 0.55,                  # 阈值
            "command_name": "base_velocity",    # 命令名称
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  # 传感器配置
        },
    )
    # 惩罚脚部滑动：防止脚部在地面上滑动
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,                            # 权重
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),  # 机器人配置
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  # 传感器配置
        },
    )
    # 脚部离地高度奖励：鼓励合适的抬脚高度
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,                             # 权重
        params={
            "std": 0.05,                        # 标准差
            "tanh_mult": 2.0,                   # tanh倍数，双曲正切函数，激活函数
            "target_height": 0.1,               # 目标高度
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),  # 机器人配置
        },
    )

    # ==================== 其他奖励 ====================
    # 惩罚不期望的接触：防止机器人其他部位接触地面
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,                              # 权重
        params={
            "threshold": 1,                     # 阈值
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),  # 排除脚踝的接触
        },
    )


@configclass
class TerminationsCfg:
    """MDP终止条件：定义何时结束一个训练回合。"""

    # 超时终止：达到最大时间
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 基座高度过低终止：机器人摔倒
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # 姿态过差终止：机器人姿态异常
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """MDP课程学习：定义训练难度的渐进式增加。"""

    # 地形等级课程：逐渐增加地形难度
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # 线速度命令等级课程：逐渐增加速度要求
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """腿式机器人速度跟踪环境的配置。"""

    # 场景设置
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)  # 4096个并行环境
    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()  # 观察配置
    actions: ActionsCfg = ActionsCfg()                 # 动作配置
    commands: CommandsCfg = CommandsCfg()              # 命令配置
    # MDP设置
    rewards: RewardsCfg = RewardsCfg()                 # 奖励配置
    terminations: TerminationsCfg = TerminationsCfg()  # 终止配置
    events: EventCfg = EventCfg()                      # 事件配置
    curriculum: CurriculumCfg = CurriculumCfg()        # 课程配置

    def __post_init__(self):
        """后初始化：设置仿真参数。"""
        # 基本设置
        self.decimation = 4                           # 降采样因子
        self.episode_length_s = 20.0                  # 回合长度（秒）
        
        # 仿真设置
        self.sim.dt = 0.005                           # 仿真时间步长
        self.sim.render_interval = self.decimation    # 渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material  # 物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # GPU最大刚体补丁数量

        # 更新传感器更新周期
        # 基于最小的更新周期（物理更新周期）来更新所有传感器
        self.scene.contact_forces.update_period = self.sim.dt  # 接触力传感器更新周期
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 高度扫描器更新周期

        # 检查是否启用地形等级课程学习
        # 如果启用了，则启用地形生成器的课程学习
        # 这会生成难度逐渐增加的地形，对训练很有用
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """用于测试和演示的环境配置。"""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32                     # 减少环境数量用于测试
        self.scene.terrain.terrain_generator.num_rows = 2   # 减少地形行数
        self.scene.terrain.terrain_generator.num_cols = 10  # 减少地形列数
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges  # 使用限制范围作为命令范围
