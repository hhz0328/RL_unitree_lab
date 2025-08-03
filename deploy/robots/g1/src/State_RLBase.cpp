// -------------------------------
//  State_RLBase.cpp  (中文注释版)
//  功能：在 g1_ctrl 运行态下创建 IsaacLab C++ 环境，
//        加载导出的 ONNX 策略，并把动作写入 DDS LowCmd，
//        供 unitree_mujoco 使用。
// -------------------------------

#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations.h"   // 观测项实现（C++）
#include "isaaclab/envs/mdp/actions/joint_actions.h" // 动作项实现（C++）

// ---------------- 构造函数 ----------------
// 1. 读取 YAML 配置（deploy.yaml）
// 2. 创建 C++ 版 ManagerBasedRLEnv
// 3. 使用 OrtRunner 加载 ONNX 策略
State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    spdlog::info("Initializing State_{}...", state_string);
    // FSM 配置树：param.yaml -> FSM -> 当前 state
    auto cfg = param::config["FSM"][state_string];
    // 解析策略目录（scripts/train 导出）
    auto policy_dir = parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // 创建 RL 环境：读取 deploy.yaml + 机器人 Articulation 适配器
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<g1::subscription::LowState::SharedPtr>>(FSMState::lowstate)
    );
    // OrtRunner 使用 ONNXRuntime 推理（CPU/GPU 由 ORT 自动选择）
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
}

// ---------------- 每个控制周期调用 ----------------
// 1. env->action_manager->processed_actions()   -> 29维关节目标角
// 2. 写入 DDS LowCmd.q() 对应关节 -> unitree_mujoco 读取并驱动仿真
void State_RLBase::run()
{
    // 取得归一化后的动作（已乘scale+offset）
    auto action = env->action_manager->processed_actions();
    // 将角度目标写入 unitree DDS LowCmd（关节顺序由 joint_ids_map 对齐）
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}