# -*- coding: utf-8 -*-
"""Omniverse UI 扩展示例（中文注释版）

本文件演示如何在 Omniverse / Isaac Sim 中编写一个最基础的 Python 扩展：
1. 扩展的生命周期：on_startup / on_shutdown
2. 如何在启动时创建简单 UI（窗口 + 按钮 + 标签）
3. 如何通过回调函数与 UI 交互

在 `extension.toml` 的 `python.modules` 字段中把本文件注册后，
Omniverse Extension Manager 启用扩展时会实例化 `ExampleExtension`，
然后依次调用：
    on_startup(ext_id)  ->  用户代码初始化
    on_shutdown()       ->  用户代码清理

运行效果：
* 打开 Isaac Sim → Extension Manager → 搜索并启用 unitree_rl_lab 扩展
* 屏幕左上角会弹出名为 “My Window” 的窗口
* 点击 “Add” 按钮计数 +1，点击 “Reset” 清零
"""

import omni.ext  # Omniverse 扩展框架
import omni.ui   # Omniverse UI 组件库


# -----------------------------------------------------------------------------
# 对外公共函数示例：其他扩展可以通过
#   `import unitree_rl_lab.ui_extension_example as ext`
# 来调用此函数。
# -----------------------------------------------------------------------------

def some_public_function(x: int):
    """示例公共函数：计算 x 的 x 次方。"""
    print("[unitree_rl_lab] some_public_function was called with x: ", x)
    return x ** x


# -----------------------------------------------------------------------------
# 核心扩展类：必须继承 omni.ext.IExt
# -----------------------------------------------------------------------------

class ExampleExtension(omni.ext.IExt):
    """最小化的扩展示例。

    Omniverse 在启用扩展时会实例化本类，随后调用 `on_startup`，
    在禁用扩展时调用 `on_shutdown`。
    """

    def on_startup(self, ext_id):
        """扩展启动回调。

        Args:
            ext_id: 当前扩展的唯一标识，可用于查询扩展信息或文件路径。
        """
        print("[unitree_rl_lab] startup")

        # 计数器 State ---------------------------------------------------------
        self._count = 0  # 点击计数，用于演示 UI 状态

        # 创建一个 300x300 的浮动窗口 ----------------------------------------
        self._window = omni.ui.Window("My Window", width=300, height=300)

        # 使用 UI 布局：垂直堆栈 ---------------------------------------------
        with self._window.frame:
            with omni.ui.VStack():
                # 显示计数结果的标签
                label = omni.ui.Label("")

                # ---------------- 回调函数定义 -----------------------------
                def on_click():
                    """Add 按钮点击回调：计数 +1"""
                    self._count += 1
                    label.text = f"count: {self._count}"

                def on_reset():
                    """Reset 按钮点击回调：计数清零"""
                    self._count = 0
                    label.text = "empty"

                # 初始化标签内容
                on_reset()

                # 水平排列两个按钮 ---------------------------------------
                with omni.ui.HStack():
                    omni.ui.Button("Add", clicked_fn=on_click)
                    omni.ui.Button("Reset", clicked_fn=on_reset)

    # -------------------------------------------------------------------------
    # 扩展关闭时调用，用于资源清理
    # -------------------------------------------------------------------------
    def on_shutdown(self):
        print("[unitree_rl_lab] shutdown")
