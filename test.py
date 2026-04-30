import mujoco
import mujoco.viewer
import numpy as np

import torch

md = torch.load("model_12700.pt", map_location="cpu")
print(md["actor_state_dict"].keys())

# === 加载模型 ===
model = mujoco.MjModel.from_xml_path("unitree_go1/scene.xml")
data = mujoco.MjData(model)

# === 站立姿态 ===
leg_pose = np.array([0.0, 0.8, -1.5])
target_q = np.tile(leg_pose, 4)

# === 可视化 ===
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # === 直接给 position actuator 目标角度 ===
        data.ctrl[:] = target_q

        # === 仿真推进 ===
        mujoco.mj_step(model, data)

        viewer.sync()