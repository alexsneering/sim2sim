import mujoco
import mujoco.viewer
import numpy as np
import torch

import get_obs as go
import action2targetq as a2tq
import Actor as Ac

actor = Ac.Actor()
actor.eval()

checkpoint = torch.load("model_12700.pt", map_location=torch.device("cpu"))
actor_weights = checkpoint["actor_state_dict"]
actor_weights = {k: v for k, v in actor_weights.items() if k != "distribution.std_param"}
actor.load_state_dict(actor_weights)

model = mujoco.MjModel.from_xml_path("unitree_go1/scene.xml")
data = mujoco.MjData(model)

leg_pose = np.array([0.0, 0.8, -1.5])
target_q = np.tile(leg_pose, 4)
default_q = np.array([0, 0.9, -1.8] * 4)

last_action = np.zeros(12, dtype=np.float32)
'''
viewer =  mujoco.viewer.launch_passive(model, data)
with torch.no_grad():
    while viewer.is_running():
        cmd_vx, cmd_vy, cmd_yaw_rate = 0.0, 0.0, 0.0
        command = np.array([cmd_vx, cmd_vy, cmd_yaw_rate])
        obs = go.get_obs(data, model, default_q, command, last_action)
        action = actor(obs)
        last_action = action.cpu().numpy().squeeze()

        target_q = a2tq.action_to_target_q(action.detach().cpu().numpy().squeeze(), default_q)
        print(target_q)
        data.ctrl[:] = target_q
        mujoco.mj_step(model, data)
        viewer.sync()
'''
decimation = 4
step_counter = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    with torch.no_grad():
        while viewer.is_running():
            if step_counter % decimation == 0:
                command = np.array([0.0, 0.0, 0.0])
                obs = go.get_obs(data, model, default_q, command, last_action)
                action = actor(obs)
                last_action = action.numpy().squeeze()
                target_q = a2tq.action_to_target_q(last_action, default_q)

            if(step_counter < 1000):
                data.ctrl[:] = default_q
            else:
                data.ctrl[:] = target_q          # 假设 XML 中 actuator 是 position 类型
            mujoco.mj_step(model, data)
            step_counter += 1
            viewer.sync()