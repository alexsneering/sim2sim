import numpy as np
import mujoco
import torch

def uniform_noise(x, low, high):
    #return x + np.random.uniform(low, high, size=x.shape)
    return x + np.random.uniform(0, 0, size=x.shape) 

def get_base_lin_vel(data):
    v_world = data.qvel[0:3]
    R = data.xmat[1].reshape(3, 3)
    v_body = R.T @ v_world
    return uniform_noise(v_body, -0.1, 0.1)

def get_base_ang_vel(data):
    w_world = data.qvel[3:6]
    R = data.xmat[1].reshape(3, 3)
    w_body = R.T @ w_world
    return uniform_noise(w_body, -0.2, 0.2)

def get_projected_gravity(data):
    R = data.xmat[1].reshape(3, 3)
    g = np.array([0, 0, -1])
    g_body = R.T @ g
    return uniform_noise(g_body, -0.05, 0.05)

def get_joint_pos(data, default_q):
    q = data.qpos[7:]
    return uniform_noise(q - default_q, -0.01, 0.01)

def get_joint_vel(data):
    qd = data.qvel[6:]
    return uniform_noise(qd, -1.5, 1.5)

def get_height_scan(data, model, n_points=20, radius=0.3):
    heights = []

    base_pos = data.xpos[1].copy()  # base

    for i in range(n_points):
        angle = 2 * np.pi * i / n_points

        offset = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0
        ])

        ray_start = base_pos + offset
        ray_dir = np.array([0.0, 0.0, -1.0])

        # === 关键：创建输出容器 ===
        geomid = np.array([-1], dtype=np.int32)

        # === 调用 mj_ray ===
        dist = mujoco.mj_ray(
            model,
            data,
            ray_start,
            ray_dir,
            None,      # geomgroup
            1,         # flg_static
            -1,        # bodyexclude
            geomid     # 输出 geom id
        )

        # === 计算高度 ===
        if geomid[0] == -1:
            h = 0.0
        else:
            h = dist   # MuJoCo 返回的是距离！

        heights.append(h)

    heights = np.array(heights)

    # clip（和 Isaac 一样）
    #heights = np.clip(heights, -1.0, 1.0)
    height_scan = np.zeros(20)

    return height_scan

def get_obs(data, model, default_q, command, last_action):
    obs = np.concatenate([
        get_base_lin_vel(data),
        get_base_ang_vel(data),
        get_projected_gravity(data),
        command,
        get_joint_pos(data, default_q),
        get_joint_vel(data),
        last_action,
        get_height_scan(data, model)
    ])
    obs = np.pad(obs, (0, 235 - 68), mode='constant')
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # 升维成 [1,235] 匹配网络输入

    return obs_tensor