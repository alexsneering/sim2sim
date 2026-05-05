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

def get_height_scan(data, model, body_id = 1):
    """
    模拟 Isaac 的 GridPattern 高度扫描：
    在 base 坐标系下，x∈[-0.8, 0.8], y∈[-0.5, 0.5]，步长 0.1
    共 17×11=187 个点。
    """
    # 网格参数（与 Isaac 一致）
    x_range = (-0.8, 0.8)
    y_range = (-0.5, 0.5)
    resolution = 0.1
    nx = int((x_range[1] - x_range[0]) / resolution) + 1   # =17
    ny = int((y_range[1] - y_range[0]) / resolution) + 1   # =11

    # base 状态
    base_pos = data.xpos[body_id].copy()
    base_mat = data.xmat[body_id].reshape(3, 3)

    heights = []
    for i in range(nx):
        for j in range(ny):
            # 局部坐标偏移
            local_offset = np.array([
                x_range[0] + i * resolution,
                y_range[0] + j * resolution,
                0.0
            ])
            # 转到世界坐标
            world_offset = base_mat @ local_offset
            ray_start = base_pos + world_offset
            ray_dir = np.array([0.0, 0.0, -1.0])   # 世界坐标系向下

            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(
                model, data,
                ray_start, ray_dir,
                None,       # geomgroup
                1,          # flg_static
                -1,         # bodyexclude
                geomid
            )

            if geomid[0] == -1:
                h = 0.0
            else:
                h = dist          # 直接使用射线距离

            heights.append(h)

    heights = np.array(heights, dtype=np.float32)    # shape (187,)

    # 与 Isaac 对齐：clip 到 [-1, 1]
    heights = np.clip(heights, -1.0, 1.0)
    print(heights[:5])
    return heights

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
    #obs = np.pad(obs, (0, 235 - 68), mode='constant')
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # 升维成 [1,235] 匹配网络输入

    return obs_tensor