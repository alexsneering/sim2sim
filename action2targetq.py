def action_to_target_q(action, default_q):
    scale = 0.25
    return default_q + scale * action