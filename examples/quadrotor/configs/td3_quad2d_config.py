from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.model_cls = "TD3Learner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.discount = 0.99
    config.tau = 0.005
    config.exploration_noise = 0.1
    config.target_policy_noise = 0.2
    config.target_policy_noise_clip = 0.5
    config.actor_delay = 2
    config.num_qs = 2
    config.num_min_qs = None
    return config
