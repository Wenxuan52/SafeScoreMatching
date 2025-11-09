# Safe Score Matching (SSM)

This repository is based on the original **Q-Score Matching (QSM)** implementation from:

> *Psenka et al., “Learning a Diffusion Model Policy from Rewards via Q-Score Matching”, ICML 2024.*
> [https://arxiv.org/abs/2312.11752](https://arxiv.org/abs/2312.11752)

We keep the original QSM code structure and training pipeline unchanged, and **extend it with the Safe Score Matching (SSM)** designed for safe reinforcement learning.

## Currently Developing

* New training scripts and configs under:

  ```
  examples/states/train_safe_score_matching.py
  examples/states/configs/safe_score_matching_config.py
  ```
* New agent implementation under:

  ```
  jaxrl5/agents/safe_score_matching/
  ```

