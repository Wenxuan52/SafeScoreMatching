#!/usr/bin/env python
"""Lightweight TD3 save/load smoke test for QuadrotorTracking2D.

Usage:
  python examples/quadrotor/debug_td3_save_load.py --tmp_dir /tmp/ckpt_test
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from absl import app
from ml_collections import config_flags

from jaxrl5.agents import TD3Learner
from jaxrl5.envs import make_env


FLAGS = config_flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "examples/quadrotor/configs/td3_quad2d_config.py",
    lock_config=False,
    help_string="TD3 hyperparameters config file.",
)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tmp_dir", default="/tmp/td3_ckpt_test", help="Temporary directory for checkpoint smoke test"
    )
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args(argv[1:])

    tmp_dir = Path(args.tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = make_env("QuadrotorTracking2D-v0", seed=args.seed)
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent: TD3Learner = globals()[model_cls].create(
        args.seed,
        env.observation_space,
        env.action_space,
        **kwargs,
    )

    obs, _ = env.reset(seed=args.seed)
    agent.save(str(tmp_dir), step=0)
    loaded = TD3Learner.load(str(tmp_dir), step=0)
    act_b, _ = loaded.eval_actions(np.asarray(obs[None, :], dtype=np.float32))
    act = np.asarray(act_b[0])
    assert act.shape == env.action_space.shape
    assert np.all(np.isfinite(act))
    print("save/load ok; action:", act)

    env.close()
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    app.run(main)
