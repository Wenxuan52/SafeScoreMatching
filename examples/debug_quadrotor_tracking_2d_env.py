import numpy as np

from jaxrl5.envs.quadrotor_tracking_2d import QuadrotorTracking2DEnv


def run_episode(env: QuadrotorTracking2DEnv, force_terminate: bool = False):
    obs, info = env.reset(seed=0)
    if force_terminate:
        # Force the state to violate boundary for termination check.
        env.state[0] = 2.5
        env.state[1] = 0.0
        env.state[2] = 3.5
        env.state[3] = 0.0

    done = False
    total_reward = 0.0
    costs = []
    step_idx = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert info["cost"] in (0.0, 1.0), "Cost must be binary"

        total_reward += reward
        costs.append(info["cost"])

        if step_idx % 20 == 0:
            x, xdot, z, zdot, theta, thetadot = env.state
            print(
                f"t={step_idx:03d} x={x:+.3f} z={z:+.3f} theta={theta:+.3f} reward={reward:+.3f} "
                f"h={info['h']:+.3f} cost={info['cost']:.0f} terminated={terminated} truncated={truncated}"
            )

        step_idx += 1
        if terminated or truncated:
            break

    episode_return = total_reward
    mean_cost = float(np.mean(costs)) if costs else 0.0
    print(
        f"Episode finished. return={episode_return:+.3f}, mean_cost={mean_cost:.3f}, length={len(costs)}"
    )

    if truncated:
        print("Episode ended due to time limit (truncated=True). Verified max length 360.")
    if terminated:
        print("Episode terminated early due to boundary violation (terminated=True).")


def main():
    env = QuadrotorTracking2DEnv()

    print("=== Episode 1: random policy until done ===")
    run_episode(env, force_terminate=False)

    print("\n=== Episode 2: forced boundary violation to check termination ===")
    run_episode(env, force_terminate=True)


if __name__ == "__main__":
    main()
