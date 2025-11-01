import safety_gymnasium

def main():
    env = safety_gymnasium.make('SafetyCarButton1-v0', render_mode=None)
    obs, info = env.reset()
    print("\n✅ Environment reset successfully.")
    step_in_ep, ep_idx = 0, 0

    for step in range(3005):  # 跑多个 episode 看规律
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        step_in_ep += 1

        # 检查是否终止或截断
        if terminated or truncated:
            ep_idx += 1
            print(f"\n🚩 Episode {ep_idx} ended at step_in_ep = {step_in_ep}")
            print(f"  terminated={terminated}, truncated={truncated}")
            print(f"  Info: {info}")
            print("-" * 60)

            # 重置
            obs, info = env.reset()
            step_in_ep = 0

    env.close()
    print("\n✅ Finished testing.")

if __name__ == "__main__":
    main()
