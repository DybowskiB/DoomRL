import os
import torch
from DDQNAgent import DDQNAgent
from environment import DoomEnvironment
from utils import preprocess_frame, stack_frames
from torch.utils.tensorboard import SummaryWriter


def main(model_name, train_mode=True):
    config_path = os.path.join("scenarios", "defend_the_center.cfg")
    save_path = "models/" + model_name
    num_episodes = 8000
    epsilon = 1.0 if train_mode else 0.0
    epsilon_decay = 0.997
    min_epsilon = 0.1
    stack_size = 4
    learning_rate = 0.00025

    writer = SummaryWriter(log_dir="runs/ddqn_logs") if train_mode else None

    action_map = {0: [0, 0, 1], 1: [1, 0, 0], 2: [0, 1, 0]}
    num_actions = 3
    env = DoomEnvironment(config_path, action_map)
    input_shape = (stack_size, 80, 80)

    agent = DDQNAgent(
        input_shape, num_actions, writer=writer, learning_rate=learning_rate
    )

    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        agent.load(save_path)
    elif not train_mode:
        print("Model not found")

    for episode in range(num_episodes):
        raw_frame = env.reset()
        processed_frame = preprocess_frame(raw_frame)
        stacked_frames = None
        stacked_state, stacked_frames = stack_frames(
            processed_frame, stacked_frames, stack_size
        )

        total_reward = 0
        while True:
            state_tensor = (
                torch.tensor(stacked_state, dtype=torch.float32).to(agent.device)
                / 255.0
            ).unsqueeze(0)

            action = agent.choose_action(state_tensor, epsilon)

            next_raw_frame, reward, done = env.step([action])

            if next_raw_frame is not None:
                next_processed_frame = preprocess_frame(next_raw_frame)
                stacked_state_next, stacked_frames = stack_frames(
                    next_processed_frame, stacked_frames, stack_size
                )

            if train_mode:
                agent.remember(stacked_state, action, reward, stacked_state_next, done)
                agent.train()

            total_reward += reward
            stacked_state = stacked_state_next

            if done:
                break

        if train_mode:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}"
        )

        if train_mode:
            writer.add_scalar("Reward/Episode", total_reward, episode)

            if (episode + 1) % 10 == 0:
                agent.save(save_path)

    if train_mode:
        agent.save(save_path)
        writer.close()

    env.close()


if __name__ == "__main__":
    main("ddqn_agent.pth", train_mode=True)
