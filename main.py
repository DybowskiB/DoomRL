import os
import torch
from environment import DoomEnvironment
from dqn_network import DQNAgent
from utils import preprocess_frame, stack_frames
from torch.utils.tensorboard import SummaryWriter


def main(model_name, train_mode=True):
    # Paths and configuration
    config_path = os.path.join("scenarios", "basic.cfg")
    save_path = "models/" + model_name
    num_episodes = 1000
    epsilon = 1.0 if train_mode else 0.0
    epsilon_decay = 0.997
    min_epsilon = 0.1
    stack_size = 4
    learning_rate = 0.00025

    writer = SummaryWriter() if train_mode else None

    # Initialize the environment
    action_map = {0: [0, 0, 1], 1: [1, 0, 0], 2: [0, 1, 0]}
    num_actions = 3
    env = DoomEnvironment(config_path, action_map)
    input_shape = (stack_size, 80, 80)

    # Initialize the agent
    agent = DQNAgent(
        input_shape, num_actions, writer=writer, learning_rate=learning_rate
    )

    # Load the model if it exists
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        agent.load(save_path)
    elif not train_mode:
        print("Model not found")

    # Main training loop
    for episode in range(num_episodes):
        # Reset the environment and preprocess the first frame
        raw_frame = env.reset()
        processed_frame = preprocess_frame(raw_frame)
        stacked_frames = None
        stacked_state, stacked_frames = stack_frames(
            processed_frame, stacked_frames, stack_size
        )

        total_reward = 0
        while True:
            # Preprocess the stacked state for the model
            state_tensor = (
                torch.tensor(stacked_state, dtype=torch.float32).to(agent.device)
                / 255.0
            )
            state_tensor = state_tensor.unsqueeze(0)
            # Shape: [Batch Size, Channels, Height, Width]

            # Choose an action
            action = agent.choose_action(state_tensor, epsilon)

            # Take a step in the environment
            next_raw_frame, reward, done = env.step([action])

            if next_raw_frame is not None:
                next_processed_frame = preprocess_frame(next_raw_frame)
                stacked_state_next, stacked_frames = stack_frames(
                    next_processed_frame, stacked_frames, stack_size
                )

            if train_mode:
                # Store the experience in the agent's memory
                agent.remember(stacked_state, action, reward, stacked_state_next, done)

                # Train the agent
                agent.train()

            total_reward += reward
            stacked_state = stacked_state_next

            if done:
                break

        if train_mode:
            # Decrease epsilon for exploration
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print episode summary
        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}"
        )

        if train_mode:
            writer.add_scalar("Reward/Episode", total_reward, episode)
            writer.add_scalar("Epsilon", epsilon, episode)

            # Save the model every 10 episodes
            if (episode + 1) % 10 == 0:
                agent.save(save_path)

    if train_mode:
        # Save the model after finishing training
        agent.save(save_path)
        writer.close()

    env.close()


if __name__ == "__main__":
    main("dqn_vizdoom.pth", train_mode=True)
