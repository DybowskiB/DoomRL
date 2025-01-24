import os
import torch
from DDQNAgent import DDQNAgent
from DuelingDQN import DuelingDQN
from environment import DoomEnvironment
from DQNNetwork import DQNAgent
from utils import preprocess_frame, stack_frames
from torch.utils.tensorboard import SummaryWriter


def main(model_name, train_mode=True):
    # Lokalizacja plików definiujących scenariusz
    # config_path = os.path.join("scenarios", "basic.cfg")
    config_path = os.path.join("scenarios", "defend_the_center.cfg")
    save_path = "models/" + model_name

    # Parametry treningu agenta
    num_episodes = 100
    epsilon = 1.0 if train_mode else 0.0
    epsilon_decay = 0.997
    min_epsilon = 0.1
    stack_size = 4
    learning_rate = 0.00025

    average_reward = 0

    writer = SummaryWriter() if train_mode else None

    # Inicjalizacja środowiska
    action_map = {0: [0, 0, 1], 1: [1, 0, 0], 2: [0, 1, 0]}
    num_actions = 3
    env = DoomEnvironment(config_path, action_map)
    input_shape = (stack_size, 80, 80)

    # Inicjalizacja agenta
    # agent = DQNAgent(
    #     input_shape, num_actions, writer=writer, learning_rate=learning_rate
    # )
    # agent = DDQNAgent(
    #     input_shape, num_actions, writer=writer, learning_rate=learning_rate
    # )
    agent = DQNAgent(
        input_shape,
        num_actions,
        writer=writer,
        learning_rate=learning_rate,
        model_class=DuelingDQN,
    )

    # Wczytanie istniejącego modulu
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        agent.load(save_path)
    elif not train_mode:
        print("Model not found")

    # Główna pętla treningowa
    for episode in range(num_episodes):
        # Procesowanie pierwszych klatek
        raw_frame = env.reset()
        processed_frame = preprocess_frame(raw_frame)
        stacked_frames = None
        stacked_state, stacked_frames = stack_frames(
            processed_frame, stacked_frames, stack_size
        )

        total_reward = 0
        while True:
            # Przygotowanie tensora reprezetnującego stan
            state_tensor = (
                torch.tensor(stacked_state, dtype=torch.float32).to(agent.device)
                / 255.0
            )
            state_tensor = state_tensor.unsqueeze(0)
            # Shape: [Batch Size, Channels, Height, Width]

            # Wybór i podjęcie akcji
            action = agent.choose_action(state_tensor, epsilon)
            next_raw_frame, reward, done = env.step([action])

            if next_raw_frame is not None:
                # Preprocessing i dołożenie do stosu następnej klatki
                next_processed_frame = preprocess_frame(next_raw_frame)
                stacked_state_next, stacked_frames = stack_frames(
                    next_processed_frame, stacked_frames, stack_size
                )

            if train_mode:
                # Zapamiętanie przez agenta stanu i podjętej akcji
                agent.remember(stacked_state, action, reward, stacked_state_next, done)
                # Trenowanie agenta
                agent.train()

            total_reward += reward
            stacked_state = stacked_state_next

            if done:
                break

        if train_mode:
            # Zmniejszenie epsilona
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}"
        )

        average_reward += total_reward

        if train_mode:
            writer.add_scalar("Reward/Episode", total_reward, episode)
            writer.add_scalar("Epsilon", epsilon, episode)

            # Zapisywanie modelu co 10 epizodów
            if (episode + 1) % 10 == 0:
                agent.save(save_path)

    if train_mode:
        # Zapisanie modelu po całym treningu (na wypadek, gdyby liczba epizodów nie dzieliła się przez 10)
        agent.save(save_path)
        writer.close()

    if not train_mode:
        average_reward /= num_episodes
        print(f"\n\nAverage reward: {average_reward}")

    env.close()


if __name__ == "__main__":
    # Rozpoczęcie treningu agenta zapisanego/mającego być zapisanym w podanym pliku
    main("defend_the_center_duelingdqn_agent_2.pth", train_mode=True)
