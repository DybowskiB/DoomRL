import random
import numpy as np
import torch
from torch import nn, optim
import os
from collections import deque

from dqn_network import DQN


class DDQNAgent:
    def __init__(self, input_shape, num_actions, writer=None, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_shape, num_actions).to(self.device)
        self.target_model = DQN(input_shape, num_actions).to(self.device)
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 32
        self.update_target_freq = 1000
        self.steps = 0
        self.writer = writer

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.model.fc[-1].out_features - 1)

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        self.model.train()

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device) / 255.0
        )
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = (
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
            / 255.0
        )
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q_values = (
                self.target_model(next_states)
                .gather(1, next_actions.unsqueeze(-1))
                .squeeze(-1)
            )
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.writer:
            self.writer.add_scalar("Loss/Step", loss.item(), self.steps)

        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"Model loaded from {load_path}")
