import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU(),
            nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # Dynamically compute the size of the flattened layer:
        # Create a dummy input with the same shape as the input and compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output_size = self.conv(dummy_input).view(-1).size(0)

        self.fc = nn.Sequential(
            # nn.Linear(conv_output_size, 512),
            # nn.ReLU(),
            # nn.Linear(512, num_actions),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class DQNAgent:
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
        # self.tau = 0.005
        self.steps = 0
        self.writer = writer

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            # Use the number of output actions
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
            next_q_values = self.target_model(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.writer.add_scalar("Loss/Step", loss.item(), self.steps)

        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # target_model_state_dict = self.target_model.state_dict()
        # model_state_dict = self.model.state_dict()
        # for key in model_state_dict:
        #     target_model_state_dict[key] = model_state_dict[
        #         key
        #     ] * self.tau + target_model_state_dict[key] * (1 - self.tau)
        # self.target_model.load_state_dict(target_model_state_dict)

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"Model loaded from {load_path}")
