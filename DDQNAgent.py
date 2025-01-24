import random
import numpy as np
import torch
from torch import nn, optim
import os
from collections import deque

from dqn_network import DQN, DQNAgent


# Klasa reprezentująca agenta stosowanego do Double DQN
class DDQNAgent(DQNAgent):
    def __init__(self, input_shape, num_actions, writer=None, learning_rate=0.0001):
        super(DDQNAgent, self).__init__(
            input_shape, num_actions, writer, learning_rate, DQN
        )

    # Jeden krok treningu agenta - zmodyfikowany w stosunku do podstawowego algorytmu DQN
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
            # Zmodyfikowany sposób wyliczania wartości Q
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
