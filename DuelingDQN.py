import torch
from torch import nn

from DQNNetwork import DQN


# Sieć wykorzystywana w modyfikacji Dueling DQN, dziedzicząca po podstawowym DQN
class DuelingDQN(DQN):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__(input_shape, num_actions)

        # Podział na wartość stanu
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # oraz wartość przewagi danej akcji
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    # Nadpisanie metody forward względem podstawowego DQN
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
