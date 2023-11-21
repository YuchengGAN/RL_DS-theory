import random

import tensorflow as tf
import numpy as np
from collections import deque


class DQNAgent:
    def __init__(self, model, nb_actions, gamma=0.9, learning_rate=0.0001):
        self.model = model
        self.act_dim = nb_actions
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

    def memorize(self, data, reward, terminal, info):
        self.memory.append((data, reward, terminal, info))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.act_dim)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for data, reward, terminal, info in minibatch:
            target = reward
            if not terminal:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


