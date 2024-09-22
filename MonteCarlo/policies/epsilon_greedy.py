import random
from policies.abstract_policy import AbstractPolicy

class EpsilonGreedy(AbstractPolicy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_optimal_action(self, state, q_values, actions):
        return max(actions, key=lambda action: q_values.get((state, action), 0))

    def get_action(self, state, q_values, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.get_optimal_action(state, q_values, actions)