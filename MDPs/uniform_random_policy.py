import random

class UniformRandomPolicy:
    def __init__(self, env):
        self.env = env

    def get_prob(self, state, action):
        return 1.0 / len(self.env.get_available_actions(state))

    def get_action(self, state, actions):
        return random.choice(actions)