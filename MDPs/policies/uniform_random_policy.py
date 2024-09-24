import random

from policies.abstract_policy import AbstractPolicy

class UniformRandomPolicy:
    def __init__(self, env):
        self.env = env

    def get_prob(self, state, action):
        probs = {}
        probs[str(action)] = 1.0 / len(self.env.get_available_actions(state))
        #return 1.0 / len(self.env.get_available_actions(state))
        return probs

    def get_action(self, state):
        actions = self.env.get_available_actions(state)
        return random.choice(actions)