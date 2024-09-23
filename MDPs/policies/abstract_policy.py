from abc import ABC, abstractmethod

class AbstractPolicy(ABC):

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def get_prob(self, state, action):
        pass