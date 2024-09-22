from abc import ABC, abstractmethod

class AbstractPolicy(ABC):
    @abstractmethod
    def get_optimal_action(self, state, q_values, actions):
        pass

    @abstractmethod
    def get_action(self, state, q_values, actions):
        pass