import random

from policies.abstract_policy import AbstractPolicy

class GreedyPolicy(AbstractPolicy):
    def __init__(self, env, v_values, gamma):
        self.env = env
        self.v_values = v_values
        self.gamma = gamma


    def __get_optimal_action(self, state):
        max_v_value = self.__get_optimal_value(state)
        optimal_actions = [action for action in self.env.get_available_actions(state) 
                           if self.__calculate_sum(state, action) == max_v_value]
        #return random.choice(optimal_actions)
        return optimal_actions[0] #break ties arbitrarily --> first action selected
    
    def __get_optimal_value(self, state):
        action_space = self.env.get_available_actions(state)
        return max(self.__calculate_sum(state, action) for action in action_space)

    def __calculate_sum(self, state, action):
        transitions = self.env.get_transitions(state, action)
        return sum(prob * (reward + self.gamma * self.v_values[next_state]) 
                   for prob, next_state, reward in transitions)

    def get_prob(self, state, action):
        return 1.0 if action == self.__get_optimal_action(state) else 0.0

    def get_action(self, state):
        return self.__get_optimal_action(state)