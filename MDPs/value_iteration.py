from policies.greedy_policy import GreedyPolicy

class ValueIteration:
    def __init__(self, env, gamma=1.0, theta=0.00001):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.__V = {state: 0.0 for state in self.env.states}
        self.delta = 0.0
        
    def estimate(self):
        while True:
            self.delta = 0
            for state in self.env.states:
                self.__update_v_value(state)
            print(f"Delta: {self.delta}")
            if self.delta < self.theta:
                return GreedyPolicy(self.env, self.__V, self.gamma)
    
    def __update_v_value(self, state):
        v = self.__V[state] 
        available_actions = self.env.get_available_actions(state)
        v_for_action = []
        for action in available_actions:
            if self.env.is_terminal(state):
                continue
            action_sum = self.__calculate_sum(state, action)
            v_for_action.append(action_sum)
        self.__V[state] = 0 if self.env.is_terminal(state) else max(v_for_action)
        self.delta = max(self.delta, abs(v - self.__V[state]))

    def __calculate_sum(self, state, action):
        transitions = self.env.get_transitions(state, action)
        return sum(prob * (reward + self.gamma * self.__V[next_state]) 
                   for prob, next_state, reward in transitions)
    
    def get_v_values(self):
        return self.__V
    