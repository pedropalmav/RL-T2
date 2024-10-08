import time
class IterativePolicyEvaluation:
    def __init__(self, env, gamma=1.0, theta=0.0000000001):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {state: 0.0 for state in self.env.states}
        self.delta = 0.0
        
    def evaluate(self, policy):
        #start = time.time_ns()
        while True:
            self.delta = 0
            for state in self.env.states:
                self.__update_v_value(state, policy)
            #print(f"Delta: {self.delta}")
            if self.delta < self.theta:
                break
        #end = time.time_ns()
        #print(f"Time taken: {(end-start)/1e6} ms")
    
    def __update_v_value(self, state, policy):
        v = self.V[state] 
        new_v = 0
        available_actions = self.env.get_available_actions(state)
        # TODO: Refactor this to use nested sums or create a new function
        for action in available_actions:
            if self.env.is_terminal(state):
                continue
            policy_prob = policy.get_prob(state, action)
            inner_sum = 0
            for prob, next_state, reward in self.env.get_transitions(state, action):
                inner_sum += prob * (reward + self.gamma * self.V[next_state])
            new_v += policy_prob[str(action)] * inner_sum
        self.V[state] = new_v
        self.delta = max(self.delta, abs(v - self.V[state]))

    def get_state_value(self):
        return self.V