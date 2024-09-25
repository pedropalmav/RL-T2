class Episode:

    def __init__(self, env, policy, show=False):
        self.env = env
        self.policy = policy
        self.show = show

    def generate_trace(self, q_values):
        trace = []
        state = self.env.reset()
        done = False
        while not done:
            if self.show:
                self.env.show()
            action = self.policy.get_action(state=state, q_values=q_values, actions=self.env.action_space)
            next_state, reward, done = self.env.step(action)
            trace.append((state, action, reward))
            state = next_state
        return trace
    
    def evaluate_policy(self, q_values):
        state = self.env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if self.show:
                self.env.show()
            action = self.policy.get_action(state=state, q_values=q_values, actions=self.env.action_space)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            state = next_state
        return total_reward