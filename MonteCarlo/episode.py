class Episode:

    def __init__(self, env, policy, show=False):
        self.env = env
        self.policy = policy
        self.show = show

    def run(self, actions, q_values):
        trace = []
        state = self.env.reset()
        done = False
        while not done:
            if self.show:
                self.env.show()
            action = self.policy.get_action(state=state, q_values=q_values, actions=actions)
            next_state, reward, done = self.env.step(action)
            trace.append((state, action, reward))
            state = next_state
        return trace