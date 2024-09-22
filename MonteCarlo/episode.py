class Episode:

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def run(self, actions, q_values):
        trace = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.policy.get_action(state=state, q_values=q_values, actions=actions)
            next_state, reward, done = self.env.step(action)
            trace.append((state, action, reward))
            state = next_state
        return trace