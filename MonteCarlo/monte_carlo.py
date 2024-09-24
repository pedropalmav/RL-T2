from episode import Episode

class MonteCarlo:

    def __init__(self, env, policy, gamma=1.0):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.episode = Episode(env, policy)

    def run(self, num_episodes):
        q_values = {}
        n_returns = {}

        for episode in range(num_episodes):
            trace = self.episode.generate_trace(q_values)
            g = 0
            
            for state, action, reward in trace[::-1]:
                g = reward + self.gamma * g
                if (state, action) not in n_returns:
                    q_values[(state, action)] = 0
                    n_returns[(state, action)] = 0

                n_returns[(state, action)] += 1
                q_values[(state, action)] += (g - q_values[(state, action)]) / n_returns[(state, action)]

        return q_values
    
    def test_policy(self):
        pass