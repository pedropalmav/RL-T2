from episode import Episode
from Environments.CliffEnv import CliffEnv

class MonteCarlo:

    def __init__(self, env, policy, gamma=1.0, report=False):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.episode = Episode(env, policy)

        self.episodes_for_test = 1000 if isinstance(env, CliffEnv) else 500000
        self.report = report

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

            if self.__should_report(episode):
                average_return = self.test_policy(q_values)
                print(f"Episode {episode + 1}/{num_episodes}, average return: {average_return}")

        return q_values
    
    def __should_report(self, episode):
        return self.report and (episode + 1) % self.episodes_for_test == 0
    
    def test_policy(self, q_values):
        num_of_simulations = 100000
        total_reward = 0.0
        for i in range(num_of_simulations):
            total_reward += self.episode.evaluate_policy(q_values)
        return total_reward / num_of_simulations