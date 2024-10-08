from episode import Episode
from Environments.CliffEnv import CliffEnv
from policies.epsilon_greedy import EpsilonGreedy

class MonteCarlo:

    def __init__(self, env, policy, gamma=1.0, report=False):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.episode = Episode(env, policy)

        self.report = report
        if report:
            test_policy = EpsilonGreedy(0)
            self.test_episode = Episode(env, test_policy)
            self.episodes_for_test = 1000 if isinstance(env, CliffEnv) else 500000
            self.num_of_simulations = 1 if isinstance(self.env, CliffEnv) else 100000
            

    def run(self, num_episodes):
        q_values = {}
        n_returns = {}
        average_returns= []
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
                average_returns.append(average_return)
                print(f"Episode {episode + 1}/{num_episodes}, average return: {average_return}")

        return q_values, average_returns
    
    def __should_report(self, episode):
        return self.report and ((episode + 1) % self.episodes_for_test == 0 or episode == 0)
    
    def test_policy(self, q_values):
        total_reward = 0.0
        for i in range(self.num_of_simulations):
            total_reward += self.test_episode.evaluate_policy(q_values)
        return total_reward / self.num_of_simulations