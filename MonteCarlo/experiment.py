import matplotlib.pyplot as plt

from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
from Environments.env_option import EnvOption
from monte_carlo import MonteCarlo
from policies.epsilon_greedy import EpsilonGreedy
from episode import Episode

class Experiment:
    def __init__(self, gamma=1.0, report=False):
        self.gamma = gamma
        self.report = report

        self.__initialize_problem()
        self.task = self.__select_task()
        self.__initialize_monte_carlo()


    def __initialize_problem(self):
        selected_env = self.__select_problem()
        if selected_env == EnvOption.BLACKJACK:
            self.env = BlackjackEnv()
            self.epsilon = 0.01
            self.num_of_episodes = 10000000
            self.step_for_report = 500000
        elif selected_env == EnvOption.CLIFF:
            cliff_width = self.__select_cliff_width()
            self.env = CliffEnv(cliff_width)
            self.epsilon = 0.1
            self.num_of_episodes = 200000
            self.step_for_report = 1000
        else:
            raise ValueError("Invalid problem selection")
    
    def __select_problem(self):
        print("Select the problem:")
        for problem in EnvOption:
            print(f"{problem.value}. {problem.name}")
        selection = int(input())
        return EnvOption(selection)
        
    def __select_cliff_width(self):
        print("Enter the width of the cliff:")
        return int(input())
    
    def __initialize_monte_carlo(self):
        policy = EpsilonGreedy(self.epsilon)
        if self.task == 2:
            self.report = True
        self.monte_carlo = MonteCarlo(self.env, policy, self.gamma, report=self.report)

    def run(self):
        if self.task == 1:
            self.__obtain_final_policy()
        elif self.task == 2:
            self.__simulate_policy()
        else:
            raise ValueError("Invalid task selection")

    def __select_task(self):
        print("Select the task:")
        print("1. Show the final policy")
        print("2. Generate plot for 5 runs")
        selection = int(input())
        return selection
    
    def __obtain_final_policy(self):
        q_values, average_returns = self.run_problem()
        self.__show_final_episode(q_values)
    
    def __show_final_episode(self, q_values):
        last_episode = Episode(self.env, self.monte_carlo.policy, show=True)
        print("\nFinal policy:\n")
        last_episode.generate_trace(q_values)
    
    def __simulate_policy(self):
        runs_average_returns = []
        for i in range(5):
            q_values, average_returns = self.run_problem()
            runs_average_returns.append(average_returns)
        self.__generate_plot(runs_average_returns)
        
    def run_problem(self):
        q_values, average_returns = self.monte_carlo.run(self.num_of_episodes)
        return q_values, average_returns
    
    def __generate_plot(self, runs_average_returns):
        eps = [i for i in range(0, self.num_of_episodes + self.step_for_report, self.step_for_report)]
        eps[0] += 1
        x_label = 'Episodes'
        if not isinstance(self.env, CliffEnv):
            eps = [i/1000 for i in eps]
            x_label = 'Episodes (at scale 1:1000)'
        plt.figure()
        for i in  range(len(runs_average_returns)):
            plt.plot(eps, runs_average_returns[i], label=f'Run {i+1}')
        plt.xlabel(x_label)
        plt.ylabel('Average return')
        plt.legend()
        filename = input("Enter the filename to save the plot: ")
        plt.savefig(f'{filename}.png')
