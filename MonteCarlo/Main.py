from experiment import Experiment

from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv

from monte_carlo import MonteCarlo
from policies.epsilon_greedy import EpsilonGreedy
from episode import Episode
import matplotlib.pyplot as plt
def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def play(env):
    actions = env.action_space
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.show()
        action = get_action_from_user(actions)
        state, reward, done = env.step(action)
        total_reward += reward
    env.show()
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_blackjack(num_of_episodes, gamma, epsilon, show=False, print_values=False, i_run=1):
    policy = EpsilonGreedy(epsilon)
    env = BlackjackEnv()
    monte_carlo = MonteCarlo(env, policy, gamma, report=True)
    q_values, average_returns = monte_carlo.run(num_of_episodes)
    if show:
        show_last_episode(env, q_values, policy)
    if print_values:
        print_q_values(q_values)
    eps = [0.001, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    #print('len: ', len(average_returns))
    plt.plot(eps, average_returns, label=f'Run {i_run}')
    
    


def play_cliff(num_of_episodes, gamma, epsilon, cliff_width, show=False, print_values=False, i_run=1):
    policy = EpsilonGreedy(epsilon)
    env = CliffEnv(cliff_width)
    monte_carlo = MonteCarlo(env, policy, gamma, report=True)
    q_values, average_returns = monte_carlo.run(num_of_episodes)
    if show:
        show_last_episode(env, q_values, policy)
    if print_values:
        print_q_values(q_values)

    eps = [i for i in range(1, 201000, 1000)]
    #print('len: ', len(average_returns))
    plt.plot(eps, average_returns, label=f'Run {i_run}')
    

def show_last_episode(env, q_values, policy):
    last_episode = Episode(env, policy, show=True)
    trace = last_episode.generate_trace(q_values)

def print_q_values(q_values):
    print("Q-values:")
    for key, value in q_values.items():
        print(f"{key}: {value}")

if __name__ == '__main__':

    exp = Experiment(report=True)
    exp.run_problem()
    # num_of_episodes = 200000
    # num_of_episodes = 10000000
    # gamma = 1.0
    # epsilon = 0.1
    # epsilon = 0.01
    # cliff_width = 6
    # play_cliff(num_of_episodes, gamma, epsilon, cliff_width, show=True)
    # play_cliff(num_of_episodes, gamma, epsilon, cliff_width, show=True)
    # play_blackjack(num_of_episodes, gamma, epsilon, show=False, print_values=True) 
    #plt.figure(figsize=(5, 5)) 
    #for i in range(5):
     #   play_cliff(num_of_episodes, gamma, epsilon, cliff_width, show=True, i_run = i+1)
   #     play_blackjack(num_of_episodes, gamma, epsilon, show=True, print_values=False, i_run=i+1) 
    #plt.xlabel('Episodes (at scale 1:1000)')
    #plt.ylabel('Average return')
    #plt.title('Monte Carlo - Cliff')
    #plt.legend()
    #plt.show()
