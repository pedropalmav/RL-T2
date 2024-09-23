from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv

from monte_carlo import MonteCarlo
from policies.epsilon_greedy import EpsilonGreedy
from episode import Episode

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


def play_blackjack(num_of_episodes, gamma, epsilon, show=False, print_values=False):
    policy = EpsilonGreedy(epsilon)
    env = BlackjackEnv()
    monte_carlo = MonteCarlo(env, policy, gamma)
    q_values = monte_carlo.run(num_of_episodes)
    if show:
        show_last_episode(env, env.action_space, q_values, policy)
    if print_values:
        print_q_values(q_values)


def play_cliff(num_of_episodes, gamma, epsilon, cliff_width, show=False, print_values=False):
    policy = EpsilonGreedy(epsilon)
    env = CliffEnv(cliff_width)
    monte_carlo = MonteCarlo(env, policy, gamma)
    q_values = monte_carlo.run(num_of_episodes)
    if show:
        show_last_episode(env, env.action_space, q_values, policy)
    if print_values:
        print_q_values(q_values)

def show_last_episode(env, actions, q_values, policy):
    actions = env.action_space
    last_episode = Episode(env, policy, show=True)
    trace = last_episode.run(actions, q_values)

def print_q_values(q_values):
    print("Q-values:")
    for key, value in q_values.items():
        print(f"{key}: {value}")

if __name__ == '__main__':

    num_of_episodes = 500000
    gamma = 0.99
    epsilon = 0.1
    cliff_width = 6
    play_cliff(num_of_episodes, gamma, epsilon, cliff_width, show=True)
    # play_blackjack(num_of_episodes, gamma, epsilon, show=True)
