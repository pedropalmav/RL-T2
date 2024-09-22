from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv

from monte_carlo import MonteCarlo
from epsilon_greedy import EpsilonGreedy

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


def play_blackjack():
    env = BlackjackEnv()
    play(env)


def play_cliff():
    cliff_width = 6
    env = CliffEnv(cliff_width)
    play(env)


if __name__ == '__main__':
    # play_blackjack()

    num_of_episodes = 50000
    gamma = 0.99
    epsilon = 0.1
    cliff_width = 6
    policy = EpsilonGreedy(epsilon)
    env = CliffEnv(cliff_width)
    monte_carlo = MonteCarlo(env, policy, gamma)
    q_values = monte_carlo.run(num_of_episodes)

    print("Q-values:")
    for key, value in q_values.items():
        print(f"{key}: {value}")
