import random
import matplotlib.pyplot as plt

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem

from experiment import Experiment

from iterative_policy_evaluation import IterativePolicyEvaluation
from policies.uniform_random_policy import UniformRandomPolicy
from policies.greedy_policy import GreedyPolicy
from value_iteration import ValueIteration

def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward


def play(problem):
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        actions = problem.get_available_actions(state)
        action = get_action_from_user(actions)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_gambler_problem():
    p = 0.4
    problem = GamblerProblem(p)
    play(problem)


def play_grid_problem():
    size = 4
    problem = GridProblem(size)
    play(problem)

def play_cookie_problem():
    size = 3
    problem = CookieProblem(size)
    play(problem)

def evaluate_policy_on_problem(policy, problem, gamma=1, theta=0.0000000001):
    evaluator = IterativePolicyEvaluation(problem, gamma=gamma, theta=theta)
    evaluator.evaluate(policy)
    return evaluator.V

def estimate_policy_for_problem(problem, gamma=1, theta=0.0000000001):
    estimator = ValueIteration(problem, gamma=gamma, theta=theta)
    policy = estimator.estimate()
    initial_state = problem.get_initial_state()
    #print(policy.v_values[initial_state])

    states = problem.states
    for state in states[1:100]:
        policy.get_action(state)
    actions = policy.optimal_actions
    #print(actions)
    #key_list = list(actions.keys())
    #for idx, state in enumerate(key_list):
        #key_list[idx] = int(state)
    #print(list(actions.values()))
    #plot actions
    

    #x = key_list
    y = list(actions.values())
    #print(y)
    #define plot size
    plt.figure(figsize=(10, 5))
    for idx, actions in enumerate(y):
        for action in actions:
            plt.scatter(idx, action, c='orange', s=0.7)
    #plt.scatter(x, y)
    plt.xlabel('Capital')
    plt.ylabel('Stake')
    plt.title('Scatter plot of Optimal Actions for each State')
    plt.savefig('OptimalActions.png')

def evaluate_greedy_policy_on_problem(problem, gamma=1, theta=0.0000000001):
    policy = UniformRandomPolicy(problem)
    initial_v_values = evaluate_policy_on_problem(policy, problem, gamma, theta)
    greedy_policy = GreedyPolicy(problem, initial_v_values, gamma)
    greedy_v_values = evaluate_policy_on_problem(greedy_policy, problem, gamma, theta)
    return greedy_v_values

if __name__ == '__main__':
    exp = Experiment()
    exp.run()

    # estimate_policy_for_problem(GamblerProblem(0.55), gamma=1.0, theta=0.0000000001)
    # problem = CookieProblem(3)
    # problem = GamblerProblem(0.55)
    #sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    #ps = [0.25, 0.4, 0.55]
    #for size in ps:
    #    problem = GamblerProblem(size)
    #    greedy_v_values = evaluate_greedy_policy_on_problem(problem, gamma=1, theta=0.0000000001)
    #    print("V(initial_state) = ",greedy_v_values[problem.get_initial_state()])
    #    print("Size = ", size)
    #problem = GridProblem(4)
    #policy = UniformRandomPolicy(problem)
    # greedy_v_values = evaluate_greedy_policy_on_problem(problem, gamma=1, theta=0.0000000001)
    # print("V(initial_state) = ",greedy_v_values[problem.get_initial_state()])
    # v_values = evaluate_policy_on_problem(policy, problem, gamma=1.0)
    # v_values = evaluate_greedy_policy_on_problem(problem, gamma=1, theta=0.0000000001)
    # initial_state = problem.get_initial_state()
    # print(v_values[initial_state])
    #estimate_policy_for_problem(problem, gamma=0.99)