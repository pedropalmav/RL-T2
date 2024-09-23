import random

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem

from iterative_policy_evaluation import IterativePolicyEvaluation
from policies.uniform_random_policy import UniformRandomPolicy
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
    initial_state = problem.get_initial_state()
    print("V(initial_state) = ", evaluator.V[initial_state])

def estimate_policy_for_problem(problem, gamma=1, theta=0.0000000001):
    estimator = ValueIteration(problem, gamma=gamma, theta=theta)
    policy = estimator.estimate()
    print(policy.get_action(problem.get_initial_state()))

if __name__ == '__main__':
    # play_grid_problem()
    # play_cookie_problem()
    # play_gambler_problem()
    sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    #probs = [0.25, 0.4, 0.55]
    #for p in probs:
    for size in sizes:
        problem = GridProblem(size)
        #problem = CookieProblem(size)
        #problem = GamblerProblem(p)
        policy = UniformRandomPolicy(problem)
        evaluate_policy_on_problem(policy, problem, gamma=1.0)
        print("Size: ", size)
        #print("p = ", p)
   
    #estimate_policy_for_problem(problem, gamma=1.0)