from iterative_policy_evaluation import IterativePolicyEvaluation
from policies.uniform_random_policy import UniformRandomPolicy
from policies.greedy_policy import GreedyPolicy
from experiment import Experiment

def evaluate_policy_on_problem(policy, problem, gamma=1, theta=0.0000000001):
    evaluator = IterativePolicyEvaluation(problem, gamma=gamma, theta=theta)
    evaluator.evaluate(policy)
    return evaluator.V

def evaluate_greedy_policy_on_problem(problem, gamma=1, theta=0.0000000001):
    policy = UniformRandomPolicy(problem)
    initial_v_values = evaluate_policy_on_problem(policy, problem, gamma, theta)
    greedy_policy = GreedyPolicy(problem, initial_v_values, gamma)
    greedy_v_values = evaluate_policy_on_problem(greedy_policy, problem, gamma, theta)
    return greedy_v_values

if __name__ == '__main__':
    # exp = Experiment()
    # exp.run()

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