import matplotlib.pyplot as plt

from task_option import TaskOption
from Problems.problem_option import ProblemOption
from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem
from iterative_policy_evaluation import IterativePolicyEvaluation
from value_iteration import ValueIteration
from policies.greedy_policy import GreedyPolicy
from policies.uniform_random_policy import UniformRandomPolicy

class Experiment:
    def __init__(self):
        self.__initialize_problem()
        self.task = self.__select_task()


    def __initialize_problem(self):
        selection = self.__select_problem()
        problem_param = self.__get_problem_parameter(selection)
        match selection:
            case ProblemOption.GRID_PROBLEM:
                self.problem = GridProblem(problem_param)
            case ProblemOption.COOKIE_PROBLEM:
                self.problem = CookieProblem(problem_param)
            case ProblemOption.GAMBLERS_PROBLEM:
                self.problem = GamblerProblem(problem_param)
        self.gamma = 0.99 if selection == ProblemOption.COOKIE_PROBLEM else 1.0
    
    def __select_problem(self):
        print("Select the problem:")
        for problem in ProblemOption:
            problem_name = problem.name.replace("_", " ").capitalize()
            print(f"{problem.value}. {problem_name}")
        selection = int(input())
        return ProblemOption(selection)
    
    def __get_problem_parameter(self, selected_problem):
        parameter = "p" if selected_problem == ProblemOption.GAMBLERS_PROBLEM else "size"
        problem_name = selected_problem.name.replace("_", " ").capitalize()
        param = input(f"Enter {parameter} for {problem_name}: ")
        return float(param) if selected_problem == ProblemOption.GAMBLERS_PROBLEM else int(param)
    
    def __select_task(self):
        print("Select the task:")
        for task in TaskOption:
            if task == TaskOption.PLOT_OPTIMAL_POLICY and not isinstance(self.problem, GamblerProblem):
                continue
            task_name = task.name.replace("_", " ").capitalize()
            print(f"{task.value}. {task_name}")
        selection = int(input())
        return TaskOption(selection)

    def run(self):
        match self.task:
            case TaskOption.EVALUATE_UNIFORM_POLICY:
                self.__evaluate_uniform_policy()
            case TaskOption.EVALUATE_GREEDY_POLICY:
                self.__evaluate_greedy_policy()
            case TaskOption.VALUE_ITERATION:
                self.__value_iteration()
            case TaskOption.PLOT_OPTIMAL_POLICY:
                self.__plot_optimal_policy()

    def __evaluate_uniform_policy(self):
        policy = UniformRandomPolicy(self.problem)
        v_values = self.__iterative_policy_evaluation(policy)
        initial_state = self.problem.get_initial_state()
        print(f"V_0: {v_values[initial_state]}")

    def __iterative_policy_evaluation(self, policy):
        policy_evaluation = IterativePolicyEvaluation(self.problem, self.gamma)
        policy_evaluation.evaluate(policy)
        return policy_evaluation.V
    
    def __evaluate_greedy_policy(self):
        uniform_policy = UniformRandomPolicy(self.problem)
        initial_v_values = self.__iterative_policy_evaluation(uniform_policy)
        greedy_policy = GreedyPolicy(self.problem, initial_v_values, self.gamma)
        v_values = self.__iterative_policy_evaluation(greedy_policy)
        initial_state = self.problem.get_initial_state()
        print(f"V_0: {v_values[initial_state]}")

    def __value_iteration(self):
        estimator = ValueIteration(self.problem, gamma=self.gamma)
        policy = estimator.estimate()
        initial_state = self.problem.get_initial_state()
        print(f"V_0: {policy.v_values[initial_state]}")
        return policy

    def __plot_optimal_policy(self):
        policy = self.__value_iteration()
        states = self.problem.states
        for state in states[1:100]:
            policy.get_action(state)
        actions = policy.optimal_actions
        self.__generate_plot(actions)
    
    def __generate_plot(self, actions):
        y = list(actions.values())
        plt.figure(figsize=(10, 5))
        for idx, actions in enumerate(y):
            for action in actions:
                plt.scatter(idx, action, c='orange', s=0.7)
        plt.xlabel('Capital')
        plt.ylabel('Stake')
        plt.title('Scatter plot of Optimal Actions for each State')
        plt.savefig('OptimalActions.png')