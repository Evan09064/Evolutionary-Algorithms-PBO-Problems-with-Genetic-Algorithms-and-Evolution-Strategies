import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import sys
import random

budget = 5000
dimension = 50

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def encode(num):
    if num < 0.0:
        binary = 0
    else:
        binary = 1
    return binary

def initialization_f19(mu, dimension, lowerbound, upperbound):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(np.random.uniform(low=lowerbound, high=upperbound, size=dimension))
        parent_sigma.append(0.05 * (upperbound - lowerbound))
    return parent, parent_sigma


def individual_sigma_mutation_f19(parent, parent_sigma, tau, tau_prime):
    for i in range(len(parent)):
        g = np.random.normal(0, 1)
        parent_sigma[i] = parent_sigma[i] * np.exp(tau_prime*g + tau*np.random.normal(0, 1))
        for j in range(len(parent[i])):
            parent[i][j] = parent[i][j] + np.random.normal(0, parent_sigma[i])
            parent[i][j] = parent[i][j] if parent[i][j] < 3.5 else 3.5
            parent[i][j] = parent[i][j] if parent[i][j] > -3.5 else -3.5


def discrete_recombination(parent, parent_sigma):
    # Ensure parents have the same length
    assert len(parent) == len(parent_sigma), "Parents and sigma must have the same length"

    num_genes = len(parent[0])
    offspring = np.empty_like(parent[0])

    # Randomly select genes from parents
    for i in range(num_genes):
        # Randomly choose which parent contributes the gene
        parent_index = np.random.choice(len(parent), 1)[0]
        offspring[i] = parent[parent_index][i]

    # Use the average sigma of the selected parents
    sigma = np.mean(parent_sigma)

    return offspring, sigma

def tournament_selection(offspring, offspring_f, offspring_sigma, mu, tournament_size):
    selected_parents = []

    for _ in range(mu):
        tournament_indices = random.sample(range(len(offspring)), tournament_size)
        tournament_candidates = [(offspring[i], offspring_f[i], offspring_sigma[i]) for i in tournament_indices]

        # Select the best individual from the tournament
        winner = max(tournament_candidates, key=lambda x: x[1])
        selected_parents.append(winner)

    return selected_parents

def initialization_f18(mu, dimension, lowerbound, upperbound):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(np.random.uniform(low=lowerbound, high=upperbound, size=dimension))
        parent_sigma.append(0.05 * (upperbound - lowerbound))
    return parent, parent_sigma

def one_sigma_mutation_f18(parent, parent_sigma, tau):
    for i in range(len(parent)):
        parent_sigma[i] = parent_sigma[i] * np.exp(np.random.normal(0, tau))
        for j in range(len(parent[i])):
            parent[i][j] = parent[i][j] + np.random.normal(0, parent_sigma[i])
            parent[i][j] = parent[i][j] if parent[i][j] < 3.0 else 3.0
            parent[i][j] = parent[i][j] if parent[i][j] > -3.0 else -3.0

def s3817911_s4026543_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population

    f_opt = sys.float_info.min
    x_opt = None

    # `problem.state.evaluations` counts the number of function evaluations automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.


    # Parameters setting
    if flag == 19: # F19 Parameters
        mu_ = 22 
        lambda_ = 100 
        tau = 1.0 / np.sqrt(np.sqrt(2*problem.meta_data.n_variables))
        tau_prime = 1.0 / np.sqrt(2*problem.meta_data.n_variables)
        lower_bound = -3.5
        upper_bound = 3.5
        tournament_k = 5
    
    elif flag == 18:
        mu_ = 20
        lambda_ = 20
        tau = 1.0 / np.sqrt(problem.meta_data.n_variables)
        lower_bound = -5.0
        upper_bound = 5.0
        tournament_k = 5

    # Initialization
    if flag == 19:
        parent, parent_sigma = initialization_f19(mu_, dimension, lower_bound, upper_bound)
    elif flag == 18:
        parent, parent_sigma = initialization_f18(mu_, dimension, lower_bound, upper_bound)

    # Evaluation
    parent_f = []
    for i in range(mu_):
        new_parent_i = []
        for real_val in parent[i]:
            binary_val = encode(real_val)
            new_parent_i.append(binary_val)
        parent_f.append(problem(new_parent_i))
        if parent_f[i] > f_opt:
            f_opt = parent_f[i]

    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
        offspring = []
        offspring_sigma = []
        offspring_f = []

        # Recombination
        for i in range(lambda_):
            o, s = discrete_recombination(parent, parent_sigma)
            offspring.append(o)
            offspring_sigma.append(s)

        # Mutation
        if flag == 19:
            individual_sigma_mutation_f19(offspring, offspring_sigma, tau, tau_prime)
        elif flag == 18:
            one_sigma_mutation_f18(offspring, offspring_sigma, tau)

        # Evaluation
        for i in range(lambda_):
            new_offspring_i = []
            for real_val in offspring[i]:
                binary_val = encode(real_val)
                new_offspring_i.append(binary_val)
            offspring_f.append(problem(new_offspring_i))
            if offspring_f[i] > f_opt:
                f_opt = offspring_f[i]

        # Tournament Selection
        selected_parents = tournament_selection(offspring, offspring_f, offspring_sigma, mu_, tournament_k)

        parent = [p[0] for p in selected_parents]
        parent_f = [p[1] for p in selected_parents]
        parent_sigma = [p[2] for p in selected_parents]

    # no return value needed
    print("Optimal fitness value: ", f_opt)
    arr.append(f_opt)


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution_strategy",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    #this how you run your algorithm with 20 repetitions/independent run
    np.random.seed(42)
    F18, _logger = create_problem(18)
    flag = 18
    arr = []
    for run in range(20):
        s3817911_s4026543_ES(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    avg_f = np.mean(arr)
    print("Average fitness: ", avg_f)

    np.random.seed(42)
    arr = []
    F19, _logger = create_problem(19)
    flag = 19
    for run in range(20):
        s3817911_s4026543_ES(F19)
        F19.reset()
    _logger.close()
    avg_f = np.mean(arr)
    print("Average fitness: ", avg_f)



