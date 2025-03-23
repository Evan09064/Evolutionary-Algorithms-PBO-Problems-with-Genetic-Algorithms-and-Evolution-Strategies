import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import sys

budget = 5000
dimension = 50

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`


def uniform_crossover_f19(p1, p2, crossover_probability):
    if (np.random.uniform(0, 1) < crossover_probability):
        for i in range(len(p1)):
            if np.random.uniform(0, 1) < 0.5:
                t = p1[i]
                p1[i] = p2[i]
                p2[i] = t

def inversion_mutation_f19(p, mutation_rate):
    '''
    Apply inversion mutation to the binary sequence p with the given mutation rate.
    '''
    if np.random.uniform(0, 1) < mutation_rate:
        # Randomly select the start and end indices for inversion
        start_index = np.random.randint(0, len(p))
        end_index = np.random.randint(start_index, len(p))

        # Perform inversion by reversing the selected subset
        p[start_index:end_index + 1] = p[start_index:end_index + 1][::-1]

def swap (p1, p2, crossover_point):
    temp = p1[crossover_point:]
    p1[crossover_point:] = p2[crossover_point:]
    p2[crossover_point:] = temp
def n_point_crossover_f18(p1, p2, crossover_probability, n):
    # Check if crossover should occur
    if (np.random.uniform(0, 1) < crossover_probability):
        # Choose n-random crossover points and sort the array
        crossover_points = sorted(np.random.choice(len(p1), n, replace=False))
        # I used choice function and replace = False in order to generate different
        # values for the crossover_points

        # Swap parents' genes at each crossover point
        for point in crossover_points:
            # For each crossover point perform swap between two parents
            swap(p1, p2, point)

def random_resetting_mutation_f18(p):
    """Randomly reset the value of a gene to a new random value."""
    mutation_point = np.random.randint(0, len(p))
    p[mutation_point] = np.random.randint(2)

def tournament_selection(parent, parent_f, tournament_k):
    # Using the tournament selection
    select_parent = []
    for i in range(len(parent)):
        pre_select = np.random.choice(len(parent_f), tournament_k, replace=True)
        max_f = sys.float_info.min
        for p in pre_select:
            if parent_f[p] > max_f:
                index = p
                max_f = parent_f[p]
        select_parent.append(parent[index].copy())
    return select_parent


def s3817911_s4026543_GA(problem):
    # initial_pop = ... make sure you randomly create the first population
    f_opt = sys.float_info.min
    x_opt = None

    # Parameters setting
    pop_size = 7
    crossover_probability = 0.9
    mutation_rate = 1 / 2
    tournament_k = 20
    n = 25 # For n-point crossover (for F18)


    # Initialize the population
    parent = []
    parent_f = []
    for i in range(pop_size):
        parent.append(np.random.randint(2, size=problem.meta_data.n_variables))
        parent_f.append(problem(parent[i]))

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)

        # Selection
        offspring = tournament_selection(parent, parent_f, tournament_k)

        # Crossover
        for i in range(0, pop_size - (pop_size % 2), 2):
            if flag == 19:
                uniform_crossover_f19(offspring[i], offspring[i + 1], crossover_probability)
            elif flag == 18:
                n_point_crossover_f18(offspring[i], offspring[i + 1], crossover_probability, n)

        # Mutation
        for i in range(pop_size):
            if flag == 19:
                inversion_mutation_f19(offspring[i], mutation_rate)
            elif flag == 18:
                random_resetting_mutation_f18(offspring[i])

        # Evaluation
        parent = offspring.copy()
        for i in range(pop_size):
            parent_f[i] = problem(parent[i])
            if parent_f[i] >= f_opt:
                f_opt = parent_f[i]
                x_opt = parent[i].copy()
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
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l

if __name__ == "__main__":
    #this how you run your algorithm with 20 repetitions/independent run
    np.random.seed(94)
    F18, _logger = create_problem(18)
    flag = 18
    arr = []
    for run in range(20):
        s3817911_s4026543_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    avg_fitness = np.mean(arr)
    print("F18 average best fitness after 20 independent runs: ", avg_fitness)

    np.random.seed(60)
    arr = []
    F19, _logger = create_problem(19)
    flag = 19
    for run in range(20):
        s3817911_s4026543_GA(F19)
        F19.reset()
    _logger.close()
    avg_fitness = np.mean(arr)
    print("F19 average best fitness after 20 independent runs: ", avg_fitness)
