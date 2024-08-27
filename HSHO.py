import numpy as np


def SHO(objective_func, prob_size, pop_size, lb, ub, max_iter):
    # Initialize parameters
    alpha = 2.0  # Hyperparameter for exploration
    beta = 1.0  # Hyperparameter for exploitation

    # Initialize population
    population = np.random.uniform(lb, ub, (pop_size, prob_size))
    fitness = np.array([objective_func(ind) for ind in population])

    # Main optimization loop
    for iteration in range(max_iter):
        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        # Update alpha and beta
        alpha = alpha - (alpha / max_iter)
        beta = beta - (beta / max_iter)

        # Generate new population
        new_population = np.zeros_like(population)

        for i in range(pop_size):
            # Select random individuals from the population
            p1, p2 = np.random.choice(pop_size, 2, replace=False)
            A = np.random.uniform(-1, 1, prob_size)
            B = np.random.uniform(-1, 1, prob_size)

            # Update individual position
            new_population[i] = population[i] + alpha * (population[p1] - population[p2]) * A + beta * (
                        population[p1] - population[i]) * B

            # Apply bounds
            new_population[i] = np.clip(new_population[i], lb, ub)

        # Evaluate new population
        new_fitness = np.array([objective_func(ind) for ind in new_population])

        # Select individuals for the next generation
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        best_indices = np.argsort(combined_fitness)[:pop_size]

        population = combined_population[best_indices]
        fitness = combined_fitness[best_indices]

    # Return the best solution and fitness
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    return best_fitness, best_solution


# Example usage
def objective_function(x):
    return np.sum(x ** 2)  # Example: Sphere function


# Parameters
problem_size = 10
population_size = 20
lower_bound = -10
upper_bound = 10
max_iterations = 100

# Run SHO
fitness, solution = SHO(objective_function, problem_size, population_size, lower_bound, upper_bound, max_iterations)

print(f'Best fitness: {fitness}')
print(f'Best solution: {solution}')
