import random
import numpy as np
#from deap import base, creator, tools, algorithms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import math


def create_dataset(file_name):
    """
    Reads the dataset file and extracts the necessary information for the knapsack problem.

    Args:
        file_name (str): The name or path of the file containing the dataset.

    Returns:
        tuple: A tuple containing the following elements:
            - items_data (list): A list of dictionaries, where each dictionary represents an item with its weight and value.
            - max_capacity (int): The maximum capacity of the knapsack.
            - item_count (int): The total number of items in the dataset.
            - optimal_value (int): The optimal or best possible value that can be achieved for the given dataset.
    """
    weights = []
    values = []
    max_capacity = 0   # value of maximume weights
    item_count = 0     # number of the items for each individual
    optimal_value = 0  

    with open(file_name,'r') as file: 
        data = file.readlines()      

        for idx, line in enumerate(data): # extract weights and vlues and store it into list
            x = line.split()
            if idx == 0:
                max_capacity = int(x[1])
                item_count = int(x[0])
            else:
                weights.append(int(x[1]))
                values.append(int(x[0]))
        
        # Find the vlaue of optimal_value paramener. depend on value of (max_capacity) 
        if max_capacity == 269: optimal_value = 295
        elif max_capacity == 10000: optimal_value = 9767
        else: optimal_value = 1514
        
        item_dict = {"weights":weights ,"values":values}
    
    return item_dict, max_capacity, item_count, optimal_value

def initial_pop(population_size, num_items, seed_val):
    """
    Generate the initial population for the genetic algorithm.

    Args:
        population_size (int): The desired size of the initial population.
        num_items (int): The number of items in the knapsack problem.

    Returns:
        list: A list of individuals, where each individual is a binary vector
              representing a potential solution to the knapsack problem.
              The length of each individual is equal to `num_items`.
    """
    random.seed(seed_val)
    return [np.random.randint(2, size=num_items) for _ in range(population_size)]

def calculate_fitness(individual, items, max_capicity):
    """
        This function calculates the fitness of an individual. 
        The fitness is the total value of the items included in the knapsack.
    """
    total_weight = sum([items['weights'][i] * individual[i] for i in range(len(items['weights']))])
    total_value = sum([items['values'][i] * individual[i] for i in range(len(items['values']))])
    if total_weight > max_capicity:
        return 0
    else:
        return total_value

def selection(population, knapsack_items, max_capacity, tournament_size=5):
    """
    Performs selection using the tournament selection strategy.

    Args:
        population (list): A list of individuals representing the current population.
        tournament_size (int): The number of individuals to participate in each tournament.

    Returns:
        list: A list of selected individuals for reproduction.
    """
    selected_individuals = []

    for _ in range(len(population)):
        # Select tournament_size individuals randomly from the population
        tournament = random.sample(population, tournament_size)

        # Find the fittest individual in the tournament
        tournament_fitnesses = [calculate_fitness(individual, knapsack_items, max_capacity) for individual in tournament]
        fittest_idx = tournament_fitnesses.index(max(tournament_fitnesses))
        fittest_individual = tournament[fittest_idx]

        # Add the fittest individual to the list of selected individuals
        selected_individuals.append(fittest_individual)

    return selected_individuals

def crossover(parent1, parent2, items):
    """
        The function chooses a random crossover point, 
        then creates two new individuals by concatenating the genetic material of the two parents at that point
    """
    crossover_point = random.randint(1, items - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(individual, items, mutation_rate):
    """
    The function iterates over each element of the individual's genetic vector and, 
    with probability mutation_rate, flips the element (i.e., changes 0 to 1 or 1 to 0)

    PARAM: 
        - individual (list): A list representing the genetic vector or chromosome of an individual in the population.
        - items: A list of tuples, where each tuple represents an item with its weight and value (weight, value).
    RETURN:
        - mutated_individual (list): A new list representing the mutated individual's genetic vector after applying the mutation operation.
    """
    mutated_individual = individual.copy()
    for i in range(items):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
    return mutated_individual

def calc_value_wights(individual, items):
    """
    Calculate the total value and total weight of the items included in an individual's solution.

    Args:
        individual (list): A binary list representing an individual's solution, where 1 indicates that the item is included, and 0 indicates that it is not included.
        items (dict): A dictionary containing the weights and values of the items in the knapsack problem. It should have two keys: 'weights' and 'values', each mapping to a list of corresponding weights and values for the items.

    Returns:
        tuple: A tuple containing two elements:
            - total_values (int): The total value of the items included in the individual's solution.
            - total_weights (int): The total weight of the items included in the individual's solution.
    """
    total_weights = sum([items['weights'][i] * individual[i] for i in range(len(items['weights']))])
    total_values = sum([items['values'][i] * individual[i] for i in range(len(items['values']))])
    return total_values, total_weights

def find_best_individual(population, items_val_w, max_capacity):
    """
        Find the best individual from a given population based on the fitness values and the maximum capacity constraint.

        Args:
            population (list): A list of individuals.
            items_val_w (dict): A dictionary containing the weights and values of the items in the knapsack.
            max_capacity (int): The maximum capacity of the knapsack.

        Returns:
            dict: A dictionary representing the best individual found in the population. The dictionary has the following keys:
                - 'value' (int): The total value of the items included in the best individual's solution.
                - 'weight' (int): The total weight of the items included in the best individual's solution.
                - 'individual' (list): The binary array representing the best individual's solution.
    """
    best_individual = dict()
    best_individual['value'] = 0
    best_individual['weight'] = 0
    best_individual['individual'] = [0] * int(len(items_val_w['weights']))

    for individual in population:
        # calc total values and weights for current individual
        individual_value, individual_wight = calc_value_wights(individual, items_val_w)

        if individual_value > best_individual['value'] and individual_wight <= max_capacity:
            best_individual['value'] = individual_value
            best_individual['weight'] = individual_wight
            best_individual['individual'] = individual

    return best_individual

def calculate_stats(list_value):
    """
    Calculate the mean and standard deviation of a lists.

    Args:
        list_value (list): The first list of numbers.

    Returns:
        tuple: A tuple containing two elements:
            - mean1 (float): The mean of list.
            - std_dev1 (float): The standard deviation of list.
    """
    # Calculate the mean of list
    mean1 = sum(list_value) / len(list_value)

    # Calculate the standard deviation of list
    squared_diffs = [(x - mean1) ** 2 for x in list_value]
    variance1 = sum(squared_diffs) / len(list_value)
    std_dev1 = math.sqrt(variance1)

    return mean1, std_dev1

def create_table(total_wights_li, total_values_li, mean_value, std_value,
                    mean_wight, std_wight):
    """
    Create a dataset with two columns from two input lists.

    Args:
        list1 (list): The first list of values.
        list2 (list): The second list of values.
        column1_name (str): The name of the first column.
        column2_name (str): The name of the second column.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the two columns.
    """
    # Check if the lists have the same length
    if len(total_wights_li) != len(total_values_li):
        raise ValueError("The input lists must have the same length.")

    # First column
    first_column = ['Run 1','Run 2','Run 3','Run 4','Run 5']

    # Create a dictionary with the two lists as values
    data = {'': first_column, 'Total value': total_values_li, 'Total Wight': total_wights_li}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    mean_row = pd.DataFrame({'': ['Mean'], 'Total value': [mean_value], 'Total Wight': [mean_wight]})
    data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'Total value': [std_value], 'Total Wight': [std_wight]})
    data_table = pd.concat([data_table, std_row], ignore_index=True)


    return data_table

def main():
    # parameter setting 
    population_size = 50   # Population size 
    generations = 50   # number of generations to run the Genetic Algorithm
    mutation_rate = 0.2
    run_no = 5  # number of runs GA
    # load data 
    dataset_file = '10_269'  # 23_10000  10_269  100_995
    knapsack_items, max_capacity, num_items, optimal_value = create_dataset(dataset_file)  # Obtain dataset values into parameter
    
    # run GA for 5 times
    best_weights = []  # Store summation weight of best individual each run
    best_values = []  # Store summation value of best individual each run 
    best_individuals = []

    seed_value = [20, 40, 60, 80, 100]

    for run in range(run_no):
        # Generate a different seed for each run

        # Initialize populations
        populations = initial_pop(population_size, num_items, seed_value[run])

        # Apply Selection process
        for generation in range(generations):
            population = selection(populations, knapsack_items, max_capacity)

            new_population = []
            # Apply genetic operation 
            for i in range(population_size // 2):
                parent1, parent2 = random.sample(population, 2)
                child1, child2 = crossover(parent1, parent2, num_items)
                new_population.append(mutation(child1, num_items, mutation_rate))
                new_population.append(mutation(child2, num_items, mutation_rate))

            # Evaluate fitness of the new population
            new_population = [(individual, calculate_fitness(individual, knapsack_items, max_capacity)) for individual in new_population]

            # Sort the new population based on fitness
            new_population = sorted(new_population, key=lambda x: x[1], reverse=True)

            # Update population with a new ossspring, Select the fittest individuals for the next generation
            population = [individual for individual, _ in new_population[:population_size]]

        # Find the best population from produced population
        temp_best_solution = find_best_individual(population, knapsack_items, max_capacity)

        # Store best (indivisual, values, wights) for current run
        best_individuals.append(temp_best_solution['individual'])
        best_values.append(temp_best_solution['value'])
        best_weights.append(temp_best_solution['weight'])

    # Store wights and values of best 5 individual
    runs_best_individual = {'best_individuals':best_individuals, 'best_weights':best_weights, 'best_values':best_values}

    # Create a table that store 'mean' and 'stander deviation' for 5 best solution (runs_best_individual)
    best_wights_mean, best_wights_std = calculate_stats(best_weights)
    best_values_mean, best_values_std = calculate_stats(best_values)

    # Create a table hold best 5 solutions with corresponded values, which including: values, wights, mean and STD
    data_table = create_table(runs_best_individual['best_weights'], runs_best_individual['best_values'], best_values_mean, 
                              best_values_std, best_wights_mean, best_wights_std)

    print('Optimal Value is: ', optimal_value)
    print(data_table)

if __name__ == "__main__":
    main()