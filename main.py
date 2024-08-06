import random
import numpy as np
#from deap import base, creator, tools, algorithms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas

population_size = 50   # Population size 
generations = 100   # number of generations to run the Genetic Algorithm

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
        elif max_capacity == 1000: optimal_value = 9767
        else: optimal_value = 1514
        
        item_dict = {"weights":weights ,"values":values}

    return item_dict, max_capacity, item_count, optimal_value

def initial_pop(population_size, num_items):
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
    random.seed(64)
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


def selection(population, knapsack_items, max_capacity, tournament_size=3):
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

def main():
    # load data 
    dataset_file = '10_269'
    knapsack_items, max_capacity, num_items, optimal_value = create_dataset(dataset_file)  # Obtain dataset values into parameter

    # Initialize populations
    populations = initial_pop(population_size, num_items)

    # Apply Selection process
    for generation in range(generations):
        population = selection(populations, knapsack_items, max_capacity)

if __name__ == "__main__":
    main()