# Import necessary libraries
import numpy as np
import time
import random
# matplotlib to show the grid on the screen like a GUI
import matplotlib.pyplot as plt
# Priority queue for A*
from heapq import heappop, heappush


# Class representing the grid environment
class GridEnvironment:

    # Initialize the grid with size, obstacles, start and goal points
    def __init__(self, size, obstacle_density):
        """
        Initialize the grid environment with obstacles, start, and goal points.

        Parameters:
        - size: Size of the grid (size x size)
        - obstacle_density: Percentage of grid cells occupied by obstacles
        """

        self.size = size
        self.obstacle_density = obstacle_density
        self.grid = self.generate_grid()
        self.start = None
        self.goal = None
        self.place_start_and_goal()

    # This method is responsible for generating a grid
    def generate_grid(self):
        """
        Generate the grid with obstacles based on the specified density.
        Ensure start and goal points are not blocked by obstacles.
        """
        # Create an array filled with zeros, representing an empty grid
        grid = np.zeros((self.size, self.size), dtype=int)

        # Calculates the number of obstacles that will be placed on the grid, self.size * self.size = total # of cells
        # on the grid, multiply by obstacle_density which will be a percentage, divide by 100 to get the fraction
        num_obstacles = int(self.size * self.size * self.obstacle_density / 100)

        # Randomizes the positions of obstacles on the grid
        obstacle_positions = random.sample(range(self.size * self.size), num_obstacles)

        # Iterate through the selected obstacle positions
        for pos in obstacle_positions:

            # For each position, calculate x,y coordinates (positions) using divmod(), x -> row, y -> column
            x, y = divmod(pos, self.size)

            grid[x, y] = 1  # 1 represents an obstacle

            # if the x,y positions are in the start state or goal state, don't put any obstacles on it
            if (x, y) == (9, 0) or (x, y) == (0, 6):
                grid[x, y] = 0  # Ensure start and goal points are not blocked by obstacles

        return grid  # Return the grid with the obstacles

    # This method checks if the x, y coordinates are within the boundaries of the grid
    def is_valid_cell(self, cell):
        """
        Check if a cell is within the grid boundaries and not occupied by an obstacle.

        Parameters:
        - cell: Tuple representing the (x, y) coordinates of the cell.

        Returns:
        - True if the cell is valid, False otherwise.
        """
        x, y = cell

        # Check if the x and y coordinates are within range of the grid, and it's not occupied by an obstacle
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] != 1

    # This method will be responsible for selecting start and goal VALID cells on the grid
    def place_start_and_goal(self):

        """
        Place the start and goal points on valid and obstacle-free cells.
        """

        # Ensure start and goal are not blocked by obstacles by calling get_valid_random_cell() method
        self.start = self.get_valid_random_cell()
        # If the obtained cell is not valid, it will get a new cell until it finds a valid starting point.
        while not self.is_valid_cell(self.start):
            self.start = self.get_valid_random_cell()

        # Do the same as above, but check if the goal cell is not the same as the starting cell
        self.goal = self.get_valid_random_cell()
        while self.start == self.goal or not self.is_valid_cell(self.goal):
            self.goal = self.get_valid_random_cell()

    # This method returns a random, obstacle free cell in the grid
    def get_valid_random_cell(self):
        """
        Get a random valid and obstacle-free cell.

        Returns:
        - A tuple representing the (x, y) coordinates of a valid cell.
        """
        # Create a list (valid_cells) containing all the valid cells
        # Iterate over all possible (x, y) coordinates and checks if each cell is valid using method is_valid_cell()
        # If a cell is valid, it will be added to the list (valid_cells)
        valid_cells = [(x, y) for x in range(self.size) for y in range(self.size) if self.is_valid_cell((x, y))]

        # If there are no valid cells in the list (empty list)
        if not valid_cells:

            # Handle the case where no valid cells are available (adjust obstacle density, etc.)
            # Regenerate the grid to create new set of obstacles
            self.grid = self.generate_grid()
            return self.get_valid_random_cell()
        # If there is a valid cell available in the list, it will randomly select one cell from the list and return it's
        # (x, y) coordinates
        return random.choice(valid_cells)


# Class representing the Genetic Algorithm for pathfinding
# It will initialize parameters required for genetic algorithm
class GeneticAlgorithm:

    """
     -Crossover rate initialized to 0.8 (80%) to ensure diverse exploration in the grid, which will allow the algorithm
     To explore different paths thus increase the potential of getting a better solution
     -For the mutation rate, it is initialized to 0.2 (20%) which will balance between exploration and exploitation
     A lower mutation rate might lead to slower exploration of the grid, and a higher mutation rate can introduce too
     Much randomness in the population, meaning the individuals will randomly change to the point where they may stray
     Too far from their optimal state. So a balance in the mutation rate is needed, we find that 20% is a good mutation
     rate.

    """

    def __init__(self, grid_env, start, goal, population_size, generations, crossover_rate=0.8, mutation_rate=0.2):
        """
        Initialize the Genetic Algorithm parameters and attributes.

        Parameters:
        - grid_env: Instance of GridEnvironment representing the grid environment.
        - start: Tuple representing the (x, y) coordinates of the start point.
        - goal: Tuple representing the (x, y) coordinates of the goal point.

        - population_size: Size of the population in each generation.
        (Entire set of individuals where each individual represents a possible solution)

        - generations: Number of generations to run the algorithm.
        (Represents a single cycle of the algorithms evolution process, where the algorithm selects a 2 parents for
        reproduction based on their fitness, performs crossover and mutation to produce offsprings and replaces
        individuals in the current population with the new offsprings)

        - crossover_rate: Probability of crossover between parents.
        (combines genetic material from 2 parents to produce new offsprings to introduce diversity thus increasing the
        chance of getting a better solution)

        - mutation_rate: Probability of mutation in the offspring.
         (After crossover, some individuals may undergo mutation which are random changes in the individuals,
         these random changes will help increasing the chance of getting a better solution)

        """

        self.grid_env = grid_env
        self.start = start
        self.goal = goal
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    # Method to run the genetic algorithm
    def run_genetic_algorithm(self):

        """
        Run the Genetic Algorithm to find the best path from start to goal.

        Returns:
        - Tuple containing the best path found and its fitness value.
        """
        population = self.initialize_population()

        # Loop over generations, where self.generations is the number of times the loop will execute
        for generation in range(self.generations):
            # Evolve the population through selection, crossover, mutation on it, this will produce a new population
            # The updated population is assigned back to the "population"
            population = self.evolve_population(population)

        # Return the best path and its fitness found in the final population, use "min" function to find the path with
        # The minimum fitness value, the lambda function will calculate the fitness of each path using the fitness
        # Function. The path with the min fitness value will be the optimal path.
        best_path = min(population, key=lambda path: self.fitness(path))

        return best_path, self.fitness(best_path)

    # Method for initializing a population with random paths
    def initialize_population(self):

        """
        Initialize the population with random paths.

        Returns:
        - List of paths representing the initial population.
        """
        # This loop will iterate "population_size" times. For each iteration, it will generate a random path from start
        # To goal using the "generate_random_path" method, the list will contain the random generated paths
        return [self.generate_random_path() for _ in range(self.population_size)]

    # Method for generating random paths
    def generate_random_path(self):

        """
        Generate a random path from start to goal.

        Returns:
        - List representing the random path.
        """
        # Initialize path list containing start point, also set the current to the start point.
        path = [self.start]
        current = self.start

        # While loop that will continue till current arrives to the goal
        while current != self.goal:

            # Get neighbouring cells to the current in the grid environment
            next_moves = get_neighbors(current, self.grid_env)
            # If there are no valid moves (next_moves list is empty) the loop will break
            if not next_moves:
                break  # No valid moves

            # Get a random choice for the next move
            next_move = random.choice(next_moves)

            # Append the next_move to the path list
            path.append(next_move)
            # Update the current to the newly selected next_move
            current = next_move

        return path

    # Method to evolve the population
    def evolve_population(self, population):

        """
        Evolve the population through selection, crossover, and mutation.

        Parameters:
        - population: List of paths representing the current population.

        Returns:
        - List of paths representing the new population.
        """

        # Initialize the empty list
        new_population = []

        # Implement elitism (copy over the best individuals (best paths) )
        # Take 10% of population size and round the number by using int()
        elites = int(0.1 * self.population_size)

        # Sorts the population based on the fitness of each path, with lowest fitness function to be first (best path)
        # Then store it in the new_population list
        new_population.extend(sorted(population, key=lambda path: self.fitness(path))[:elites])

        # Loop until the size of the new population reaches the desired population size
        while len(new_population) < self.population_size:

            # Select 2 parents randomly from the old population
            parent1, parent2 = random.sample(population, 2)

            # If a randomly generated number (0 to 1) is less than the crossover rate, then the crossover will be
            # Performed
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1  # No crossover, just copy the parent

            # If a randomly generated number between 0 and 1 is less than the mutation rate, then the mutation will be
            # Performed
            if random.random() < self.mutation_rate:
                child = self.mutate(child)

            # Add to the list after determining the child's path
            new_population.append(child)

        return new_population

    # Method that performs crossover
    def crossover(self, parent1, parent2):

        """
        Perform crossover between two parents.

        Parameters:
        - parent1: List representing the first parent.
        - parent2: List representing the second parent.

        Returns:
        - List representing the child path after crossover.
        """

        # Implement two-point crossover; 2 random points are selected
        # For example, select two random points and swap the segments between them
        # minus 1 to ensure the points are within  bounds
        point1, point2 = sorted(random.sample(range(1, min(len(parent1), len(parent2)) - 1), 2))

        # Extracts sub path from beginning of parent1 to point 1 + extract sub path from point 1 to point2 + extract
        # sub path from point2 to the end of parent1
        # This combines the genetic material between 2 parents to give to the child
        child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        return child

    # Method for mutation
    def mutate(self, path):

        """
        Perform mutation on a path.

        Parameters:
        - path: List representing the path to be mutated.

        Returns:
        - List representing the mutated path.
        """

        # Implement path mutation
        # For example, randomly select a point in the path and change it to a valid neighbor

        # Create a copy of the original path, to compare between the original path and the mutated path
        mutated_path = path.copy()

        # Select random index within the mutated path, excluding the start and goal state since they shouldn't be
        # mutated, minus 2 so the index fall within the bounds of the mutated path
        index = random.randint(1, len(mutated_path) - 2)

        # Get the selected index in the mutated path list and replace it with a random neighbor
        mutated_path[index] = random.choice(get_neighbors(mutated_path[index], self.grid_env))

        return mutated_path

    # Method for calculating fitness function
    def fitness(self, path):

        """
        Calculate the fitness of a path based on path length and distance to the goal.

        Parameters:
        - path: List representing the path.

        Returns:
        - Fitness value of the path.
        """

        # Calculating fitness, sum up the length of the path + heuristic value for the estimated distance from start
        # to goal. path[-1] for the current position to the goal
        return len(path) + heuristic(path[-1], self.grid_env.goal)


# Heuristic function for A* search
def heuristic(a, b):

    """
    Calculate the Manhattan distance heuristic between two points.

    Parameters:
    - a: Tuple representing the (x, y) coordinates of the first point.
    - b: Tuple representing the (x, y) coordinates of the second point.

    Returns:
    - Manhattan distance between the two points.
    """

    # Calculate manhattan distance between a and b
    # a[0] - b[0] calculates abs difference in the x coordinates
    # a[1] - b[1] calculates abs difference in the y coordinates
    # Adding these together will give the total manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# A* search algorithm
def astar_search(grid_env, start, goal):

    """
    Perform A* search to find the path from start to goal.

    Parameters:
    - grid_env: Instance of GridEnvironment representing the grid environment.
    - start: Tuple representing the (x, y) coordinates of the start point.
    - goal: Tuple representing the (x, y) coordinates of the goal point.

    Returns:
    - List representing the path from start to goal.

    """
    # Priority queue that stores (estimated cost from start to current point, current point itself)
    # Here we represent the start point
    open_set = [(0, start)]

    # A dictionary which will store the parent of each explored point
    came_from = {}

    # Dictionary which will keep track of the cost of the best path from start point to each encountered point so far
    g_score = {start: 0}
    # Dictionary stores the estimated cost of the cheapest path from start to goal point
    f_score = {start: heuristic(start, goal)}

    # The loop will continue until there are no more point to explore
    while open_set:
        # Retrieve point with the lowest score from the open_set
        current = heappop(open_set)[1]
        # If this point (current) is the goal then reconstruct the path from the parent (came_from) dic. to the goal
        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path

        # Explore neighbors of the current point, for each neighbor calculate(g_score) the cost of reaching the neighbor
        # through the current point
        for neighbor in get_neighbors(current, grid_env):
            tentative_g_score = g_score[current] + 1
            # If the neighbor is not visited yet or if the g_score is lower the the previously recorded g_score
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:

                # Then update the following dictionaries
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    # Did not find the goal (no valid path from start to goal)
    return None  # No path found


# Method for finding the neighbors
def get_neighbors(cell, grid_env):

    """
    Get valid neighbors of a cell in the grid.

    Parameters:
    - cell: Tuple representing the (x, y) coordinates of the cell.
    - grid_env: Instance of GridEnvironment representing the grid environment.

    Returns:
    - List of valid neighboring cells.
    """
    # Get coordinates of cell
    x, y = cell
    # Get all possible neighbors around a cell
    # Covers all neighbors in 3x3 square around cell
    neighbors = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    # Filter out invalid neighboring cells
    return [neighbor for neighbor in neighbors if grid_env.is_valid_cell(neighbor)]


def reconstruct_path(came_from, current):

    """
    Reconstruct the path from start to current based on the came_from dictionary.

    Parameters:
    - came_from: Dictionary containing the mapping of cells to their predecessors.
    - current: Tuple representing the (x, y) coordinates of the current cell.

    Returns:
    - List representing the reconstructed path.
    """
    # Initialize path list containing current cell
    path = [current]

    # Reconstruct path tracing from the predecessor
    while current in came_from:
        current = came_from[current]

        # Add to the list
        path.append(current)

        # Since it's in a reverse order, we will reverse it again to get the correct order
    return path[::-1]


# Function to run an algorithm once and measure time
def run_algorithm_once(algorithm, grid_env, start, goal):

    """
    Run a pathfinding algorithm once and measure the time taken.

    Parameters:
    - algorithm: String indicating the algorithm ('A*' or 'Genetic').
    - grid_env: Instance of GridEnvironment representing the grid environment.
    - start: Tuple representing the (x, y) coordinates of the start point.
    - goal: Tuple representing the (x, y) coordinates of the goal point.

    Returns:
    - Tuple containing the path, cost, and time taken by the algorithm.
    """

    # Record the time when we start the algorithm
    start_time = time.time()

    if algorithm == 'A*':

        # Call A* algorithm
        path = astar_search(grid_env, start, goal)

        # Calculate the cost of the path if it exists, else set to infinity
        cost = len(path) if path is not None else float('inf')

    else:  # algorithm == 'Genetic'
        genetic_algorithm = GeneticAlgorithm(grid_env, start, goal, population_size=10, generations=50)

        # Run the genetic algorithm and retrieve the resulting path and cost
        path, cost = genetic_algorithm.run_genetic_algorithm()

    end_time = time.time()

    # Return the path and the cost and the time it needed to execute the algorithm
    return path, cost, end_time - start_time


# Constants
size = 10
obstacle_density = 10
generations = 50
start = (9, 0)
goal = (0, 6)

# Create grid environment
grid_env = GridEnvironment(size, obstacle_density)

# Run A* algorithm once and measure time
astar_path, astar_cost, astar_time = run_algorithm_once('A*', grid_env, start, goal)
# If a path is found
if astar_path:
    print("A* Algorithm:")
    print(f"Time: {astar_time:.5f} seconds")
    print(f"Path: {astar_path}")
    print(f"Path Cost (transmissions count): {astar_cost}")

# Run Genetic Algorithm once and measure time
genetic_path, genetic_cost, genetic_time = run_algorithm_once('Genetic', grid_env, start, goal)

# If a path is found
if genetic_path is not None:
    print("\nGenetic Algorithm:")
    print(f"Time: {genetic_time:.5f} seconds")
    print(f"Path: {genetic_path}")
    print(f"Path Cost (transmissions count): {genetic_cost}")

# Visualization
plt.figure(figsize=(12, 6))

# A* Visualization
plt.subplot(1, 2, 1)
plt.imshow(grid_env.grid, cmap='binary', origin='upper')
plt.scatter(start[1], start[0], color='green', s=50, marker='o', label='Start')
plt.scatter(goal[1], goal[0], color='red', s=50, marker='x', label='Goal')
if astar_path is not None:
    path_x, path_y = zip(*astar_path)
    plt.plot(path_y, path_x, color='blue', label='A* Path')
plt.title("A* Pathfinding")
plt.legend()

# Genetic Algorithm Visualization
plt.subplot(1, 2, 2)
plt.imshow(grid_env.grid, cmap='binary', origin='upper')
plt.scatter(start[1], start[0], color='green', s=50, marker='o', label='Start')
plt.scatter(goal[1], goal[0], color='red', s=50, marker='x', label='Goal')
if genetic_path is not None:
    path_x, path_y = zip(*genetic_path)
    plt.plot(path_y, path_x, color='orange', label='Genetic Path')
plt.title("Genetic Algorithm Pathfinding")
plt.legend()

# Display the visuals
plt.show()