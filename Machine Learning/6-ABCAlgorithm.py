# =============================================================================
# Artificial Bee Colony (ABC) Algorithm Implementation
# Based on: https://github.com/omursahin/ABCPython
# Author: Omur Sahin (Original), Educational adaptation
# =============================================================================
"""
Artificial Bee Colony (ABC) Algorithm

ABC is a swarm intelligence optimization algorithm inspired by the foraging
behavior of honey bees. The algorithm divides bees into three groups:

1. EMPLOYED BEES: Search around food sources (solutions)
2. ONLOOKER BEES: Choose food sources based on probability (fitness)
3. SCOUT BEES: Randomly search for new food sources when old ones are abandoned

This implementation demonstrates ABC on standard benchmark functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


# =============================================================================
# Benchmark Functions
# =============================================================================

def sphere(x: np.ndarray) -> float:
    """
    Sphere Function (Convex, Unimodal)
    Global minimum: f(0, 0, ..., 0) = 0
    """
    return np.sum(x ** 2)


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin Function (Non-convex, Multimodal)
    Global minimum: f(0, 0, ..., 0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock Function (Non-convex, Unimodal)
    Global minimum: f(1, 1, ..., 1) = 0
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x: np.ndarray) -> float:
    """
    Ackley Function (Non-convex, Multimodal)
    Global minimum: f(0, 0, ..., 0) = 0
    """
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def griewank(x: np.ndarray) -> float:
    """
    Griewank Function (Non-convex, Multimodal)
    Global minimum: f(0, 0, ..., 0) = 0
    """
    sum_sq = np.sum(x ** 2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1


# =============================================================================
# ABC Algorithm Configuration
# =============================================================================

class ABCConfig:
    """Configuration class for ABC algorithm parameters"""
    
    def __init__(
        self,
        food_number: int = 20,          # Number of food sources (solutions)
        dimension: int = 10,             # Problem dimension
        limit: int = 100,                # Limit for scout bee phase
        max_evaluation: int = 10000,     # Maximum function evaluations
        lower_bound: float = -5.12,      # Lower bound for variables
        upper_bound: float = 5.12,       # Upper bound for variables
        objective_function: Callable = sphere,  # Objective function to minimize
        seed: int = 42                   # Random seed for reproducibility
    ):
        self.FOOD_NUMBER = food_number
        self.DIMENSION = dimension
        self.LIMIT = limit
        self.MAX_EVALUATION = max_evaluation
        self.LOWER_BOUND = lower_bound
        self.UPPER_BOUND = upper_bound
        self.OBJECTIVE_FUNCTION = objective_function
        self.SEED = seed


# =============================================================================
# ABC Algorithm Implementation
# =============================================================================

class ArtificialBeeColony:
    """
    Artificial Bee Colony (ABC) Optimization Algorithm
    
    The algorithm mimics the foraging behavior of honey bees:
    - Employed bees exploit known food sources
    - Onlooker bees select food sources based on quality
    - Scout bees explore new random food sources
    """
    
    def __init__(self, config: ABCConfig):
        self.config = config
        np.random.seed(config.SEED)
        
        # Initialize food sources (solutions)
        self.foods = np.zeros((config.FOOD_NUMBER, config.DIMENSION))
        self.function_values = np.zeros(config.FOOD_NUMBER)
        self.fitness = np.zeros(config.FOOD_NUMBER)
        self.trial = np.zeros(config.FOOD_NUMBER)  # Abandonment counter
        self.probabilities = np.zeros(config.FOOD_NUMBER)
        
        # Best solution tracking
        self.global_best_value = np.inf
        self.global_best_solution = np.zeros(config.DIMENSION)
        self.convergence_history = []
        
        # Counters
        self.eval_count = 0
        self.cycle = 0
    
    def _calculate_fitness(self, func_value: float) -> float:
        """Convert function value to fitness value (higher is better)"""
        if func_value >= 0:
            return 1 / (func_value + 1)
        else:
            return 1 + abs(func_value)
    
    def _initialize_food_source(self, index: int):
        """Initialize a single food source randomly"""
        self.foods[index] = np.random.uniform(
            self.config.LOWER_BOUND,
            self.config.UPPER_BOUND,
            self.config.DIMENSION
        )
        self.function_values[index] = self.config.OBJECTIVE_FUNCTION(self.foods[index])
        self.fitness[index] = self._calculate_fitness(self.function_values[index])
        self.trial[index] = 0
        self.eval_count += 1
    
    def initialize(self):
        """Initialize all food sources (Phase 0)"""
        for i in range(self.config.FOOD_NUMBER):
            self._initialize_food_source(i)
        self._memorize_best()
    
    def _memorize_best(self):
        """Remember the best solution found so far"""
        best_idx = np.argmin(self.function_values)
        if self.function_values[best_idx] < self.global_best_value:
            self.global_best_value = self.function_values[best_idx]
            self.global_best_solution = np.copy(self.foods[best_idx])
    
    def _generate_neighbor(self, index: int) -> np.ndarray:
        """Generate a neighbor solution for given food source"""
        # Select random dimension to modify
        param = np.random.randint(0, self.config.DIMENSION)
        
        # Select random neighbor (different from current)
        neighbor = np.random.randint(0, self.config.FOOD_NUMBER)
        while neighbor == index:
            neighbor = np.random.randint(0, self.config.FOOD_NUMBER)
        
        # Generate new solution
        new_solution = np.copy(self.foods[index])
        phi = np.random.uniform(-1, 1)
        new_solution[param] = self.foods[index][param] + phi * (
            self.foods[index][param] - self.foods[neighbor][param]
        )
        
        # Boundary constraint
        new_solution[param] = np.clip(
            new_solution[param],
            self.config.LOWER_BOUND,
            self.config.UPPER_BOUND
        )
        
        return new_solution
    
    def send_employed_bees(self):
        """
        Employed Bee Phase: Each employed bee searches around its food source
        """
        for i in range(self.config.FOOD_NUMBER):
            if self.eval_count >= self.config.MAX_EVALUATION:
                break
            
            new_solution = self._generate_neighbor(i)
            new_value = self.config.OBJECTIVE_FUNCTION(new_solution)
            new_fitness = self._calculate_fitness(new_value)
            self.eval_count += 1
            
            # Greedy selection
            if new_fitness > self.fitness[i]:
                self.foods[i] = new_solution
                self.function_values[i] = new_value
                self.fitness[i] = new_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1
    
    def calculate_probabilities(self):
        """Calculate selection probabilities for onlooker bees"""
        max_fitness = np.max(self.fitness)
        self.probabilities = 0.9 * (self.fitness / max_fitness) + 0.1
    
    def send_onlooker_bees(self):
        """
        Onlooker Bee Phase: Select food sources based on probability
        """
        t = 0
        i = 0
        while t < self.config.FOOD_NUMBER and self.eval_count < self.config.MAX_EVALUATION:
            if np.random.random() < self.probabilities[i]:
                t += 1
                
                new_solution = self._generate_neighbor(i)
                new_value = self.config.OBJECTIVE_FUNCTION(new_solution)
                new_fitness = self._calculate_fitness(new_value)
                self.eval_count += 1
                
                # Greedy selection
                if new_fitness > self.fitness[i]:
                    self.foods[i] = new_solution
                    self.function_values[i] = new_value
                    self.fitness[i] = new_fitness
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1
            
            i = (i + 1) % self.config.FOOD_NUMBER
    
    def send_scout_bees(self):
        """
        Scout Bee Phase: Abandon exhausted food sources
        """
        max_trial_idx = np.argmax(self.trial)
        if self.trial[max_trial_idx] >= self.config.LIMIT:
            self._initialize_food_source(max_trial_idx)
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the ABC optimization algorithm
        
        Returns:
            Tuple of (best_solution, best_value)
        """
        self.initialize()
        
        while self.eval_count < self.config.MAX_EVALUATION:
            self.send_employed_bees()
            self.calculate_probabilities()
            self.send_onlooker_bees()
            self._memorize_best()
            self.send_scout_bees()
            
            self.convergence_history.append(self.global_best_value)
            self.cycle += 1
            
            if verbose and self.cycle % 50 == 0:
                print(f"Cycle {self.cycle}: Best Value = {self.global_best_value:.6f}")
        
        return self.global_best_solution, self.global_best_value
    
    def plot_convergence(self, title: str = "ABC Convergence"):
        """Plot the convergence curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history, 'b-', linewidth=2)
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('Best Value', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()


# =============================================================================
# Main Demonstration
# =============================================================================

def run_benchmark_tests():
    """Run ABC on multiple benchmark functions"""
    
    benchmarks = {
        'Sphere': (sphere, -5.12, 5.12),
        'Rastrigin': (rastrigin, -5.12, 5.12),
        'Rosenbrock': (rosenbrock, -5, 10),
        'Ackley': (ackley, -32.768, 32.768),
        'Griewank': (griewank, -600, 600)
    }
    
    results = {}
    
    print("=" * 60)
    print("Artificial Bee Colony (ABC) Algorithm Benchmark")
    print("=" * 60)
    
    for name, (func, lb, ub) in benchmarks.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name} Function")
        print(f"{'='*60}")
        
        config = ABCConfig(
            food_number=30,
            dimension=10,
            limit=100,
            max_evaluation=10000,
            lower_bound=lb,
            upper_bound=ub,
            objective_function=func,
            seed=42
        )
        
        abc = ArtificialBeeColony(config)
        best_solution, best_value = abc.optimize(verbose=False)
        
        results[name] = {
            'best_value': best_value,
            'cycles': abc.cycle,
            'evaluations': abc.eval_count
        }
        
        print(f"Best Value: {best_value:.10f}")
        print(f"Cycles: {abc.cycle}")
        print(f"Function Evaluations: {abc.eval_count}")
    
    # Summary Table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Function':<15} {'Best Value':<20} {'Cycles':<10} {'Evals':<10}")
    print("-" * 55)
    for name, data in results.items():
        print(f"{name:<15} {data['best_value']:<20.6e} {data['cycles']:<10} {data['evaluations']:<10}")
    
    return results


def plot_all_convergence():
    """Plot convergence curves for all benchmark functions"""
    
    benchmarks = {
        'Sphere': (sphere, -5.12, 5.12, 'blue'),
        'Rastrigin': (rastrigin, -5.12, 5.12, 'red'),
        'Rosenbrock': (rosenbrock, -5, 10, 'green'),
        'Ackley': (ackley, -32.768, 32.768, 'orange'),
        'Griewank': (griewank, -600, 600, 'purple')
    }
    
    plt.figure(figsize=(12, 8))
    
    for name, (func, lb, ub, color) in benchmarks.items():
        config = ABCConfig(
            food_number=30,
            dimension=10,
            limit=100,
            max_evaluation=10000,
            lower_bound=lb,
            upper_bound=ub,
            objective_function=func,
            seed=42
        )
        
        abc = ArtificialBeeColony(config)
        abc.optimize(verbose=False)
        
        plt.plot(abc.convergence_history, label=name, linewidth=2, color=color)
    
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('Best Value (log scale)', fontsize=12)
    plt.title('ABC Algorithm Convergence on Benchmark Functions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('abc_convergence.png', dpi=150)
    plt.show()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run benchmark tests
    results = run_benchmark_tests()
    
    # Plot convergence curves
    print("\nGenerating convergence plot...")
    plot_all_convergence()
    
    print("\n" + "=" * 60)
    print("ABC Algorithm demonstration complete!")
    print("=" * 60)


# =============================================================================
# Algorithm Explanation
# =============================================================================
"""
ARTIFICIAL BEE COLONY (ABC) ALGORITHM EXPLANATION

The ABC algorithm simulates the foraging behavior of honey bees:

1. EMPLOYED BEES (50% of colony):
   - Each employed bee is associated with one food source
   - They search for new food sources in the neighborhood
   - If a better source is found, they move to it
   - They share information about food quality with onlooker bees

2. ONLOOKER BEES (50% of colony):
   - Wait in the hive and observe the waggle dance of employed bees
   - Select food sources based on probability (better sources have higher prob)
   - Search around the selected food source
   - Probability formula: P(i) = 0.9 * (fitness(i) / max_fitness) + 0.1

3. SCOUT BEES:
   - If a food source is not improved for 'limit' cycles, it's abandoned
   - The employed bee becomes a scout
   - Scout bees search for new random food sources

PARAMETERS:
- food_number: Number of food sources (= half of colony size)
- dimension: Problem dimension (number of variables)
- limit: Maximum trials before abandoning a food source
- max_evaluation: Maximum function evaluations

ADVANTAGES:
- Simple to implement
- Few control parameters
- Good balance between exploration and exploitation
- Effective for both unimodal and multimodal functions

APPLICATIONS:
- Function optimization
- Neural network training
- Feature selection
- Engineering design problems
- Scheduling and routing problems
"""
