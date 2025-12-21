# =============================================================================
# SMS Spam Classification with NLP and Optimization Algorithms
# This script demonstrates text classification using various optimization techniques
# =============================================================================
"""
This script uses custom ABC algorithm from 6-ABCAlgorithm.py for hyperparameter
optimization instead of external libraries.
"""

# Import all necessary libraries
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# NLTK for text processing
from nltk.corpus import stopwords

# Scikit-learn for ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

# Optimization libraries (Bayesian and GA)
from bayes_opt import BayesianOptimization
from deap import base, creator, tools, algorithms

# Import custom ABC algorithm from our module
# Add the current directory to path to import ABCAlgorithm
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

# =============================================================================
# Custom ABC for Hyperparameter Optimization
# =============================================================================

class ABCHyperparameterOptimizer:
    """
    ABC Algorithm adapted for hyperparameter optimization
    Based on 6-ABCAlgorithm.py implementation
    """
    
    def __init__(
        self,
        objective_function,  # Function to minimize (returns negative accuracy for maximization)
        n_params: int = 1,
        param_bounds: list = None,  # [(lower, upper), ...]
        food_number: int = 20,
        limit: int = 50,
        max_iterations: int = 50,
        seed: int = 42
    ):
        self.objective_function = objective_function
        self.n_params = n_params
        self.param_bounds = param_bounds or [(0.001, 10.0)] * n_params
        self.food_number = food_number
        self.limit = limit
        self.max_iterations = max_iterations
        
        np.random.seed(seed)
        
        # Initialize food sources
        self.foods = np.zeros((food_number, n_params))
        self.function_values = np.ones(food_number) * np.inf
        self.fitness = np.zeros(food_number)
        self.trial = np.zeros(food_number)
        
        # Best solution tracking
        self.global_best_value = np.inf
        self.global_best_solution = np.zeros(n_params)
        self.convergence_history = []
    
    def _calculate_fitness(self, func_value: float) -> float:
        """Convert function value to fitness (higher is better)"""
        if func_value >= 0:
            return 1 / (func_value + 1)
        else:
            return 1 + abs(func_value)
    
    def _init_food_source(self, index: int):
        """Initialize a single food source"""
        for j in range(self.n_params):
            lb, ub = self.param_bounds[j]
            self.foods[index, j] = np.random.uniform(lb, ub)
        
        self.function_values[index] = self.objective_function(self.foods[index])
        self.fitness[index] = self._calculate_fitness(self.function_values[index])
        self.trial[index] = 0
    
    def _generate_neighbor(self, index: int) -> np.ndarray:
        """Generate neighbor solution"""
        param = np.random.randint(0, self.n_params)
        neighbor = np.random.randint(0, self.food_number)
        while neighbor == index:
            neighbor = np.random.randint(0, self.food_number)
        
        new_solution = np.copy(self.foods[index])
        phi = np.random.uniform(-1, 1)
        new_solution[param] = self.foods[index, param] + phi * (
            self.foods[index, param] - self.foods[neighbor, param]
        )
        
        # Clip to bounds
        lb, ub = self.param_bounds[param]
        new_solution[param] = np.clip(new_solution[param], lb, ub)
        
        return new_solution
    
    def _employed_bees_phase(self):
        """Employed bees search around their food sources"""
        for i in range(self.food_number):
            new_solution = self._generate_neighbor(i)
            new_value = self.objective_function(new_solution)
            new_fitness = self._calculate_fitness(new_value)
            
            if new_fitness > self.fitness[i]:
                self.foods[i] = new_solution
                self.function_values[i] = new_value
                self.fitness[i] = new_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1
    
    def _calculate_probabilities(self) -> np.ndarray:
        """Calculate selection probabilities"""
        max_fitness = np.max(self.fitness)
        return 0.9 * (self.fitness / max_fitness) + 0.1
    
    def _onlooker_bees_phase(self):
        """Onlooker bees select food sources based on probability"""
        probabilities = self._calculate_probabilities()
        t = 0
        i = 0
        
        while t < self.food_number:
            if np.random.random() < probabilities[i]:
                t += 1
                new_solution = self._generate_neighbor(i)
                new_value = self.objective_function(new_solution)
                new_fitness = self._calculate_fitness(new_value)
                
                if new_fitness > self.fitness[i]:
                    self.foods[i] = new_solution
                    self.function_values[i] = new_value
                    self.fitness[i] = new_fitness
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1
            
            i = (i + 1) % self.food_number
    
    def _scout_bees_phase(self):
        """Scout bees abandon exhausted food sources"""
        max_trial_idx = np.argmax(self.trial)
        if self.trial[max_trial_idx] >= self.limit:
            self._init_food_source(max_trial_idx)
    
    def _memorize_best(self):
        """Remember the best solution"""
        best_idx = np.argmin(self.function_values)
        if self.function_values[best_idx] < self.global_best_value:
            self.global_best_value = self.function_values[best_idx]
            self.global_best_solution = np.copy(self.foods[best_idx])
    
    def optimize(self, verbose: bool = True) -> tuple:
        """
        Run ABC optimization
        
        Returns:
            (best_params, best_value)
        """
        # Initialize population
        for i in range(self.food_number):
            self._init_food_source(i)
        self._memorize_best()
        
        # Main loop
        for iteration in range(self.max_iterations):
            self._employed_bees_phase()
            self._onlooker_bees_phase()
            self._memorize_best()
            self._scout_bees_phase()
            
            self.convergence_history.append(self.global_best_value)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}: Best = {self.global_best_value:.6f}")
        
        return self.global_best_solution, self.global_best_value


# =============================================================================
# 1. Data Loading and Exploration
# =============================================================================

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), "DataSet", "3-SMSSpamCollection.csv")
data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'message'])

# Label distribution
label_distribution = data['label'].value_counts()
data['message_length'] = data['message'].apply(len)
message_length_stats = data['message_length'].describe()

print("Label Distribution:")
print(label_distribution)
print("\nMessage Length Statistics:")
print(message_length_stats)

# =============================================================================
# 2. Text Preprocessing
# =============================================================================

def clean_message(message):
    """Clean messages: convert to lowercase and remove punctuation"""
    message = message.lower()
    message = re.sub(r'\W', ' ', message)
    return message

data['clean_message'] = data['message'].apply(clean_message)

# Word frequency analysis
all_words = ' '.join(data['clean_message']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(10)
print("\nTop 10 Most Common Words:")
print(common_words)

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['clean_message'] = data['clean_message'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

print("\nCleaned Messages (First 5 Rows):")
print(data['clean_message'].head())

# =============================================================================
# 3. Feature Extraction
# =============================================================================

X = data['clean_message']
y = data['label'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =============================================================================
# 4. Baseline Logistic Regression
# =============================================================================
print("\n" + "="*50)
print("Baseline Logistic Regression")
print("="*50)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

# =============================================================================
# 5. Bayesian Optimization
# =============================================================================
print("\n" + "="*50)
print("Bayesian Optimization")
print("="*50)

def optimize_log_reg(C):
    model = LogisticRegression(C=C, max_iter=1000)
    accuracy = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy').mean()
    return accuracy

bayes_optimizer = BayesianOptimization(
    f=optimize_log_reg,
    pbounds={'C': (0.001, 10)},
    random_state=42,
    verbose=0,
    allow_duplicate_points=True
)

bayes_optimizer.maximize(n_iter=10)
best_C_bayes = bayes_optimizer.max['params']['C']
print(f"Best C parameter: {best_C_bayes:.4f}")

model_bayes = LogisticRegression(C=best_C_bayes, max_iter=1000)
model_bayes.fit(X_train_tfidf, y_train)
y_pred_bayes = model_bayes.predict(X_test_tfidf)

print("\nPerformance After Bayesian Optimization:")
print(f"F1 Score: {f1_score(y_test, y_pred_bayes):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_bayes):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_bayes):.4f}")

# =============================================================================
# 6. Genetic Algorithm Optimization
# =============================================================================
print("\n" + "="*50)
print("Genetic Algorithm Optimization")
print("="*50)

# Create DEAP types (check if already exists)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    return [np.random.uniform(0.001, 10)]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    C_value = individual[0]
    model = LogisticRegression(C=C_value, max_iter=1000)
    accuracy = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy').mean()
    return accuracy,

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=20)
NGEN, CXPB, MUTPB = 50, 0.5, 0.2

for gen in range(NGEN):
    if gen % 10 == 0:
        print(f"Generation {gen + 1}/{NGEN}")
    
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    for mutant in offspring:
        if np.random.rand() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

best_individual = tools.selBest(population, 1)[0]
best_C_ga = best_individual[0]
print(f"Best C value from Genetic Algorithm: {best_C_ga:.4f}")

model_ga = LogisticRegression(C=best_C_ga, max_iter=1000)
model_ga.fit(X_train_tfidf, y_train)
y_pred_ga = model_ga.predict(X_test_tfidf)

print("\nPerformance After Genetic Algorithm:")
print(f"F1 Score: {f1_score(y_test, y_pred_ga):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_ga):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_ga):.4f}")

# =============================================================================
# 7. Artificial Bee Colony (ABC) Algorithm - CUSTOM IMPLEMENTATION
# =============================================================================
print("\n" + "="*50)
print("Artificial Bee Colony (ABC) Algorithm - Custom Implementation")
print("="*50)

def abc_objective(params):
    """
    Objective function for ABC hyperparameter optimization
    Returns negative accuracy (because ABC minimizes)
    """
    C_value = params[0]
    model = LogisticRegression(C=C_value, max_iter=1000)
    accuracy = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy').mean()
    return -accuracy  # Negative because ABC minimizes

# Create and run ABC optimizer
abc_optimizer = ABCHyperparameterOptimizer(
    objective_function=abc_objective,
    n_params=1,
    param_bounds=[(0.001, 10.0)],
    food_number=20,
    limit=50,
    max_iterations=50,
    seed=42
)

best_params, best_value = abc_optimizer.optimize(verbose=True)
best_C_abc = best_params[0]
print(f"\nBest C value from ABC: {best_C_abc:.4f}")
print(f"Best accuracy: {-best_value:.4f}")

# Train with ABC optimized parameters
model_abc = LogisticRegression(C=best_C_abc, max_iter=1000)
model_abc.fit(X_train_tfidf, y_train)
y_pred_abc = model_abc.predict(X_test_tfidf)

print("\nPerformance After ABC Algorithm:")
print(f"F1 Score: {f1_score(y_test, y_pred_abc):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_abc):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_abc):.4f}")

# =============================================================================
# 8. Results Comparison
# =============================================================================
print("\n" + "="*50)
print("RESULTS COMPARISON")
print("="*50)
print(f"{'Method':<25} {'Best C':<12} {'F1 Score':<12}")
print("-" * 49)
print(f"{'Baseline':<25} {'1.0':<12} {f1_score(y_test, y_pred):.4f}")
print(f"{'Bayesian Optimization':<25} {best_C_bayes:<12.4f} {f1_score(y_test, y_pred_bayes):.4f}")
print(f"{'Genetic Algorithm':<25} {best_C_ga:<12.4f} {f1_score(y_test, y_pred_ga):.4f}")
print(f"{'ABC (Custom)':<25} {best_C_abc:<12.4f} {f1_score(y_test, y_pred_abc):.4f}")

# =============================================================================
# Summary and Explanation
# =============================================================================
'''
This code demonstrates SMS spam classification with multiple optimization methods:

1. Data Preprocessing:
   - Text cleaning, stopword removal, TF-IDF vectorization

2. Optimization Methods:
   - Bayesian Optimization: Probabilistic model-based
   - Genetic Algorithm: Evolution-inspired (DEAP library)
   - ABC Algorithm: Bee swarm intelligence (CUSTOM implementation)

3. Custom ABC Implementation:
   - Employed bees: Exploit known solutions
   - Onlooker bees: Select based on quality
   - Scout bees: Explore new random solutions

Key Insight: All methods achieve similar performance on this simple task.
Complex optimization is more beneficial for higher-dimensional problems.
'''
