# Required libraries
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords

# Load the dataset
file_path = r"...\Machine Learning\DataSet\3-SMSSpamCollection.csv"
data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'message'])

# 1. Label Distribution and Message Length Analysis

# Label distribution (number of spam and ham messages)
label_distribution = data['label'].value_counts()

# Calculate message lengths
data['message_length'] = data['message'].apply(len)

# General statistics about message length
message_length_stats = data['message_length'].describe()

# Print results
print("Label Distribution:")
print(label_distribution)
print("\nMessage Length Statistics:")
print(message_length_stats)

# 2. Distribution of Message Lengths (Visualization)
'''
plt.figure(figsize=(10,6))
data[data['label'] == 'ham']['message_length'].hist(bins=50, alpha=0.5, label='Ham')
data[data['label'] == 'spam']['message_length'].hist(bins=50, alpha=0.5, label='Spam')
plt.title('Distribution of Message Lengths')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()
'''
# 3. Word Frequency Analysis

# Clean messages (convert to lowercase, remove punctuation)
def clean_message(message):
    message = message.lower()
    message = re.sub(r'\W', ' ', message)
    return message

# Add cleaned messages
data['clean_message'] = data['message'].apply(clean_message)

# Word frequency analysis
all_words = ' '.join(data['clean_message']).split()
word_freq = Counter(all_words)

# Top 10 most common words
common_words = word_freq.most_common(10)
print("\nTop 10 Most Common Words:")
print(common_words)

# 4. Data Preprocessing and Cleaning

# Load stopwords from NLTK
# Use 'turkish' instead of 'english' for Turkish stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
data['clean_message'] = data['clean_message'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Show the first few cleaned messages
print("\nCleaned Messages (First 5 Rows):")
print(data['clean_message'].head())

from sklearn.model_selection import train_test_split

X = data['clean_message']
y = data['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

from sklearn.metrics import f1_score

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)


# Required libraries for Bayesian Optimization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

# Set up the Bayesian optimization function
def optimize_log_reg(C):
    model = LogisticRegression(C=C, max_iter=1000)
    accuracy = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy').mean()
    return accuracy

# Set up Bayesian optimization
bayes_optimizer = BayesianOptimization(
    f=optimize_log_reg,  # Function to optimize
    pbounds={'C': (0.001, 10)},  # Range of C parameter
    random_state=42,  # Random seed for reproducibility
    verbose=2,  # Show progress
    allow_duplicate_points=True  # Allow duplicate values for C
)

# Run the optimization
bayes_optimizer.maximize(n_iter=10)

# Get the best parameters
best_params = bayes_optimizer.max['params']
best_C = best_params['C']

print(f"Best C parameter: {best_C}")

# Retrain the model with the optimized parameters
model_optimized = LogisticRegression(C=best_C, max_iter=1000)
model_optimized.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_optimized = model_optimized.predict(X_test_tfidf)

# Performance metrics
f1_optimized = f1_score(y_test, y_pred_optimized)
precision_optimized = precision_score(y_test, y_pred_optimized)
recall_optimized = recall_score(y_test, y_pred_optimized)

# Print results
print("Performance After Bayesian Optimization:")
print("F1 Score:", f1_optimized)
print("Precision:", precision_optimized)
print("Recall:", recall_optimized)


# Required libraries for Genetic Algorithm
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create fitness function (maximize accuracy)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create individual function (C value in range 0.001 - 10)
def create_individual():
    return [np.random.uniform(0.001, 10)]

# Create population function
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define fitness function
def evaluate(individual):
    C_value = individual[0]
    model = LogisticRegression(C=C_value, max_iter=1000)
    accuracy = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy').mean()
    return accuracy,

# Set up crossover, mutation, and selection functions for DEAP
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
toolbox.register("evaluate", evaluate)  # Fitness function

# Create population
population = toolbox.population(n=20)  # Population of 20 individuals

# Genetic algorithm parameters
NGEN = 100  # Number of generations
CXPB = 0.5  # Crossover probability
MUTPB = 0.2  # Mutation probability

# Run the genetic algorithm
for gen in range(NGEN):
    print(f"-- Generation {gen + 1} --")
    
    # Calculate fitness values for individuals
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Select the best individuals
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    # Apply mutation
    for mutant in offspring:
        if np.random.rand() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Recalculate fitness values
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Create new generation
    population[:] = offspring

# Select the best individual
best_individual = tools.selBest(population, 1)[0]
best_C = best_individual[0]
print(f"Best C value from Genetic Algorithm: {best_C}")

# Retrain model with optimized parameters
model_optimized = LogisticRegression(C=best_C, max_iter=1000)
model_optimized.fit(X_train_tfidf, y_train)

# Make predictions on test set
y_pred_optimized = model_optimized.predict(X_test_tfidf)

# Performance metrics
f1_optimized = f1_score(y_test, y_pred_optimized)
precision_optimized = precision_score(y_test, y_pred_optimized)
recall_optimized = recall_score(y_test, y_pred_optimized)

# Print results
print("Performance After Genetic Algorithm:")
print("F1 Score:", f1_optimized)
print("Precision:", precision_optimized)
print("Recall:", recall_optimized)


# Required libraries for Artificial Bee Colony (ABC) Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from niapy.algorithms.basic import ArtificialBeeColonyAlgorithm
from niapy.task import Task
from niapy.problems import Problem

# Define problem class for ABC
class LogisticRegressionProblem(Problem):
    def __init__(self):
        super().__init__(dimension=1, lower=0.001, upper=10)  # Define bounds for C parameter
       
    def _evaluate(self, solution):
        C_value = solution[0]
        model = LogisticRegression(C=C_value, max_iter=1000)
        accuracy = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy').mean()
        return -accuracy  # Minimize negative accuracy to maximize accuracy

# Set up and run ABC optimization
task = Task(problem=LogisticRegressionProblem(), max_iters=100)  # Run for 100 iterations
abc = ArtificialBeeColonyAlgorithm(population_size=20)  # 20 bee population
best_solution = abc.run(task)  # Run and get the best solution

# Convert best C value to float
best_C = float(best_solution[0])
print(f"Best C value from ABC: {best_C}")

# Train logistic regression model with the best C value
model_optimized = LogisticRegression(C=best_C, max_iter=1000)
model_optimized.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_optimized = model_optimized.predict(X_test_tfidf)

# Performance metrics
f1_optimized = f1_score(y_test, y_pred_optimized)
precision_optimized = precision_score(y_test, y_pred_optimized)
recall_optimized = recall_score(y_test, y_pred_optimized)

# Print results
print("Performance After ABC Algorithm:")
print("F1 Score:", f1_optimized)
print("Precision:", precision_optimized)
print("Recall:", recall_optimized)
