# =============================================================================
# ANN vs SNN Comparison on Wine Dataset
# This script compares Artificial Neural Networks with Spiking Neural Networks
# =============================================================================

# Import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Norse library for Spiking Neural Networks
import norse
from norse.torch import LIFCell, LIFParameters


# =============================================================================
# Data Loading and Analysis
# =============================================================================

class DataLoader:
    """Class for loading and managing the Wine dataset"""
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.target_names = None

    def load_data(self):
        """Load the Wine dataset"""
        wine = load_wine()
        self.data = wine
        self.X = wine.data
        self.y = wine.target
        self.target_names = wine.target_names
        return self.X, self.y

    def get_data_info(self):
        """Get information about the dataset"""
        return {
            "feature_shape": self.X.shape,
            "target_shape": self.y.shape,
            "target_names": self.target_names
        }


class DataAnalyzer:
    """Class for analyzing the dataset"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, X.shape[1] + 1)])

    def describe_data(self):
        """Get statistical summary of features"""
        return self.df.describe()

    def class_distribution(self):
        """Get class distribution"""
        class_counts = pd.Series(self.y).value_counts()
        return class_counts

    def feature_correlation(self):
        """Get feature correlation matrix"""
        return self.df.corr()


# =============================================================================
# Artificial Neural Network (ANN) Model
# =============================================================================

class ANNModel(nn.Module):
    """Simple feedforward neural network"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ANNTrainer:
    """Trainer class for ANN model"""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, epochs=1000, batch_size=8, lr=0.001):
        self.model = model
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        """Train the ANN model"""
        for epoch in range(1, self.epochs + 1):
            permutation = torch.randperm(self.X_train.size()[0])
            for i in range(0, self.X_train.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = self.X_train[indices], self.y_train[indices]
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            if epoch % 100 == 1 or epoch == self.epochs:
                print(f"Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}")
        print("ANN Training Complete!")

    def evaluate(self):
        """Evaluate the ANN model"""
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(self.y_test, predicted)
            precision = precision_score(self.y_test, predicted, average='macro', zero_division=0)
            recall = recall_score(self.y_test, predicted, average='macro', zero_division=0)
            f1 = f1_score(self.y_test, predicted, average='macro', zero_division=0)
            cm = confusion_matrix(self.y_test, predicted)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print(f"Confusion Matrix:\n{cm}")
            return accuracy, precision, recall, f1


# =============================================================================
# Spiking Neural Network (SNN) Model
# =============================================================================

class SNNModel(nn.Module):
    """Spiking Neural Network using LIF (Leaky Integrate-and-Fire) neurons"""
    
    def __init__(self, input_size, output_size, hidden_size=16):
        super(SNNModel, self).__init__()
        lif_params = LIFParameters()
        self.hidden_size = hidden_size
        self.lif1 = LIFCell(p=lif_params)
        self.lif2 = LIFCell(p=lif_params)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        spiking_out1, _ = self.lif1(x)
        spiking_out2, _ = self.lif2(spiking_out1)
        spiking_out2_flat = spiking_out2.view(spiking_out2.size(0), -1)
        output = self.fc(spiking_out2_flat)
        return output


class SNNTrainer:
    """Trainer class for SNN model"""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, epochs=1000, batch_size=8, lr=0.001):
        self.model = model
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        """Train the SNN model"""
        for epoch in range(1, self.epochs + 1):
            permutation = torch.randperm(self.X_train.size()[0])
            for i in range(0, self.X_train.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = self.X_train[indices], self.y_train[indices]
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            if epoch % 100 == 1 or epoch == self.epochs:
                print(f"Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}")
        print("SNN Training Complete!")

    def evaluate(self):
        """Evaluate the SNN model"""
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(self.y_test, predicted)
            precision = precision_score(self.y_test, predicted, average='macro', zero_division=0)
            recall = recall_score(self.y_test, predicted, average='macro', zero_division=0)
            f1 = f1_score(self.y_test, predicted, average='macro', zero_division=0)
            cm = confusion_matrix(self.y_test, predicted)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print(f"Confusion Matrix:\n{cm}")
            return accuracy, precision, recall, f1


# =============================================================================
# Main Application
# =============================================================================

class Main:
    """Main application class"""
    
    def __init__(self):
        self.data_loader = DataLoader()

    def run(self):
        # Load and prepare data
        X, y = self.data_loader.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Analyze data
        data_analyzer = DataAnalyzer(X, y)
        print("="*50)
        print("Data Analysis")
        print("="*50)
        print("Statistical Summary:\n", data_analyzer.describe_data())
        print("\nClass Distribution:\n", data_analyzer.class_distribution())
        print("\nFeature Correlation:\n", data_analyzer.feature_correlation())

        # Train and evaluate ANN
        print("\n" + "="*50)
        print("ANN Training Starting...")
        print("="*50)
        ann_model = ANNModel(input_size=X_train.shape[1], hidden_size=16, output_size=3)
        ann_trainer = ANNTrainer(ann_model, X_train, y_train, X_test, y_test, epochs=1000)
        ann_trainer.train()
        ann_trainer.evaluate()

        # Train and evaluate SNN
        print("\n" + "="*50)
        print("SNN Training Starting...")
        print("="*50)
        snn_model = SNNModel(input_size=X_train.shape[1], hidden_size=16, output_size=3)
        snn_trainer = SNNTrainer(snn_model, X_train, y_train, X_test, y_test, epochs=1000)
        snn_trainer.train()
        snn_trainer.evaluate()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main = Main()
    main.run()


# =============================================================================
# Summary and Explanation
# =============================================================================
'''
This script compares Artificial Neural Networks (ANN) with Spiking Neural Networks (SNN):

1. Data Loading:
   - Uses the Wine dataset from scikit-learn
   - 178 samples, 13 features, 3 classes

2. Artificial Neural Network (ANN):
   - Traditional feedforward network
   - Uses ReLU activation and CrossEntropy loss
   - Trained with mini-batch gradient descent

3. Spiking Neural Network (SNN):
   - Uses LIF (Leaky Integrate-and-Fire) neurons from Norse library
   - Biologically inspired, encodes information as spikes
   - More energy-efficient for neuromorphic hardware

Key Differences:
- ANN: Continuous values, rate-based encoding
- SNN: Discrete spikes, temporal encoding

Use Cases:
- ANN: General-purpose deep learning tasks
- SNN: Neuromorphic computing, edge devices, low-power applications
'''