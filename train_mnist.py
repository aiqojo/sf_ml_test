"""
Simple MNIST classification with three models:
1. Linear (Logistic Regression)
2. XGBoost
3. Neural Network (1 Dense Layer)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def load_mnist():
    """Load MNIST dataset"""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Convert to numpy arrays
    x_train = train_dataset.data.numpy().astype('float32') / 255.0
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy().astype('float32') / 255.0
    y_test = test_dataset.targets.numpy()
    
    # Flatten images for linear and XGBoost models
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    return (x_train_flat, y_train, x_test_flat, y_test,
            x_train, x_test)


def train_linear_model(x_train, y_train, x_test, y_test):
    """Train Logistic Regression model"""
    print("\n" + "="*50)
    print("Training Linear Model (Logistic Regression)")
    print("="*50)
    
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy


def train_xgboost_model(x_train, y_train, x_test, y_test):
    """Train XGBoost model"""
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy


class SimpleNN(nn.Module):
    """Neural Network with 1 Dense Layer"""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_nn_model(x_train, y_train, x_test, y_test):
    """Train Neural Network with 1 Dense Layer"""
    print("\n" + "="*50)
    print("Training Neural Network (1 Dense Layer)")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Acc: {epoch_acc:.2f}%")
    
    # Evaluation
    model.eval()
    x_test_tensor = x_test_tensor.to(device)
    with torch.no_grad():
        outputs = model(x_test_tensor)
        _, predicted = torch.max(outputs, 1)
        test_accuracy = accuracy_score(y_test, predicted.cpu().numpy())
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return model, test_accuracy


def main():
    """Main training function"""
    print("Loading MNIST dataset...")
    x_train_flat, y_train, x_test_flat, y_test, x_train_nn, x_test_nn = load_mnist()
    
    print(f"Training set shape: {x_train_flat.shape}")
    print(f"Test set shape: {x_test_flat.shape}")
    
    results = {}
    
    # Train Linear Model
    linear_model, linear_acc = train_linear_model(
        x_train_flat, y_train, x_test_flat, y_test
    )
    results['Linear'] = linear_acc
    
    # Train XGBoost Model
    xgb_model, xgb_acc = train_xgboost_model(
        x_train_flat, y_train, x_test_flat, y_test
    )
    results['XGBoost'] = xgb_acc
    
    # Train Neural Network
    nn_model, nn_acc = train_nn_model(
        x_train_nn, y_train, x_test_nn, y_test
    )
    results['Neural Network'] = nn_acc
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    for model_name, accuracy in results.items():
        print(f"{model_name:20s}: {accuracy:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
