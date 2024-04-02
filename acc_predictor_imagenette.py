import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class Predictor(nn.Module):
    def __init__(self, input_size):
        super(Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1, bias=False),
        )

    def forward(self, x):
        return self.model(x)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

# Load the pickle file
with open("imagenette_features.pkl", "rb") as f:
    data = pickle.load(f)

# Extract features and accuracies
features = np.array(data["features"])
accuracies = np.array(data["accuracies"]).reshape(-1, 1)

X_train = features[: int(0.8 * len(features))]
y_train = accuracies[: int(0.8 * len(accuracies))]
X_test = features[int(0.8 * len(features)) :]
y_test = accuracies[int(0.8 * len(accuracies)) :]

# Print the features and their accuracies
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# Convert numpy arrays to PyTorch tensors if needed
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Remove the mean from the accuracies
base_acc = y_train.mean()
print("Base Accuracy:", base_acc)
y_train -= base_acc
y_test -= base_acc

# Create DataLoader for batching and shuffling data
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
model = Predictor(X_train.shape[1])

print(model)

# Define loss function
criterion = nn.L1Loss()

# Define optimizer
optimizer = optim.Adam(model.parameters())

# Lists to store training and test losses
train_losses = []
test_losses = []

# Train the model
num_epochs = 75
for epoch in range(num_epochs):
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate test loss
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()

    # Append average losses
    train_losses.append(train_loss / len(train_dataloader))
    test_losses.append(test_loss / len(test_dataloader))

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}"
    )

# Save the model
save_model(model, "imagenette_acc_predictor.pth")

# Evaluate the model
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    test_loss = criterion(outputs, y_test)
    print("Evaluation Loss:", test_loss.item())

# Create a plot of training and test losses
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(
    outputs + base_acc, y_test + base_acc, alpha=0.5, c="blue", label="Predicted"
)
# x=y line
plt.plot(
    [y_test.min() + base_acc, y_test.max() + base_acc],
    [y_test.min() + base_acc, y_test.max() + base_acc],
    c="red",
    label="Actual",
)
plt.xlabel("Predicted Accuracy")
plt.ylabel("Actual Accuracy")
plt.title("Actual vs. Predicted Accuracies")

# Compute the MSE between actual and predicted accuracies
mse = ((outputs - y_test) ** 2).mean()
plt.text(0.2, 0.8, f"MSE: {mse:.2e}", transform=plt.gca().transAxes)

# Fit the plot window
plt.tight_layout()

# Save the plot
plt.savefig("losses_and_accuracies.png")