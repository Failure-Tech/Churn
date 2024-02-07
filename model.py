import torch
import torch.nn as nn
import pandas as pd

# Read the data
df = pd.read_csv("./churn.csv")

# Identify string columns except 'Churn'
string_columns = df.select_dtypes(include=['object']).columns.tolist()
string_columns.remove('Churn')

# Drop identified string columns
df.drop(columns=string_columns, inplace=True)

# Convert 'Churn' column to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate input features (X) and target variable (y)
X = df.drop(columns=['Churn']).values
y = df['Churn'].values.reshape(-1, 1)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the model
class Classification(nn.Module):
    def __init__(self, input_size):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# Initialize the model
input_size = X.shape[1]
model = Classification(input_size=input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model and data ready for training.")

batch_size = 64
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        epoch_loss += loss.item()

    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}, Accuracy: {(correct/total)*100:.2f}%")