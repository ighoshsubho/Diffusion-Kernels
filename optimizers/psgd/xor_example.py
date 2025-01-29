import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from kron_psgd import KronPSGD

class XORNet(nn.Module):
    def __init__(self, hidden_size=4):
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

# Generate XOR dataset
def create_xor_dataset():
    X = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)
    
    y = torch.tensor([
        [0],
        [1],
        [1],
        [0]
    ], dtype=torch.float32)
    
    return X, y

def train_xor_kfpsgd():
    model = XORNet(hidden_size=4)
    X, y = create_xor_dataset()
    
    optimizer = KronPSGD(
        model.parameters(),
        lr=0.1,                    # Learning rate for parameters
        lr_precond=0.1,           # Learning rate for preconditioner
        momentum=0.9,             # Momentum factor
        max_size=10,              # Max size for triangular matrices
        max_skew=2.0,             # Max skewness before using diagonal
        step_normalizer='2nd',    # Use second-order normalization
        precond_type="Newton"     # Use Newton-type preconditioner
    )
    
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(1000):
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            return loss
            
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')
    
    return model, losses

model, losses = train_xor_kfpsgd()

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)

def test_xor_model(model):
    X, y = create_xor_dataset()
    with torch.no_grad():
        output = model(X)
        predictions = (output > 0.5).float()
        
    print("\nTest Results:")
    print("Input  Target  Prediction  Raw Output")
    print("-" * 45)
    for i in range(len(X)):
        print(f"{X[i].numpy()}  {y[i].item():.0f}       {predictions[i].item():.0f}          {output[i].item():+.3f}")

test_xor_model(model)

def plot_decision_boundary(model):
    x_range = torch.linspace(-0.5, 1.5, 100)
    y_range = torch.linspace(-0.5, 1.5, 100)
    X, Y = torch.meshgrid(x_range, y_range)
    
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        Z = model(grid).reshape(X.shape)
    
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='RdBu')
    plt.colorbar(label='Model Output')
    
    data_X, data_y = create_xor_dataset()
    plt.scatter(data_X[data_y[:, 0]==0, 0], data_X[data_y[:, 0]==0, 1], 
               c='red', marker='o', label='Class 0')
    plt.scatter(data_X[data_y[:, 0]==1, 0], data_X[data_y[:, 0]==1, 1], 
               c='blue', marker='x', label='Class 1')
    
    plt.title('XOR Decision Boundary')
    plt.xlabel('Input X1')
    plt.ylabel('Input X2')
    plt.legend()
    plt.grid(True)

plot_decision_boundary(model)
plt.show()