import torch
from torch import nn, optim

# 1. Create the data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Shape: (4, 1) - 4 rows, 1 column each
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Shape: (4, 1)

# 2. Build the model
model = nn.Linear(in_features=1, out_features=1)

# 3. Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Train the model
for step in range(10000):
    optimizer.zero_grad()
    guess = model(x)
    loss = loss_function(guess, y)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# 5. Test it
test = torch.tensor([[50.0]])
print(f"\nPrediction for 10: {model(test).item():.2f}")  # Should be close to 20