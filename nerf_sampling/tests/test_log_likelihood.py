import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from nerf_sampling.nerf_pytorch.loss_functions import gaussian_log_likelihood


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.basic_network = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, input_tensor):
        return self.basic_network(input_tensor)


torch.manual_seed(42)

# Mean and standard deviation of the true Gaussian
mu_true = torch.tensor(-2.0)
s = torch.tensor(0.4)

# Generate samples from the true Gaussian
samples = torch.normal(mu_true, s, size=(100, 1))

# Initialize the neural network
basic_network = NN()

# Input tensor for the network (dummy input, since we only care about the output)
input_tensor = torch.tensor([[0.0]])

# Optimizer
optimizer = optim.Adam(basic_network.parameters(), lr=0.001)

# Training loop
for i in range(200):
    mu_pred = basic_network(input_tensor)
    optimizer.zero_grad()
    loss = gaussian_log_likelihood(samples, mu_pred, s)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.item()}, Predicted mu: {mu_pred.item()}")

# Final predicted mean
mu_final = basic_network(input_tensor).item()

# Plotting the result
# x = torch.linspace(mu_true - 3 * s, mu_true + 3 * s, steps=100)
# y = (1 / torch.sqrt(2 * torch.pi * s**2)) * torch.exp(
#     -((x - mu_final) ** 2) / (2 * s**2)
# ).detach()
# plt.plot(x, y, label="Estimated Gaussian")
# plt.hist(samples.numpy(), bins=30, density=True, alpha=0.6, color="g", label="Samples")
# plt.legend()
# plt.show()
