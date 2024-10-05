import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Parameters
latent_dim = 2
input_dim = 100  # Length of the sine wave
learning_rate = 1e-3
epochs = 1000


# Generate sine wave data
def generate_sine_wave_data(batch_size, num_samples):
    x = np.linspace(0, 2 * np.pi, num_samples)
    sine_waves = []
    for _ in range(batch_size):
        phase_shift = np.random.uniform(0, 2 * np.pi)
        sine_wave = np.sin(x + phase_shift)
        sine_waves.append(sine_wave)
    return np.array(sine_waves)


# Define a simple VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc21 = nn.Linear(50, latent_dim)  # Mean
        self.fc22 = nn.Linear(50, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 50)
        self.fc4 = nn.Linear(50, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss function: Reconstruction + KL Divergence
def loss_function(recon_x, x, mu, logvar):
    # MSE for sine wave reconstruction
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


# Instantiate model and optimizer
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Generate random sine wave batch data
    batch_size = 32
    data = generate_sine_wave_data(batch_size, input_dim)
    data = torch.tensor(data, dtype=torch.float32)

    # Forward pass
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Generate and plot the results
model.eval()
with torch.no_grad():
    # Generate sine waves from latent space
    z = torch.randn(10, latent_dim)
    generated_sine_waves = model.decode(z)

    # Plot generated sine waves
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.plot(generated_sine_waves[i].numpy())
        plt.title(f"Generated {i+1}")
    plt.tight_layout()
    plt.show()
