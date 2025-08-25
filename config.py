import torch

# Configuration variables
batch_size = 32
embed_size = 256
hidden_size = 512
num_epochs = 10
learning_rate = 0.001
num_layers = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
