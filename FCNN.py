import torch
import torch.nn as nn
import torchinfo

# Define a custom neural network module called FCNN that extends the nn.Module class
class FCNN(nn.Module):
    # Constructor for the FCNN module
    # input_dim: size of the input to the neural network
    # drop_p: probability of dropping out a neuron in the network
    def __init__(self, input_dim=2048, drop_p=0.1):
        # Call the constructor of the parent class
        super(FCNN, self).__init__()

        # Define the architecture of the neural network as a sequence of layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024), # First linear layer with input_dim input and 1024 output dimensions
            nn.ReLU(), # ReLU activation function
            nn.Linear(1024, 512), # Second linear layer with 1024 input and 512 output dimensions
            nn.ReLU(), # ReLU activation function
            nn.Linear(512, 32), # Third linear layer with 512 input and 32 output dimensions
            nn.ReLU(), # ReLU activation function
            nn.Dropout(drop_p), # Dropout layer with probability drop_p
            nn.Linear(32, 1), # Fourth linear layer with 32 input and 1 output dimensions
            nn.Sigmoid() # Sigmoid activation function
        )

        # Initialize the weights of the linear layers using Xavier normal initialization
        self.weight_init()

    # Helper function to initialize the weights of the linear layers using Xavier normal initialization
    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    # Define the forward pass of the neural network
    def forward(self, x):
        # Pass the input x through each layer of the neural network in sequence
        for i, module in enumerate(self.classifier):
            x = module(x)
        # Return the output of the final layer as the prediction of the network
        return x
