import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_shape=4, latent_dim=2, hidden_size=10, hidden_layers=1):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers

        # encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Linear(in_features=input_shape, out_features=hidden_size))
        if hidden_layers > 1:
            for ii in range(hidden_layers - 1):
                self.encoder_layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.encoder_layers.append(nn.Linear(in_features=hidden_size, out_features=latent_dim))
        # decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(in_features=latent_dim, out_features=hidden_size))
        if hidden_layers > 1:
            for ii in range(hidden_layers - 1):
                self.decoder_layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.decoder_layers.append(nn.Linear(in_features=hidden_size, out_features=input_shape))

    def forward(self, x):
        latent = self.run_encoder(x)
        x_hat = self.run_decoder(latent)
        return x_hat

    def run_encoder(self, x):
        output = F.relu(self.encoder_layers[0](x))
        if self.hidden_layers > 1:
            for ii in range(1, self.hidden_layers):
                output = F.relu(self.encoder_layers[ii](output))
        latent = F.relu(self.encoder_layers[-1](output))
        return latent

    def run_decoder(self, latent):
        output = F.relu(self.decoder_layers[0](latent))
        if self.hidden_layers > 1:
            for ii in range(1, self.hidden_layers):
                output = F.relu(self.decoder_layers[ii](output))
        x_hat = F.relu(self.decoder_layers[-1](output))
        return x_hat


class LeakyAE(nn.Module):
    def __init__(self, input_shape=4, latent_dim=2, hidden_size=10, hidden_layers=1, leaky_slope=0.01):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.leaky_slope = leaky_slope

        # encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Linear(in_features=input_shape, out_features=hidden_size))
        # nn.init.xavier_uniform(self.encoder_layers[0].weight)
        # nn.init.xavier_uniform(self.encoder_layers[0].bias)
        if hidden_layers > 1:
            for ii in range(hidden_layers - 1):
                self.encoder_layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.encoder_layers.append(nn.Linear(in_features=hidden_size, out_features=latent_dim))
        # decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(in_features=latent_dim, out_features=hidden_size))
        if hidden_layers > 1:
            for ii in range(hidden_layers - 1):
                self.decoder_layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.decoder_layers.append(nn.Linear(in_features=hidden_size, out_features=input_shape))

    def forward(self, x):
        latent = self.run_encoder(x)
        x_hat = self.run_decoder(latent)
        return x_hat

    def run_encoder(self, x):
        output = F.leaky_relu(self.encoder_layers[0](x), self.leaky_slope)
        if self.hidden_layers > 1:
            for ii in range(1, self.hidden_layers):
                output = F.leaky_relu(self.encoder_layers[ii](output), self.leaky_slope)
        latent = F.leaky_relu(self.encoder_layers[-1](output), self.leaky_slope)
        return latent

    def run_decoder(self, latent):
        output = F.leaky_relu(self.decoder_layers[0](latent), self.leaky_slope)
        if self.hidden_layers > 1:
            for ii in range(1, self.hidden_layers):
                output = F.leaky_relu(self.decoder_layers[ii](output), self.leaky_slope)
        x_hat = F.leaky_relu(self.decoder_layers[-1](output), self.leaky_slope)  # Leaky output layer
        # x_hat = F.relu(self.decoder_layers[-1](output))  # Not leaky output layer
        return x_hat
