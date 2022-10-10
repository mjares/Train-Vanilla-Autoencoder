import os
import pickle

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from VAEDataset import VAEDataset
from Autoencoder import Autoencoder, LeakyAE
from AutoencoderResults import AutoencoderResults
from SupervisedDataset import HyperSupervisedDataset
# Script

# Loading data
filename = 'HyperUnsupDataset_Nom_LOE_Single_Train_1_4_Shfl_10K.ds'

if os.path.getsize(filename) > 0:
    datasetFile = open(filename, 'rb')  # Creating file to read
    Dataset = pickle.load(datasetFile)
    Dataset, scales = Dataset.normalize(return_scale=True)


# Iterating
# * Missing 15 hs for lat 15
lat = [15]  # Latent dimension
hs = [15]   # Number of nodes in hidden layers
hl = [1, 2]  # Number of hidden layers
leak = [0.01]
n_epochs = 500

for lk in leak:
    for ll in lat:
        for hh in hl:
            for ss in hs:
                # Setting up model
                Results = AutoencoderResults()
                latent_dim = ll
                hidden_size = ss
                hidden_layers = hh
                leaky_slope = lk
                saveFilename = \
                    f'Model/Lat{latent_dim}_HL{hidden_layers}_HS{hidden_size}_Leak{int(leaky_slope*1000)}' \
                    f'_Lr0001_E{n_epochs}_10Kdset'
                print(saveFilename)

                model = LeakyAE(input_shape=28, latent_dim=latent_dim, hidden_layers=hidden_layers, hidden_size=hidden_size,
                                leaky_slope=leaky_slope)
                model.train()

                # create optimizer
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # mse loss
                mse = nn.MSELoss()

                full_outputs = []
                full_latent = []
                loss_log = np.zeros(n_epochs)
                for epoch in range(n_epochs):
                    loss = 0
                    total_outputs = np.zeros((Dataset.no_samples, 28))
                    latent_outputs = np.zeros((Dataset.no_samples, latent_dim))
                    # Dataset.shuffle()
                    for ii in range(Dataset.no_samples):
                        # Resetting gradients back to zero
                        optimizer.zero_grad()
                        # Loading datapoint
                        x = torch.Tensor(Dataset.features[ii, :])

                        # Compute reconstruction
                        outputs = model(x)
                        total_outputs[ii, :] = outputs.detach().numpy()
                        latent_outputs[ii, :] = model.run_encoder(x).detach().numpy()
                        # Compute loss
                        train_loss = mse(outputs, x)
                        train_loss.backward()

                        # Perform parameter update
                        optimizer.step()

                        # add minibatch training loss to epoch loss
                        loss += train_loss.item()

                    # compute the epoch training loss
                    loss = loss / Dataset.no_samples
                    loss_log[epoch] = loss
                    full_outputs.append(total_outputs)
                    full_latent.append(latent_outputs)

                    if epoch % 50 == 0:
                        print("epoch : {}/{}. loss = {:.6f}".format(epoch + 1, n_epochs, loss))
                    if loss < 0.0001:
                        break

                print("epoch : {}/{}. loss = {:.6f}".format(epoch + 1, n_epochs, loss))

                if loss > 0.01:
                    saveFilename = saveFilename + '_Fail'
                # Model
                torch.save(model.state_dict(), saveFilename + '.m')
                # Results
                # Results.loss_log = loss_log
                # Results.full_outputs = full_outputs
                # Results.full_latent_space = full_latent
                # Results.latent_dim = latent_dim
                # Results.hidden_layers = hidden_layers
                # Results.hidden_size = hidden_size
                # Results.leaky_slope = leaky_slope
                # Results.scales = scales
                # Results.notes = 'All layers use Leaky ReLU activation function'
                # datasetFile = open(saveFilename + '.rs', 'wb')  # Creating file to write
                # pickle.dump(Results, datasetFile)
                # datasetFile.close()

# plt.figure()
# plt.plot(loss_log, lw=3)
# plt.xlabel('epochs', size=20)
# plt.ylabel('total_loss', size=20)
#
# plt.show()
