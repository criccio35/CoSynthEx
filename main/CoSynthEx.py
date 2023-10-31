####################################################################################################
# import libraries
####################################################################################################

# basic libraries
import numpy as np
import pandas as pd

# import timing libraries
import time
from tqdm import tqdm

# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# import libraries for cGAN
import torch
import torch.nn as nn
import math

####################################################################################################
# define functions for cGAN
####################################################################################################

# generator
class Generator(nn.Module):
    '''
    Class for the generator of a cGAN.
    
    The generator has 4 layers: one input layer, two hidden layers, and one output layer.
    
    The input layer has input_size + n_covariates nodes.
    
    The hidden layers have hidden_size nodes.
    
    The output layer has output_size nodes.
    
    The activation function for the hidden layers is ReLU.
    
    The activation function for the output layer is sigmoid.
    '''
    def __init__(self, input_size, n_covariates, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size + n_covariates, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, condition, phenotype):
        x = torch.cat([x, condition, phenotype], dim=1)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        output = self.sigmoid(self.fc3(hidden))
        return output

# discriminator
class Discriminator(nn.Module):
    '''
    Class for the discriminator of a cGAN.
    
    The discriminator has 4 layers: one input layer, two hidden layers, and one output layer.
    
    The input layer has input_size + n_covariates nodes.
    
    The hidden layers have hidden_size nodes.
    
    The output layer has 1 node.
    
    The activation function for the hidden layers is ReLU.
    
    The activation function for the output layer is sigmoid.
    
    The hidden layers have dropout with probability 0.3.
    '''
    def __init__(self, input_size, n_covariates, hidden_size):
        '''
        :param input_size: input size for the generator and discriminator.
        :type input_size: int.
        :param n_covariates: number of covariates (binary plus continuous).
        :type n_covariates: int.
        :param hidden_size: number of nodes in the hidden layers.
        :type hidden_size: int.
        '''
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size + n_covariates, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, condition, phenotype):
        x = torch.cat([x, condition, phenotype], dim=1)
        hidden = self.dropout(self.relu(self.fc1(x)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        output = self.sigmoid(self.fc3(hidden))
        return output

# define function to train the model
def train_model(G, D, criterion, G_optimizer, D_optimizer,
                num_epochs, batch_size, input_size,
                X_train, condition_train, phenotype_train):
    '''
    Function to train a cGAN. For each epoch, the function loops through the batches of the training data.
    For each batch, the function trains the discriminator and generator.
    
    :param G: instance of the generator.
    :type G: Generator.
    :param D: instance of the discriminator.
    :type D: Discriminator.
    :param G_optimizer: optimizer for the generator.
    :type G_optimizer: torch.optim.Adam.
    :param D_optimizer: optimizer for the discriminator.
    :type D_optimizer: torch.optim.Adam.
    :param num_epochs: number of epochs.
    :type num_epochs: int.
    :param batch_size: batch size.
    :type batch_size: int.
    :param input_size: input size for the generator and discriminator.
    :type input_size: int.
    :param X_train: training/real data.
    :type X_train: pandas.DataFrame.
    :param condition_train: binary covariate for the training/real data.
    :type condition_train: pandas.Series.
    :param phenotype_train: continuous covariate for the training/real data.
    :type phenotype_train: pandas.DataFrame.
    
    :return: lists of the losses for the generator and discriminator, the trained generator, and the trained discriminator.
    :rtype: list, list, Generator, Discriminator.
    '''
    D_losses = []
    G_losses = []

    num_samples = X_train.shape[0]

    for epoch in tqdm(range(num_epochs)):
        for i in range(math.ceil(num_samples/batch_size)):
            #real data
            data_real = torch.tensor(X_train.iloc[i*batch_size:(i+1)*batch_size,:].values, dtype=torch.float)
            condition_real = torch.tensor(condition_train.iloc[i*batch_size:(i+1)*batch_size].values, dtype=torch.float).view(-1,1)
            phenotype_real = torch.tensor(phenotype_train.iloc[i*batch_size:(i+1)*batch_size,:].values, dtype=torch.float)
            label_real = torch.ones(batch_size, 1)

            #fake data
            noise = torch.randn(batch_size, input_size)
            data_fake = G(noise, condition_real, phenotype_real)
            label_fake = torch.zeros(batch_size, 1)

            #train discriminator
            D_real = D(data_real, condition_real, phenotype_real)
            D_fake = D(data_fake, condition_real, phenotype_real)
            D_loss_real = criterion(D_real, label_real)
            D_loss_fake = criterion(D_fake, label_fake)
            D_loss = D_loss_real + D_loss_fake
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            #train generator
            noise = torch.randn(batch_size, input_size)
            data_fake = G(noise, condition_real, phenotype_real)
            D_fake = D(data_fake, condition_real, phenotype_real)
            G_loss = criterion(D_fake, label_real)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            #store loss
            if i==0:
                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())

    return D_losses, G_losses, G, D

# define function to plot the loss
def plot_loss(D_losses, G_losses, output_path_figures='./'):
    '''
    Function to plot the loss curves for the generator and discriminator.
    
    :param D_losses: list of the losses for the discriminator.
    :type D_losses: list
    :param G_losses: list of the losses for the generator.
    :type G_losses: list
    :param output_path_figures: path to save the figure.
    :type output_path_figures: str
    '''
    plt.figure()
    plt.plot(D_losses, label='Discriminator loss')
    plt.plot(G_losses, label='Generator loss')
    plt.legend()
    plt.savefig(output_path_figures + 'loss.png')
    plt.show()
    plt.close()


def plot_all_expression_fake_vs_real(scaled_data_real, scaled_data_fake,output_path_figures='./'):
    '''
    Function to plot the expression of all genes in the real and fake data.
    
    :param scaled_data_real: scaled real data (i.e., count data after log2 transformation and minmax normalization).
    :type scaled_data_real: pandas.DataFrame
    :param scaled_data_fake: scaled fake data (i.e., expression data as output by the generator).
    :type scaled_data_fake: pandas.DataFrame
    :param output_path_figures: path to save the figure.
    :type output_path_figures: str
    '''
    # organize the data in a single dataframe with columns: expression, and type (real/fake)
    df = pd.DataFrame(columns=['expression', 'type'])
    df['expression'] = scaled_data_fake.values.flatten().tolist() + scaled_data_real.values.flatten().tolist()

    df['type'] = ['fake'] * scaled_data_fake.values.flatten().shape[0] + \
                 ['real'] * scaled_data_real.values.flatten().shape[0]
    df['condition'] = ['control'] * (scaled_data_fake.values.flatten().shape[0] // 2) + \
                      ['stress'] * (scaled_data_fake.values.flatten().shape[0] // 2) + \
                      ['control'] * (scaled_data_real.values.flatten().shape[0] // 2) + \
                      ['stress'] * (scaled_data_real.values.flatten().shape[0] // 2)
    plt.figure()
    sns.set(font_scale=1)
    sns.displot(df, x='expression', hue='type', col='condition', kind='kde')
    plt.savefig(output_path_figures + 'expression_all.png')
    plt.show()


def plot_gene_expression_fake_vs_real(control_samples, stress_samples,
                                      scaled_data_real, scaled_data_fake,
                                      gene=None, output_path_figures='./'):
    '''
    Function to plot the expression of a gene in the real and fake data.
    
    :param control_samples: list of the control samples.
    :type control_samples: list
    :param stress_samples: list of the stress samples.
    :type stress_samples: list
    :param scaled_data_real: scaled real data (i.e., count data after log2 transformation and minmax normalization).
    :type scaled_data_real: pandas.DataFrame
    :param scaled_data_fake: scaled fake data (i.e., expression data as output by the generator).
    :type scaled_data_fake: pandas.DataFrame
    :param gene: gene to plot.
    :type gene: str
    :param output_path_figures: path to save the figure.
    :type output_path_figures: str
    
    :return: the gene that was plotted.
    :rtype: str
    '''
    if gene is None:
        # compute mean expression for each gene
        mean_expression = scaled_data_real.mean(axis=0)
        # select a random gene whose mean expression is near to 0.6
        gene = mean_expression[(mean_expression > 0.55) & (mean_expression < 0.65)].sample().index[0]

    # organize the data in a single dataframe with columns: expression, condition (control/stress), and type (real/fake)
    df = pd.DataFrame(columns=['expression', 'condition', 'type'])
    df['expression'] = scaled_data_fake.loc[control_samples, gene].tolist() + \
                       scaled_data_fake.loc[stress_samples, gene].tolist() + \
                       scaled_data_real.loc[control_samples, gene].tolist() + \
                       scaled_data_real.loc[stress_samples, gene].tolist()
    df['condition'] = ['control'] * (len(control_samples) + len(stress_samples)) + \
                      ['stress'] * (len(control_samples) + len(stress_samples))
    df['type'] = ['fake'] * len(control_samples) + ['real'] * len(stress_samples) + \
                 ['real'] * len(control_samples) + ['fake'] * len(stress_samples)

    plt.figure()
    sns.set(font_scale=1)
    ax = sns.displot(df, x='expression', hue='type', col='condition', kde=True)
    ax.fig.subplots_adjust(top=.85)
    ax.fig.suptitle('Gene: ' + gene)
    plt.savefig(output_path_figures + 'expression_' + gene + '.png')
    plt.show()

    return gene


