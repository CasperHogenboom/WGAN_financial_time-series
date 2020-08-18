import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, ts_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ts_dim = ts_dim

        self.noise_to_latent = nn.Sequential(
            nn.Linear(self.latent_dim, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.ts_dim, self.ts_dim),
            #nn.Tanh()
        )

    def forward(self, input_data):
        x = self.noise_to_latent(input_data)

        return x[:, None, :]


class Discriminator(nn.Module):
    def __init__(self, ts_dim):
        super(Discriminator,self).__init__()

        self.ts_dim = ts_dim

        self.features_to_score = nn.Sequential(
            nn.Linear(self.ts_dim, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, 4*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4*self.ts_dim, 5*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(5*self.ts_dim, 5*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(5*self.ts_dim, 6*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6*self.ts_dim, 2*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*self.ts_dim, 1)

            #nn.Sigmoid() #todo add acitivation or not, whole batch has same activiation?

        )

    def forward(self, input_data):

        x = self.features_to_score(input_data)
        return x