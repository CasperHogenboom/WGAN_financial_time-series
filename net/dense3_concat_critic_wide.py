import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, ts_dim, condition):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ts_dim = ts_dim
        self.condition = condition

        self.noise_to_latent = nn.Sequential(
            nn.Linear(self.latent_dim, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 200),
        
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 512),
            nn.LeakyReLU(inplace=True),     
            nn.Linear(512, 10*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*self.ts_dim, 10*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*self.ts_dim, 10*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*self.ts_dim, self.ts_dim-self.condition),
            #nn.Tanh()
        )

    def forward(self, input_data):
        x = self.noise_to_latent(input_data)

        return x


class Discriminator(nn.Module):
    def __init__(self, ts_dim):
        super(Discriminator,self).__init__()

        self.ts_dim = ts_dim
        self.features_to_score = nn.Sequential(
            nn.Linear(self.ts_dim, 10*self.ts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*self.ts_dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 10*self.ts_dim),
            nn.LeakyReLU(inplace=True),

            nn.Linear(10*self.ts_dim, 1)

            #nn.Sigmoid() #todo add acitivation or not, whole batch has same activiation?

        )

    def forward(self, input_data):

        x = self.features_to_score(input_data)
        return x