import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
from scipy.special import lambertw

from net.dense3 import Generator, Discriminator


#pip install torch==1.2.0 torchvision==0.4.0 
#pip install -U scikit-learn
#pip install statsmodels

class Data(object):
    def __init__(self,length, path):
        
        self.data = pd.read_csv(path,skiprows=[1])#
        self.delta = 0.6
        self.length = length
        self.init_all()
        
        
    def init_all(self):
        self.get_scalar()
        self.store_price()
        self.bid_return = self.preprocessing(self.data['Close'])
        
        self.store_dates()
        self.data_augment()
    
    def store_price(self):
        self.bid = self.data['Close'].to_numpy()
        
    def store_dates(self):
        self.dates = pd.to_datetime(self.data['date'])

        
    def get_scalar(self):
        self.scalar = StandardScaler()
        self.scalar2 = StandardScaler()
        
    def moving_window(self,x, length):
        return [x[i: i+ length] for i in range(0,(len(x)+1)-length, 4)]
    
    def preprocessing(self, data):
        #get return
        log_returns = np.log(data/data.shift(1)).fillna(0).to_numpy()
        log_returns = np.reshape(log_returns, (log_returns.shape[0],1))
        #scale the values
        self.scalar = self.scalar.fit(log_returns)
        log_returns = np.squeeze(self.scalar.transform(log_returns))
        
        log_returns_w = (np.sign(log_returns)*np.sqrt(lambertw(self.delta*log_returns**2)/self.delta)).real
        log_returns_w = log_returns_w.reshape(-1,1)
        self.scalar2 = self.scalar2.fit(log_returns_w)
        log_returns_w = np.squeeze(self.scalar2.transform(log_returns_w))
        return log_returns_w

    
    def data_augment(self):
        self.bid_return_aug = np.array(self.moving_window(self.bid_return, self.length))
        self.bid_aug = np.array(self.moving_window(self.bid, self.length))
        self.dates_aug = np.array(self.moving_window(self.dates, self.length))
        print(self.bid_return_aug.shape)
        
        
        
    def post_processing(self, return_data,init):
        
        return_data = self.scalar2.inverse_transform(return_data)
        #print(np.max(return_data))
        return_data = return_data*np.exp(0.5*self.delta*return_data**2)
        return_data = self.scalar.inverse_transform(return_data)
        return_data = np.exp(return_data)
        
        post_return = np.empty((return_data.shape[0],))
        post_return[0] = init
        for i in range(1,return_data.shape[0]):
            post_return[i] = post_return[i-1] * return_data[i]
        return post_return 

    
    def __len__(self):
        return len(self.bid_return_aug)
    
    def get_single_sample(self):
        idx = np.random.randint(self.bid_return_aug.shape[0], size=1)
        
        
        real_samples = self.bid_return_aug[idx, :]
        real_start_prices = self.bid_aug[idx, 0]
        real_samples = np.expand_dims(real_samples, axis=1)
        real_samples = torch.from_numpy(real_samples)
        
        return real_samples.float(), real_start_prices, idx
        
    def get_samples(self, G, latent_dim, n, ts_dim, conditional, use_cuda):
        noise = torch.randn((n,1,latent_dim))
        idx = np.random.randint(self.bid_return_aug.shape[0], size=n)

        real_samples = self.bid_return_aug[idx, :]
        
        real_start_prices = self.bid_aug[idx, 0]
        real_samples = np.expand_dims(real_samples, axis=1)
        real_samples = torch.from_numpy(real_samples)
        
        if conditional>0:
            noise[:,:,:conditional] = real_samples[:,:,:conditional]

        if use_cuda:
            noise = noise.cuda()
            real_samples = real_samples.cuda()
            G.cuda()

        y = G(noise)
        
        y = y.float()
        
        y = torch.cat((real_samples[:,:,:conditional].float().cpu(),y.float().cpu()), dim=2)
        
        if use_cuda:
            y = y.cuda()
        return y.float(), real_samples.float(), real_start_prices
    
