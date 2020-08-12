from data import Data

import torch
import numpy as np
from torch.autograd import grad as torch_grad
from eval import plt_loss, plt_progress, plt_gp, plt_lr

import psutil
import humanize
import os
import GPUtil as gpu

class Trainer():

    def __init__(self, generator, critic, gen_optimizer, critic_optimizer, batch_size, path, ts_dim, latent_dim, D_scheduler, G_scheduler, gp_weight=10,critic_iter=5, n_eval=20, use_cuda=False):
        self.G = generator
        self.D = critic
        self.G_opt = gen_optimizer
        self.D_opt = critic_optimizer
        self.G_scheduler = G_scheduler
        self.D_scheduler = D_scheduler
        self.batch_size = batch_size
        self.scorepath = path
        self.gp_weight = gp_weight
        self.critic_iter = critic_iter
        self.n_eval = n_eval
        self.use_cuda = use_cuda
        self.conditional = 3
        self.ts_dim = ts_dim
        #data_load_path = '/opt/app-root/data/data_bucket_week_3s.csv'
        data_load_path = '/opt/app-root/git_repositories/wgan_plots/data_appl/dataAPPL.csv'
        self.data = Data(self.ts_dim, data_load_path)


        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

        
        self.latent_dim = latent_dim
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'LR_G': [], 'LR_D':[]}

    def train(self, epochs):
        plot_num=0
        for epoch in range(epochs):
            for i in range(self.critic_iter):
                # train the critic
                fake_batch, real_batch, start_prices = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, n=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional, use_cuda=self.use_cuda)
                if self.use_cuda:
                    real_batch = real_batch.cuda()
                    fake_batch = fake_batch.cuda()
                    self.D.cuda()
                    self.G.cuda()
                
                d_real = self.D(real_batch)
                d_fake = self.D(fake_batch)

                grad_penalty, grad_norm_ = self._grad_penalty(real_batch, fake_batch)
                # backprop with minimizing the difference between distribution fake and distribution real
                self.D_opt.zero_grad()
                 
                d_loss = d_fake.mean() - d_real.mean() + grad_penalty.to(torch.float32)
                d_loss.backward()
                self.D_opt.step()
                

                if i == self.critic_iter-1:
                    self.D_scheduler.step()
                    self.losses['LR_D'].append(self.D_scheduler.get_lr())
                    self.losses['D'].append(float(d_loss))
                    self.losses['GP'].append(grad_penalty.item())
                    self.losses['gradient_norm'].append(float(grad_norm_))
            self.GPUs = gpu.getGPUs()
            self.gpu = self.GPUs[0]
            self.printm(epoch)
            
            self.G_opt.zero_grad()
            fake_batch_critic, real_batch_critic, start_prices = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, n=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional, use_cuda=self.use_cuda)
            if self.use_cuda:
                real_batch_critic = real_batch_critic.cuda()
                fake_batch_critic = fake_batch_critic.cuda()
                self.D.cuda()
                self.G.cuda()
            # feed-forward
            d_critic_fake = self.D(fake_batch_critic)
            d_critic_real = self.D(real_batch_critic)
            
            
            
            g_loss =  - d_critic_fake.mean()  # d_critic_real.mean()
            # backprop
            g_loss.backward()
            self.G_opt.step()
            self.G_scheduler.step()
            self.losses['LR_G'].append(self.G_scheduler.get_lr())

            # save the loss of feed forward
            self.losses['G'].append(g_loss.item())  # outputs tensor with single value

            if (epoch + 1) % self.n_eval == 0:
                if (epoch+1) % 1000 ==0:
                    plot_num = plot_num+1
                plt_loss(self.losses['G'], self.losses['D'], self.scorepath, plot_num)
                plt_gp(self.losses['gradient_norm'], self.losses['GP'], self.scorepath)
                plt_lr(self.losses['LR_G'],self.losses['LR_D'], self.scorepath)
            if (epoch + 1) % (10*self.n_eval) == 0:
                fake_lines, real_lines, start_prices = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, n=4,       ts_dim=self.ts_dim,conditional=self.conditional, use_cuda=self.use_cuda)
                real_lines = np.squeeze(real_lines.cpu().data.numpy())
                fake_lines = np.squeeze(fake_lines.cpu().data.numpy())
                real_lines = np.array([self.data.post_processing(real_lines[i], start_prices[i]) for i in range(real_lines.shape[0])])
                fake_lines = np.array([self.data.post_processing(fake_lines[i], start_prices[i]) for i in range(real_lines.shape[0])])
                plt_progress(real_lines, fake_lines, epoch, self.scorepath)
            if (epoch + 1) % 500 ==0:
                name = 'CWGAN-GP_model_Dense3_concat_fx'
                torch.save(self.G.state_dict(), self.scorepath + '/gen_' + name + '.pt')
                torch.save(self.D.state_dict(), self.scorepath + '/dis_' + name + '.pt')    
            #if (epoch + 1) % 9000 == 0:
            #    distribution_hist(self.G, self.latent_dim, self.ts_dim, self.use_cuda, self.scorepath, epoch)


    def _grad_penalty(self, real_data, gen_data):
        batch_size = real_data.size()[0]
        t = torch.rand((batch_size, 1, 1), requires_grad=True)
        t = t.expand_as(real_data)

        if self.use_cuda:
            t = t.cuda()

        # mixed sample from real and fake; make approx of the 'true' gradient norm
        interpol = t * real_data.data + (1-t) * gen_data.data

        if self.use_cuda:
            interpol = interpol.cuda()
        
        prob_interpol = self.D(interpol)
        torch.autograd.set_detect_anomaly(True)
        gradients = torch_grad(outputs=prob_interpol, inputs=interpol,
                               grad_outputs=torch.ones(prob_interpol.size()).cuda() if self.use_cuda else torch.ones(
                                   prob_interpol.size()), create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        #grad_norm = torch.norm(gradients, dim=1).mean()
        #self.losses['gradient_norm'].append(grad_norm.item())

        # add epsilon for stability
        eps = 1e-10
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1, dtype=torch.double) + eps)
        #gradients = gradients.cpu()
        # comment: precision is lower than grad_norm (think that is double) and gradients_norm is float
        return self.gp_weight * (torch.max(torch.zeros(1,dtype=torch.double).cuda() if self.use_cuda else torch.zeros(1,dtype=torch.double), gradients_norm.mean() - 1) ** 2), gradients_norm.mean().item()
    
    def printm(self,epoch):
        process = psutil.Process(os.getpid())
        print(epoch)
        print("GPU Ram free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB ".format(self.gpu.memoryFree, self.gpu.memoryUsed, self.gpu.memoryUtil*100, self.gpu.memoryTotal))       
        