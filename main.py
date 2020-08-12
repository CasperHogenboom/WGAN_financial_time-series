import torch
import torch.optim as optim
from net.dense3_skip2 import Generator, Discriminator
from training import Trainer
from torch.autograd import Variable
import os
import datetime

latent_dim = 10 #####50
ts_dim = 23#12 #33
conditional=3#2

time = datetime.datetime.utcnow() + datetime.timedelta(hours = 2)
scorepath = "/opt/app-root/git_repositories/wgan_gp_fx/output/result_" + str(time.isoformat(timespec='minutes'))

plot_scorepath = scorepath +"/line_generation"
if not os.path.exists(scorepath):
    os.makedirs(scorepath)
    os.makedirs(plot_scorepath)

generator = Generator(latent_dim=latent_dim, ts_dim=ts_dim,condition=conditional)
discriminator = Discriminator(ts_dim=ts_dim)
path_gen = "/opt/app-root/git_repositories/wgan_gp_fx/output/result_2020-06-18T17:12/gen_CWGAN-GP_model_Dense3_concat_fx.pt"
path_dis = "/opt/app-root/git_repositories/wgan_gp_fx/output/result_2020-06-18T17:12/dis_CWGAN-GP_model_Dense3_concat_fx.pt"
#generator.load_state_dict(torch.load(path_gen))
#discriminator.load_state_dict(torch.load(path_dis))

# Init optimizers
lr_a = 1e-4
lr_b = 1e-4
betas = (0, 0.9)

#G_opt = optim.Adam(generator.parameters(), lr=lr_a, betas=betas)
#D_opt = optim.Adam(discriminator.parameters(), lr=lr_b, betas=betas)
G_opt = optim.RMSprop(generator.parameters(), lr=lr_a)
D_opt = optim.RMSprop(discriminator.parameters(), lr=lr_b)

D_scheduler = optim.lr_scheduler.CyclicLR(D_opt, base_lr = 1e-4, max_lr= 8e-4, step_size_up=100, step_size_down=900, mode ='triangular')
G_scheduler = optim.lr_scheduler.CyclicLR(G_opt, base_lr = 1e-4, max_lr= 6e-4, step_size_up=100, step_size_down=900, mode ='triangular')

epochs = 25000
batch_size = 58
use_cuda = torch.cuda.is_available()
print(use_cuda)

train = Trainer(generator, discriminator, G_opt, D_opt, batch_size, scorepath, ts_dim, latent_dim, D_scheduler, G_scheduler, use_cuda=use_cuda)
train.train(epochs=epochs)

