import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def plt_lr(lr_g, lr_d, path):
    lr_g = np.array(lr_g)
    lr_d = np.array(lr_d)
    plt.figure(figsize=(9,4.5))
    plt.plot(lr_g, label='lr_g', alpha=0.7)
    plt.plot(lr_d, label='lr_d', alpha=0.7)
    plt.legend()
    plt.savefig(path + '/LR_' +'.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()

def plt_loss(gen_loss, dis_loss, path, num):
    idx = num *1000
    gen_loss = np.clip(np.array(gen_loss),a_min =-1500, a_max=1500)
    dis_loss = np.clip(-np.array(dis_loss),a_min =-1500, a_max=1500)
    plt.figure(figsize=(9,4.5))
    plt.plot(gen_loss[idx:idx+1001], label='g_loss', alpha=0.7)
    plt.plot(dis_loss[idx:idx+1001], label='d_loss', alpha=0.7)
    plt.title('Loss')
    plt.legend()
    plt.savefig(path + '/Loss_' + str(num) +'.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()


def plt_progress(real, fake, epoch, path):
    real = np.squeeze(real)
    fake = np.squeeze(fake)
    
    fig, ax = plt.subplots(2,2,figsize=(10, 10))
    ax = ax.flatten()
    fig.suptitle('Data generation, iter:' +str(epoch))
    for i in range(ax.shape[0]):
        ax[i].plot(real[i], color='red', label='Real', alpha =0.7)
        ax[i].plot(fake[i], color='blue', label='Fake', alpha =0.7)
        ax[i].legend()

    plt.savefig(path+"/line_generation"+'/Iteration_' + str(epoch) + '.png', bbox_inches='tight', pad_inches=0.5)
    plt.clf()
    
def plt_gp(norm, gp, path):
    plt.plot(np.clip(gp,a_min =-15, a_max=15), label='GP', alpha=0.7)
    plt.plot(np.clip(norm,a_min =-15, a_max=15), label='G_norm', alpha=0.7)
    plt.hlines(1, 0, len(norm), label='Opt_norm', colors='r', linestyles='dotted')
    plt.title('Gradient norm and Penalty')
    plt.legend()
    plt.savefig(path + '/GP.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()
