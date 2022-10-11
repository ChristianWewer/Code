import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from sklearn.decomposition import PCA



def plot_latent(autoencoder, data, best_epoch, best_model_loss, latent_dims):
    z_list = []
    y_list = []
    for i, (x,y) in enumerate(data):
        z = autoencoder.encoder(x).detach().numpy()
        z_list.append(z)
        y_list.append(y)
    if latent_dims == 2:
        plt.scatter(np.array(z_list)[:,0][:,0],np.array(z_list)[:,0][:,1],c=y_list)
    elif latent_dims == 3:
        plt.scatter(np.array(z_list)[:,0][:,0],np.array(z_list)[:,0][:,1],np.array(z_list)[:,0][:,2],c=y_list)
    plt.colorbar()
    plt.title(f"VAE latent space - Best epoch: {best_epoch}, best loss: {best_model_loss}")
    wandb.log({"chart":wandb.Image(plt)})
