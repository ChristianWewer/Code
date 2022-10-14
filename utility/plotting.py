import matplotlib.pyplot as plt
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



def plot_latent(all_data, labels,experiment_name,current_iteration,fold,plot_name,pca=None):
    if pca == None:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(all_data)
    else: 
        reduced_data = pca.transform(all_data)

    plt.figure(figsize=(5,5))
    plt.scatter(reduced_data[:,0],reduced_data[:,1],c=labels)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of latent space.")
    plt.colorbar()
    # set x and y lims to ensure similar plots


    plt.savefig(f"results/{experiment_name}/figures/{current_iteration}-{fold}-{plot_name}.png")
    plt.close()

    return pca