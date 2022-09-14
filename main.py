import torch
import torchvision
from VAE import AE, VAE
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

import wandb



def train(autoencoder, data, epochs=1000):
    autoencoder.train()
    opt = torch.optim.Adam(autoencoder.parameters(),lr=0.03)
    for epoch in range(epochs):
        for x in data:
            opt.zero_grad()
            x_hat = autoencoder(x[0])
            loss = ((x[0] - x_hat)**2).sum()
            loss.backward()

            opt.step()

        print(epoch, loss)
    return autoencoder

def train_vae(vae, data,config,output_filename,epochs=5000):
    vae.train()
    best_model_loss = np.inf
    opt = torch.optim.Adam(vae.parameters(),lr=config["lr"])
    for epoch in range(epochs):
        train_loss = 0
        loss_step = 0
        for x in data:
            opt.zero_grad()
            x_hat = vae(x[0])
            recon_loss = ((x[0] - x_hat)**2).sum()
            KL_divergence = vae.encoder.kl
            loss = recon_loss + KL_divergence
            loss.backward()
            opt.step()
            train_loss += loss
            loss_step += 1


        if train_loss < best_model_loss:
            best_model_loss = train_loss
            best_epoch = epoch
            torch.save(vae,f"trained_models/{output_filename}.pt")
        

        wandb.log({"epoch":epoch,"loss":train_loss/loss_step,"recon_loss":recon_loss/loss_step, "KL-divergence":KL_divergence/loss_step},step=epoch)
    
    print("Best epoch:", best_epoch)

    return vae, best_epoch, best_model_loss


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



def use_autoencoder():
    latent_dims = 2
    autoencoder = AE(latent_dims)


    oven_df = pd.read_csv("dataset.csv")
    oven_df = oven_df[["drying_time", "end_temp", "mean_oven_input_temperature", "mean_blower_differential_pressure","oven_inlet_end_temperature","initial_mass","end_mc","oven_chamber_position"]]
    ocps = pd.get_dummies(oven_df["oven_chamber_position"]).rename(columns={1:"OCN1",2:"OCN2",3:"OCN3",4:"OCN4"})
    oven_df = pd.concat([oven_df, ocps],axis=1)
    oven_df = oven_df.drop(columns=["oven_chamber_position"])

    x = oven_df.drop(columns=["end_mc"]).values # for plot
    ys = oven_df["end_mc"]  # for plot


    x_train = torch.from_numpy(oven_df.values).float()
    y_train = torch.squeeze(torch.from_numpy(np.array(ys)).float()) #for plot

    train_dataset = TensorDataset(x_train)
    data = DataLoader(train_dataset, batch_size=32,shuffle=True,drop_last=True)
    autoencoder = train(autoencoder, data)


    plot_dataset = TensorDataset(x_train, y_train)
    plot_data = DataLoader(plot_dataset)

    plot_latent(autoencoder, plot_data)


def use_vae(config, output_filename="bestmodel"):
    latent_dims = config["latent_dims"]
    vae = VAE(latent_dims)

    wandb.watch(vae, log_freq=100)


    oven_df = pd.read_csv("dataset.csv")
    oven_df = oven_df[["drying_time", "end_temp", "mean_oven_input_temperature", "mean_blower_differential_pressure","oven_inlet_end_temperature","initial_mass","end_mc","oven_chamber_position"]]
    ocps = pd.get_dummies(oven_df["oven_chamber_position"]).rename(columns={1:"OCN1",2:"OCN2",3:"OCN3",4:"OCN4"})
    oven_df = pd.concat([oven_df, ocps],axis=1)
    oven_df = oven_df.drop(columns=["oven_chamber_position"])

    x = oven_df.drop(columns=["end_mc"]).values # for plot
    ys = oven_df["end_mc"]  # for plot


    x_train = torch.from_numpy(oven_df.values).float()
    y_train = torch.squeeze(torch.from_numpy(np.array(ys)).float()) #for plot

    train_dataset = TensorDataset(x_train)
    data = DataLoader(train_dataset, batch_size=config["batch_size"],shuffle=True,drop_last=True)
    vae, best_epoch, best_model_loss = train_vae(vae, data, config, output_filename)


    plot_dataset = TensorDataset(x_train, y_train)
    plot_data = DataLoader(plot_dataset)


    # Load best model:
    best_model = torch.load(f"trained_models/{output_filename}.pt")

    if latent_dims <= 3:
        plot_latent(best_model, plot_data, best_epoch, best_model_loss, latent_dims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This scripts trains a variational autoencoder")
    parser.add_argument('-o','--output-name',dest="output_name", help="Output filename")
    parser.add_argument('-d','--latent-dims',dest="latent_dims", help="Sets the number of latent dimensions minimum 2, maximum 5")
    args = parser.parse_args()

    config = {"lr":0.01, "latent_dims":int(args.latent_dims), "batch_size":128}

    device = 'cpu'

    wandb.init(project="VAE-training-1",name=args.output_name,config=config)    
    #use_autoencoder()
    use_vae(config, args.output_name)