import torch
from networks.VAE import VAE, VAE_3hl
from torch.utils.data import DataLoader
import argparse
from utility.data_io import load_data, load_data_no_y
from utility.train_network import train_vae
from utility.plotting import plot_latent
import wandb

import seaborn as sns




def use_vae(config, output_filename="bestmodel"):
    latent_dims = config["latent_dims"]
    vae = VAE(latent_dims)

    wandb.watch(vae, log_freq=100)


    # Load dataset
    dataset = "dataset"
    train_set, test_set = load_data(dataset,split_size=0.30,include_initial_mass=True)
    print()

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last = True)
    val_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, drop_last = True)
    vae, best_epoch, best_model_loss = train_vae(vae, train_dataloader, val_dataloader, config, output_filename)


    # Load best model:
    best_model = torch.load(f"trained_models/{output_filename}.pt")

    #if latent_dims <= 3:
    #    plot_latent(best_model, train_dataloader, best_epoch, best_model_loss, latent_dims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This scripts trains a variational autoencoder")
    parser.add_argument('-o','--output-name',dest="output_name", help="Output filename")
    parser.add_argument('-d','--latent-dims',dest="latent_dims", help="Sets the number of latent dimensions minimum 2, maximum 5")
    args = parser.parse_args()

    config = {"lr":0.01, "latent_dims":int(args.latent_dims), "batch_size":128}

    device = 'cpu'

    wandb.init(project="4d-run1",name=args.output_name,config=config)    

    config = {
    "layers": [512, 256, 128],
    "lr": 0.0006,
    "batch_size": 64,
    "latent_dims": args.latent_dims
    }
    vae = VAE_3hl(int(config["latent_dims"]),config["layers"])

    train_set, val_data = load_data("dataset", 0.3, True)

    train_vae(vae,train_set,val_data,config,args.output_name,4000)
    #tune_train_vae(config, train_data=train_set)