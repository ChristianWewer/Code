from utility.data_io import load_data
from utility.train_network import train_vae
from networks.VAE import VAE
from torch.utils.data import DataLoader
from utility.train_network import tune_train_vae



#config:
config={"latent_dims":2, "lr":0.2}




# Load Variational autoencoder:
#vae = VAE(config["latent_dims"])


# Load dataset
dataset = "dataset"
train_set, test_set = load_data(dataset,split_size=0.3,include_initial_mass=True)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last = True)
val_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, drop_last = True)
config = {"latent_dims":2, "layers":[30,20,10],"lr":0.002}
tune_train_vae(config, train_data = train_dataloader, val_data = val_dataloader)
#vae = VAE(2)
#train_vae(vae,train_dataloader,val_dataloader,config,"super_file_name")