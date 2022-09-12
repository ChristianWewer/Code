from VAE import VAE
import torch
import numpy as np
import pandas as pd 




def do_lucky_frost():
    vae = torch.load("lucky-frost-29.pt")


    arr = np.random.rand(10000,2)
    arr[:,0] = arr[:,0]*(-5)
    arr[:,1] = arr[:,1]*2-1

    tensor = torch.from_numpy(arr).float()

    generated_data = vae.decoder(tensor)

    data = generated_data.detach().numpy()
    col_names = ["drying_time", "end_temp", "mean_oven_input_temperature", "mean_blower_differential_pressure","oven_inlet_end_temperature","initial_mass","end_mc","OCN1","OCN2","OCN3","OCN4"]
    data_df = pd.DataFrame(data,columns=col_names)




def do_mild_water():
    vae = torch.load("mild_water_30.pt")


    arr = np.random.rand(1000,2)
    arr[:,0] = arr[:,0]*2.8-2
    arr[:,1] = arr[:,1]*2.6-1.3

    tensor = torch.from_numpy(arr).float()

    generated_data = vae.decoder(tensor)

    data = generated_data.detach().numpy()
    col_names = ["drying_time", "end_temp", "mean_oven_input_temperature", "mean_blower_differential_pressure","oven_inlet_end_temperature","initial_mass","end_mc","OCN1","OCN2","OCN3","OCN4"]
    data_df = pd.DataFrame(data,columns=col_names)





def do_current_best():
    vae = torch.load("bestmodel_vae.pt")


    arr = np.random.rand(1000,2)
    arr[:,0] = arr[:,0]*2.8-2
    arr[:,1] = arr[:,1]*2.6-1.3

    tensor = torch.from_numpy(arr).float()

    generated_data = vae.decoder(tensor)

    data = generated_data.detach().numpy()
    col_names = ["drying_time", "end_temp", "mean_oven_input_temperature", "mean_blower_differential_pressure","oven_inlet_end_temperature","initial_mass","end_mc","OCN1","OCN2","OCN3","OCN4"]
    data_df = pd.DataFrame(data,columns=col_names)

if __name__ == "__main__":
    #do_mild_water()
    do_current_best()

    