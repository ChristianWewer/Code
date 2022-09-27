import torch
from torch.utils.data import  TensorDataset
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd

def load_data(dataset: str,split_size=0, include_initial_mass=True) -> tuple:
    """ 
    Inputs: 
    - dataset: (str) name of a csv dataset without .csv
    - split_size: (Int) train/test split in %
    - include_initial_conditions: bool
    Output: 
    - train_data, test_data (tuple) """

    oven_df = pd.read_csv("{}.csv".format(dataset))

    ocps = pd.get_dummies(oven_df["oven_chamber_position"]).rename(columns={1:"OCN1",2:"OCN2",3:"OCN3",4:"OCN4"})
    oven_df = pd.concat([oven_df, ocps],axis=1)
    oven_df = oven_df.drop(columns=["oven_chamber_position"])     
    
    
    if include_initial_mass:
        X = oven_df[["drying_time","end_temp","mean_oven_input_temperature","mean_blower_differential_pressure","oven_inlet_end_temperature","initial_mass","end_mc","OCN1","OCN2","OCN3","OCN4"]].values
    else:
        X = oven_df[["drying_time","end_temp","mean_oven_input_temperature","mean_blower_differential_pressure","oven_inlet_end_temperature","end_mc","oven_chamber_position","OCN1","OCN2","OCN3","OCN4"]].values


    Y = oven_df["end_mc"]
    if split_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = split_size, random_state=random.randint(1,1000))  # really shouldn't be doing split unless absolutely neccessary

        x_test = torch.from_numpy(x_test).float()
        y_test = torch.squeeze(torch.from_numpy(np.array(y_test)).float())
        test_tensor = TensorDataset(x_test, y_test)
    else:
        x_train=X
        y_train=Y
        x_test = None
        y_test = None
        test_tensor = None

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.squeeze(torch.from_numpy(np.array(y_train)).float())

    train_tensor = TensorDataset(x_train, y_train)

    return (train_tensor, test_tensor)



def load_data_no_y(dataset: str,split_size=0, include_initial_mass=True) -> tuple:
    """ 
    Inputs: 
    - dataset: (str) name of a csv dataset without .csv
    - split_size: (Int) train/test split in %
    - include_initial_conditions: bool
    Output: 
    - train_data, test_data (tuple) """

    oven_df = pd.read_csv("{}.csv".format(dataset))

    ocps = pd.get_dummies(oven_df["oven_chamber_position"]).rename(columns={1:"OCN1",2:"OCN2",3:"OCN3",4:"OCN4"})
    oven_df = pd.concat([oven_df, ocps],axis=1)
    oven_df = oven_df.drop(columns=["oven_chamber_position"])     
    
    
    if include_initial_mass:
        X = oven_df[["drying_time","end_temp","mean_oven_input_temperature","mean_blower_differential_pressure","oven_inlet_end_temperature","initial_mass","end_mc","OCN1","OCN2","OCN3","OCN4"]].values
    else:
        X = oven_df[["drying_time","end_temp","mean_oven_input_temperature","mean_blower_differential_pressure","oven_inlet_end_temperature","end_mc","oven_chamber_position","OCN1","OCN2","OCN3","OCN4"]].values


    Y = oven_df["end_mc"]
    if split_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = split_size, random_state=random.randint(1,1000))  # really shouldn't be doing split unless absolutely neccessary

        x_test = torch.from_numpy(x_test).float()
        y_test = torch.squeeze(torch.from_numpy(np.array(y_test)).float())
        test_tensor = TensorDataset(x_test, y_test)
    else:
        x_train=X
        y_train=Y
        x_test = None
        y_test = None
        test_tensor = None

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.squeeze(torch.from_numpy(np.array(y_train)).float())

    train_tensor = TensorDataset(x_train)

    return train_tensor