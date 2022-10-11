import torch
import os
import torch.nn as nn
from torch import optim
import numpy as np
import wandb
from ray import tune
from networks.VAE import VAE_3hl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from utility.torchtools import EarlyStopping


def tune_train_vae(config, train_data = None, kfolds=10):

    max_epochs = 4000

    kfold = KFold(n_splits=kfolds)


    min_fold_validation_loss = []

    for fold, (train_index, test_index) in enumerate(kfold.split(train_data)):

        vae = VAE_3hl(int(config["latent_dims"]),config["layers"])

        earlystopper = EarlyStopping(patience=200,verbose=False)

        validation_loss_list = []
        
        train_set = train_data[train_index]
        val_set = train_data[test_index]

        train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=True)


        opt = torch.optim.Adam(vae.parameters(),lr=config["lr"])
        vae.train()
        for epoch in range(max_epochs):
            train_loss = 0
            loss_step = 0
            for x in train_dataloader:
                opt.zero_grad()
                x_hat = vae(x[0])
                recon_loss = ((x[0] - x_hat)**2).sum()
                KL_divergence = vae.encoder.kl
                loss = recon_loss + KL_divergence
                loss.backward()
                opt.step()
                train_loss += loss
                loss_step += 1



            val_loss = 0
            val_steps = 0
            vae.eval()
            for x in val_dataloader:
                with torch.no_grad():
                    x_hat = vae(x[0])

                    val_recon_loss = ((x[0]-x_hat)**2).sum()
                    val_KL_divergence = vae.encoder.kl
                    loss_val = val_recon_loss + val_KL_divergence
                    val_loss += loss_val.cpu().numpy()
                    val_steps += 1

            validation_loss_list.append(val_loss/val_steps)

            earlystopper(val_loss/val_steps, vae)


            #wandb.log({f"{fold}-fold-loss":val_loss/val_steps, "epoch":epoch})

            if (np.isnan(val_loss/val_steps) == True) or (np.isinf(val_loss/val_steps) == True):
                break
            if earlystopper.early_stop:
                #print("early stopping")
                #print(val_loss/val_steps)
                break


        
        min_fold_validation_loss.append(min(validation_loss_list))
        #wandb.log({"train_loss":train_loss/loss_step,"train_recon_loss":recon_loss/loss_step, "train_KL-divergence":KL_divergence/loss_step,"val_loss":val_loss/val_steps,"val_recon_loss":val_recon_loss/val_steps, "val_KL-divergence":val_KL_divergence/val_steps},step=epoch)
        tune.report(loss=np.mean(min_fold_validation_loss),std=np.std(min_fold_validation_loss))
        if (np.isnan(min_fold_validation_loss).any() == True) or (np.isinf(min_fold_validation_loss).any() == True):
            #skip this because it wont be useful anyways
            break
            

def train(net, train_set, test_set, config, max_epochs=4000):

    max_num_epochs = max_epochs

    criterion = nn.MSELoss()
    val_criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), config["lr"])
    earlystopper = EarlyStopping(patience=200,verbose=False)
    # Run on GPU if possible
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device)
    
    #new
    best_model_loss = np.inf
    best_epoch = 0
    

    validation_loss_list = []

    
    
    train_loader = DataLoader(train_set, batch_size=config["batch_size"],shuffle=False,drop_last=False)
    val_loader = DataLoader(test_set, batch_size=len(test_set),shuffle=False,drop_last=False)

        

    for epoch in range(max_num_epochs):
        net.train()
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.view(-1,1)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        
        val_loss = 0.0
        val_steps = 0
        net.eval()
        for _, data in enumerate(val_loader, 0):  # this really needsd to be simplified, such that it only takes 1:
            with torch.no_grad():
                inputs, labels = data
                labels = labels.view(-1,1)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss_val = val_criterion(outputs, labels)
                val_loss += loss_val.cpu().numpy()
                val_steps += 1 
                
            
        validation_loss_list.append(val_loss/val_steps)
        earlystopper(val_loss/val_steps, net)
        
        if loss_val < best_model_loss:
            best_model_loss = loss_val
            best_epoch = epoch
            torch.save(net, "temp/Bestmodel.pt") # This should be saved to a temp folder

        if earlystopper.early_stop:
            print("Early stopping")
            break

        #if epoch % 10 == 0:
        #    print("Epoch: {} - epoch loss: {} - Best loss: {} - During epoch {}".format(epoch,val_loss/val_steps,min(validation_loss_list), best_epoch))
    # Do stuff: 
    #TODO: This model should be saved in a temporary folder or pickled object
    model = torch.load("temp/Bestmodel.pt")
    model.eval()
    #residual=model(x_val_tensor_scaled).detach().numpy()-Y[test_index].numpy().transpose()[0]
    #print(residual)

    # cleanup of temporary files: 
    os.remove("temp/Bestmodel.pt")

    print("Best epoch: {} - Val loss: {}".format(validation_loss_list.index(min(validation_loss_list)), min(validation_loss_list)))
    results = {"min_validation_loss_list": [min(validation_loss_list)], "model_list": [model]}
    return results
    #return min_fold_validation_loss_list, residuals, x_training_folds, y_training_folds, x_test_folds, y_test_folds

        

def train_vae(vae, train_data, val_data, config, output_filename, epochs=3000):
    
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    earlystopper = EarlyStopping(patience=400,verbose=False)
    
    vae.train()
    best_model_loss = np.inf
    opt = torch.optim.Adam(vae.parameters(),lr=config["lr"])
    for epoch in range(epochs):
        train_loss = 0
        loss_step = 0
        for x in train_dataloader:
            opt.zero_grad()
            x_hat = vae(x)
            recon_loss = ((x - x_hat)**2).sum()
            KL_divergence = vae.encoder.kl
            loss = recon_loss + KL_divergence
            loss.backward()
            opt.step()
            train_loss += loss
            loss_step += 1



        val_loss = 0
        val_steps = 0
        vae.eval()
        for x in val_dataloader:
            with torch.no_grad():
                x_hat = vae(x)
                val_recon_loss = ((x-x_hat)**2).sum()
                val_KL_divergence = vae.encoder.kl
                loss_val = val_recon_loss + val_KL_divergence
                val_loss += loss_val.cpu().numpy()
                val_steps += 1
        
        earlystopper(val_loss/val_steps, vae)


        if val_loss < best_model_loss:
            best_model_loss = val_loss
            best_epoch = epoch
            torch.save(vae,f"trained_models/{output_filename}.pt")

        
        if earlystopper.early_stop:
            print("Early stopping")
            break
        

        wandb.log({"train_loss":train_loss/loss_step,"train_recon_loss":recon_loss/loss_step, "train_KL-divergence":KL_divergence/loss_step,"val_loss":val_loss/val_steps,"val_recon_loss":val_recon_loss/val_steps, "val_KL-divergence":val_KL_divergence/val_steps},step=epoch)

    print("Best epoch:", best_epoch)

    return vae, best_epoch, best_model_loss


    