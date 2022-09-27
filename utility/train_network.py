import torch
import numpy as np
import wandb
from ray import tune
from networks.VAE import VAE_3hl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
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
            


        

def train_vae(vae, train_data, val_data, config, output_filename, epochs=3000):
    vae.train()
    best_model_loss = np.inf
    opt = torch.optim.Adam(vae.parameters(),lr=config["lr"])
    for epoch in range(epochs):
        train_loss = 0
        loss_step = 0
        for x in train_data:
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
        for x in val_data:
            with torch.no_grad():
                x_hat = vae(x[0])
                val_recon_loss = ((x[0]-x_hat)**2).sum()
                val_KL_divergence = vae.encoder.kl
                loss_val = val_recon_loss + val_KL_divergence
                val_loss += loss_val.cpu().numpy()
                val_steps += 1


        if val_loss < best_model_loss:
            best_model_loss = val_loss
            best_epoch = epoch
            torch.save(vae,f"trained_models/{output_filename}.pt")
        

        wandb.log({"train_loss":train_loss/loss_step,"train_recon_loss":recon_loss/loss_step, "train_KL-divergence":KL_divergence/loss_step,"val_loss":val_loss/val_steps,"val_recon_loss":val_recon_loss/val_steps, "val_KL-divergence":val_KL_divergence/val_steps},step=epoch)

    print("Best epoch:", best_epoch)

    return vae, best_epoch, best_model_loss


    