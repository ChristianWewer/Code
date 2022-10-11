from networks.VAE import VAE_3hl
from networks.MLP import Net
from utility.data_io import load_data
from sklearn.model_selection import KFold, train_test_split
from utility.train_network import train_vae, train
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np
import random
import json
import argparse


def vae_loop(augmented_data_sample_size, augment_data=True, random_seed=10):
    rng = np.random.default_rng()

    """ Runs the entire experiment loop of training VAE, Generating data, training MCE, testing MCE.
    Returns list of MCE performance across 10 folds.
    """
    # Optimal parameters found
    optimal_VAE_hyperparameter_config = {
        "lr":0.0003,
        "layers":[512,256,128],
        "batch_size":64,
        "latent_dims":5
        }

    optimal_MCE_hyperparameter_config = {
        "lr":0.029924,
        "l1":231,
        "l2":421,
        "l3":392,
        "batch_size":16
        }

    mse_error_list = []
    mae_error_list = []

    # load original dataset:
    dataset_name = "dataset"
    train_set_vae, _ = load_data(dataset_name, 0, include_initial_mass=True, for_vae=True) # I guess i don't really want this.
    train_set_mce, _ = load_data(dataset_name, 0, include_initial_mass=True, for_vae=False)
    n_splits = 10
    test_size = 20
    kfold = KFold(n_splits=n_splits)

    wandb.init(project="kfold_check",name="test_name",config=optimal_VAE_hyperparameter_config)    

    for fold, (train_index, test_index) in enumerate(kfold.split(train_set_vae)):
        print(f"Fold: {fold}")
        # Same datasaet, with same - vae has MC inside, mce only has MC in y
        train_data_vae = train_set_vae[train_index] # this should be split into two: 1 containing the train split and 1 containing the val split
        train_data_mce = train_set_mce[train_index]
        test_data_vae = train_set_vae[test_index] #this will be held out untill the very end. Must only be used to validate MCE performance
        test_data_mce = train_set_mce[test_index] # This is used for final testing
        # Format lists so that i can use it.
        train_x_mce = train_data_mce[0]
        train_y_mce = train_data_mce[1]

        train_x_vae = train_data_vae[0]
        train_y_vae = train_data_vae[1] # strictly not needed
        # the thing is, maybe this is all unnecesary and i can just glue the two together for the VAE training?
        x_train_mce, x_test_mce, y_train_mce, y_test_mce = train_test_split(train_x_mce, train_y_mce, test_size=test_size,random_state=random_seed) # Please verify that these result in the same splits
        x_train_vae, x_test_vae, _, _ = train_test_split(train_x_vae, train_y_vae, test_size=test_size,random_state=random_seed) # please verify that these result in the same splits
        ##################
        # Train the VAE: #
        ##################
        if augment_data:
            print("Train Variational Autoencoder")
            vae = VAE_3hl(optimal_VAE_hyperparameter_config["latent_dims"],optimal_VAE_hyperparameter_config["layers"])
            vae_model_output_name = f"{fold}-fold-best_model_vae"
            train_vae(vae,x_train_vae,x_test_vae,optimal_VAE_hyperparameter_config,vae_model_output_name,5000)
            
            #####################################
            # Generate a dataset using the VAE: #
            #####################################
            print("Generate augmented data")
            best_vae_model = torch.load(f"trained_models/{vae_model_output_name}.pt")

            generated_data_list = []
            for i in range(100):
                sample = best_vae_model.draw_sample()
                sample = sample.float()
                generated_data = best_vae_model.decoder(sample)
                generated_data = generated_data.detach().numpy()
                generated_data_list.append(generated_data)

            generated_data = np.concatenate(generated_data_list,0)
            generated_data = rng.choice(generated_data, augmented_data_sample_size)
            # Generated data for MCE (remove end_rh)
            y_generated_for_mce = [observation[6] for observation in generated_data]
            x_generated_for_mce = [np.delete(observation, 6) for observation in generated_data]

            
            ########################
            # Merge data as needed #
            ########################


            x_train_mce = torch.cat((torch.Tensor(x_generated_for_mce), x_train_mce), axis=0)
            y_train_mce = torch.cat((torch.Tensor(y_generated_for_mce), y_train_mce), axis=0)


        ############################
        # Train MCE on chosen data #
        ############################
        print("Train Moisture Content Estimator")
        #implement training of MCE
        net = Net(10, optimal_MCE_hyperparameter_config["l1"], optimal_MCE_hyperparameter_config["l2"], optimal_MCE_hyperparameter_config["l3"])
        
        train_dataset_mce = TensorDataset(x_train_mce, y_train_mce)
        test_dataset = TensorDataset(x_test_mce, y_test_mce)


        results = train(net, train_dataset_mce, test_dataset, optimal_MCE_hyperparameter_config)

        ############################################################
        # test performance using the testset called: test_data_mce #
        ############################################################
        print("Estimate performance")
        best_model = results["model_list"][0]

        best_model.eval()
        predictions = best_model(test_data_mce[0]).detach().numpy().transpose()[0]
        residuals = predictions-test_data_mce[1].numpy()
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))

        mse_error_list.append(mse)
        mae_error_list.append(mae)

        
    print(f"List of MSE: {mse_error_list}")
    print(f"Mean MSE: {np.mean(mse_error_list)}")

    print(f"List of MAE: {mae_error_list}")
    print(f"Mean MAE: {np.mean(mae_error_list)}")

    print("it all worked, huray! :D")

    return mse_error_list, mae_error_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This scripts runs the VAE-MCE-RCV-loop")
    parser.add_argument('-o','--output-name',dest="output_name", help="Output filename")
    parser.add_argument('-n','--nAugmented',dest="n_augmented_samples", help="Sets the number of augmented samples to be used in the training set")
    parser.add_argument('-ncv','--nRepeatedCV',dest="n_rcv", help="Sets the number of repeats of cross validation")
    args = parser.parse_args()



    rcv_mse_mean_error_list = []
    rcv_mae_mean_error_list = []
    rcv_mse_error_list = []
    rcv_mae_error_list = []


    augmented_data_sample_size=int(args.n_augmented_samples)

    for i in range(int(args.n_rcv)):
        mse_error_list, mae_error_list = vae_loop(augmented_data_sample_size, augment_data=True, random_seed=random.randint(1,100000))
        rcv_mse_error_list.append(mse_error_list)
        rcv_mae_error_list.append(mae_error_list)

        rcv_mse_mean_error_list.append(np.mean(mse_error_list))
        rcv_mae_mean_error_list.append(np.mean(mae_error_list))


    print(f"Mean MSE: {np.mean(rcv_mse_mean_error_list)}")
    print(f"Mean MAE: {np.mean(rcv_mae_mean_error_list)}")

    print(f"STD MSE: {np.std(rcv_mse_mean_error_list)}")
    print(f"STD MAE: {np.std(rcv_mae_mean_error_list)}")
    

    data = {
        "rcv_mse_error_list": str(rcv_mse_error_list), 
        "rcv_mae_error_list": str(rcv_mae_error_list),
        "rcv_mse_mean_error_list": str(rcv_mse_mean_error_list),
        "rcv_mae_mean_error_list": str(rcv_mae_mean_error_list)
        }

    output_filename = str(args.output_name)
    with open(f"output_filename",'w') as output_file:
        output_file.write(json.dumps(data))
