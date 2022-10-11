# How to use this project





## main.py
main.py uses the dataset "dataset.csv" and trains a Variational Auto Encoder (VAE). main.py logs training of the VAE to Weights and Biases (wandb)
inputs. dataset.csv 
output: trained_models/<best_trained_model>.pt

> python main.py -o <OUTPUT_NAME (str)> -d <LATENT_DIMS (int)>

## ray_tune_search.py
Performs a hyper parameter search using the ASHA algorithm.
inputs: dataset.csv, config of search parameters
outputs: loss value, std value, to C:/data/ <---- remember to set this value!
> python ray_tune_search.py



## vae_loop.py
This function does the entire experiment.
Using optimal VAE-NN and MCE MLP-NN hyper parameters 
it does 10 fold cross validation of the loop: 

1. Train VAE
2. Generate data using VAE
3. Train MCE on dataset containing VAE generated data + real data
4. Evaluate performance using test fold

> python vae_loop.py 

## test.py (deprecated)
Unused code

## Generate data (deprecated)
Unused code