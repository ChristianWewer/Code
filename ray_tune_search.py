from functools import partial
from multiprocessing import reduction
from sched import scheduler
from secrets import choice
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split, KFold
from utility.data_io import load_data_no_y
from utility.train_network import tune_train_vae
import time


def optimize_parameters(train_data, num_samples=1):
    config = {
        "layers": tune.grid_search([[512, 256, 128],[256, 128, 64], [128, 64, 32], [64, 32, 16], [32, 16, 8]]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8,16,32,64]),
        "latent_dims": 4
    }

    scheduler = ASHAScheduler(
        metric = "loss",
        mode ="min",
        time_attr="training_iteration",
        grace_period=3,
        reduction_factor=3,
        brackets=1
    )

    reporter = CLIReporter(
        metric_columns=["training_iteration","loss","std"]
        )


    tune.run(
        partial(tune_train_vae, train_data=train_data, kfolds=10),
        resources_per_trial={"cpu": 2, "gpu": 0},
        local_dir="C:\\data\\VAE",
        name=f"3hl_4d",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )


if __name__ == "__main__":
    num_samples = 100
    dataset_name = "dataset"

    train_data = load_data_no_y(dataset_name, 0, True)

    optimize_parameters(train_data, num_samples)

