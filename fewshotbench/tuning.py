import os
import optuna
from hydra import compose, initialize
from hydra.utils import instantiate
import wandb
import torch
import time
from omegaconf import OmegaConf
from utils.io_utils import get_resume_file, hydra_setup, fix_seed, model_to_dict, opt_to_dict, get_model_file
from run import train, test
from prettytable import PrettyTable
import pickle

def tune(dataset="swissprot_no_backbone"):

    with initialize(config_path="conf/dataset", version_base=None):
        cfg = compose(config_name=dataset)
        
    train_dataset = instantiate(cfg.dataset.set_cls, mode='train')
    val_dataset = instantiate(cfg.dataset.set_cls, mode='val')

    train_loader = train_dataset.get_data_loader()
    val_loader = val_dataset.get_data_loader()

    results = []

    def objective(trial):
        with initialize(config_path="conf"):
            # Compose the configuration with trial-specific overrides
            cfg = compose(config_name="main", overrides=[
                "model=protoformer",
                "method=protoformer",
                "method.stop_epoch=100",
                f"dataset={dataset}",  # Fixed dataset
                f"lr={trial.suggest_float('lr', 1e-7, 1e-4)}",
                f"weight_decay={trial.suggest_float('weight_decay', 1e-5, 1e-3)}",
                f"method.cls.n_sub_support={trial.suggest_int('n_sub_support', 2, 4)}",
                f"method.cls.n_layer={trial.suggest_int('n_layer', 1, 3)}",
                f"method.cls.n_head={trial.suggest_categorical('n_head', [1, 2, 4, 5, 8])}",
                f"method.cls.contrastive_coef={trial.suggest_float('contrastive_coef', 0.1, 2.0)}",
                f"method.cls.dropout={trial.suggest_float('dropout', 0.0, 0.3)}",
                f"method.cls.norm_first={trial.suggest_categorical('norm_first', [True, False])}",
                f"method.cls.contrastive_loss=original", #TODO: you can modify this
                f"exp.name=optuna_trial_{trial.number}",
            ])

            fix_seed(cfg.exp.seed)

            print(OmegaConf.to_yaml(cfg))

            # Initialize model and backbone for this trial
            backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim)
            model = instantiate(cfg.method.cls, backbone=backbone)

            if torch.cuda.is_available():
                model = model.cuda()

            model = train(train_loader, val_loader, model, cfg)

            acc_mean, acc_std = test(cfg, model, split='val')

            results.append([trial.number, acc_mean, acc_std])

            # Log results to WandB
            table = wandb.Table(data=results, columns=["trial", "acc_mean", "acc_std"])
            wandb.log({"eval_results": table})
            wandb.finish()

            return acc_mean  # or any other metric you want to optimize

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Output the optimization results
    best_trial = study.best_trial
    print(f"Best Trial: {best_trial.number}")
    print(f"Best Value: {best_trial.value}")
    print(f"Best Parameters: {best_trial.params}")

    # Save the study
    optuna_studies_file = f"{dataset}_studies.pkl"
    with open(optuna_studies_file, "wb") as f:
        pickle.dump(study, f)

    # Display results in a pretty table
    display_table = PrettyTable(["trial", "acc_mean", "acc_std"])
    for row in results:
        display_table.add_row(row)
    print(display_table)


tune()