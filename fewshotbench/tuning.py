import optuna
from hydra import compose, initialize
from hydra.utils import instantiate
import wandb
import torch
from omegaconf import OmegaConf
from utils.io_utils import fix_seed, get_model_file
from run import train
import pickle
import argparse
from math import ceil
import numpy as np
import time


def test(cfg, model, test_dataset):
    """Modified test function - Avoid loading data every time in hyperparameter tuning"""
    
    test_loader = test_dataset.get_data_loader()

    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file)['state'])
    model.eval()

    if cfg.method.eval_type == 'simple':
        acc_all = []

        num_iters = ceil(cfg.iter_num / len(test_dataset.get_data_loader()))
        cfg.iter_num = num_iters * len(test_dataset.get_data_loader())
        print("num_iters", num_iters)
        for i in range(num_iters):
            acc_mean, acc_std = model.test_loop(test_loader, return_std=True)
            acc_all.append(acc_mean)

        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

    else:
        # Don't need to iterate, as this is accounted for in num_episodes of set data-loader
        acc_mean, acc_std = model.test_loop(test_loader, return_std=True)

    with open(f'./checkpoints/{cfg.exp.name}/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = '%s-%s-%s-%s %sshot %sway' % (
            cfg.dataset.name, test_dataset.mode, cfg.model, cfg.method.name, cfg.n_shot, cfg.n_way)

        acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(cfg.iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s, Model: %s \n' % (timestamp, exp_setting, acc_str, model_file))

    return acc_mean, acc_std


def tune(dataset, embed, n_trials, stop_epoch):
    """_summary_

    Args:
        dataset (_type_): _description_
        embed (_type_): _description_
        n_trials (_type_): _description_
        stop_epoch (_type_): _description_

    Returns:
        _type_: _description_
    """
    with initialize(config_path="conf/dataset", version_base=None):
        cfg = compose(config_name=dataset, overrides=[
            "+iter_num=600",
            f"dataset.embed_dir={embed}",
        ])
        
    train_dataset = instantiate(cfg.dataset.set_cls, mode='train')
    val_dataset   = instantiate(cfg.dataset.set_cls, mode='val')

    train_loader = train_dataset.get_data_loader()
    val_loader   = val_dataset.get_data_loader()

    # Pre-load episodic datasets for evaluation
    episodic_datasets = []
    for mode in cfg.eval_split:
        episodic_datasets.append(instantiate(cfg.dataset.set_cls, n_episode=cfg.iter_num, mode=mode))

    def objective(trial):
        with initialize(config_path="conf"):
            # Compose the configuration with trial-specific overrides
            cfg = compose(config_name="main", overrides=[
                "model=protoformer",
                "method=protoformer",
                f"method.stop_epoch={stop_epoch}",
                f"dataset={dataset}", 
                f"dataset.embed_dir={embed}", 
                f"lr={trial.suggest_float('lr', 1e-7, 1e-4)}",
                f"weight_decay={trial.suggest_float('weight_decay', 1e-5, 1e-3)}",
                f"method.cls.n_sub_support={trial.suggest_int('n_sub_support', 2, 4)}",
                f"method.cls.n_layer={trial.suggest_int('n_layer', 1, 3)}",
                f"method.cls.n_head={trial.suggest_categorical('n_head', [1, 2, 4, 5, 8])}",
                f"method.cls.contrastive_coef={trial.suggest_float('contrastive_coef', 0.1, 2.0)}",
                f"method.cls.dropout={trial.suggest_float('dropout', 0.0, 0.3)}",
                f"method.cls.norm_first={trial.suggest_categorical('norm_first', [True, False])}",
                f"method.cls.contrastive_loss={trial.suggest_categorical('contrastive_loss', ['original', 'info_nce'])}",
                f"method.cls.ffn_dim={trial.suggest_int('ffn_dim', 512, 2048)}", 
                f"exp.name={dataset}_trial_{trial.number}",
            ])

            fix_seed(cfg.exp.seed)
            results = []

            print(OmegaConf.to_yaml(cfg))

            # Initialise model and backbone for this trial
            backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim)
            model = instantiate(cfg.method.cls, backbone=backbone)

            if torch.cuda.is_available():
                model = model.cuda()

            model = train(train_loader, val_loader, model, cfg)

            # Tuning Hyper-parameter only log result on validation set
            for d in episodic_datasets:
                acc_mean, acc_std = test(cfg, model, test_dataset=d)
                results.append([trial.number, d.mode, acc_mean, acc_std])

            # Log results to WandB
            table = wandb.Table(data=results, columns=["trial", "split", "acc_mean", "acc_std"])
            wandb.log({"eval_results": table})
            wandb.finish()

            return results[-2][-2] # validation accuracy

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Output the optimization results
    best_trial = study.best_trial
    print(f"Best Trial: {best_trial.number}")
    print(f"Best Value: {best_trial.value}")
    print(f"Best Parameters: {best_trial.params}")

    # Save the study
    optuna_studies_file = f"{dataset}_studies.pkl"
    with open(optuna_studies_file, "wb") as f:
        pickle.dump(study, f)

if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for a model.')
    parser.add_argument('--dataset', type=str, help='Dataset to use for training')
    parser.add_argument('--embed', type=str, default='embeds', help='Embedding directory')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials for hyper-parameter tuning')
    parser.add_argument('--stop_epoch', type=int, default=60, help='Number of epochs to stop training')

    # Parse the arguments
    args = parser.parse_args()

    tune(
        args.dataset, 
        args.embed, 
        args.n_trials, 
        args.stop_epoch
    )

    tune()
    wandb.finish()