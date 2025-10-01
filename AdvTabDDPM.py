import sys
import os
# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tddpm_lib

import argparse
import shutil

import torch
import zero
from train import train
from sample import sample
from classifier_train import eval_func
import glob
import pandas as pd
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple

import matplotlib.pyplot as plt

import optuna
from pathlib import Path

def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def tune_with_optuna(base_config_path, n_trials=50, train_size=6500, args=None, do_tune=True):
    exps_path = Path(base_config_path).parent.parent / "many-exps"

    def build_trial_config(trial):
        base_config = tddpm_lib.load_config(base_config_path)
        exp_name = Path(base_config_path).parts[1]  # e.g., 'colosseum'
        base_config['exp_name'] = exp_name

        # 超參數調參空間
        lr = trial.suggest_loguniform('lr', 1e-5, 3e-3)
        d_layers = _suggest_mlp_layers(trial)
        weight_decay = 0.0
        batch_size = trial.suggest_categorical('batch_size', [256, 4096])
        steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
        num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
        num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
        gaussian_loss_type = 'mse'

        # 更新 config
        base_config['train']['main'].update({
            'lr': lr,
            'steps': steps,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        })
        base_config['model_params']['rtdl_params']['d_layers'] = d_layers
        base_config['eval']['type']['eval_type'] = args.eval_type
        base_config['sample']['num_samples'] = num_samples
        base_config['diffusion_params'].update({
            'gaussian_loss_type': gaussian_loss_type,
            'num_timesteps': num_timesteps,
            'scheduler': scheduler
        })
        base_config['eval']['type']['eval_model'] = args.eval_model
        if args.eval_model == "mlp":
            base_config['eval']['T']['normalization'] = "quantile"
            base_config['eval']['T']['cat_encoding'] = "one-hot"

        # 設定儲存路徑
        trial_dir = exps_path / f"{trial.number}"
        base_config['parent_dir'] = str(trial_dir)
        config_path = trial_dir / "config.toml"
        # 儲存當前 config 進 user_attr
        trial.set_user_attr("config", base_config)
        trial.set_user_attr("trial_dir", str(trial_dir))

        os.makedirs(trial_dir, exist_ok=True)
        tddpm_lib.dump_config(base_config, config_path)

        return base_config, trial_dir

    def objective(trial):
        config, trial_dir = build_trial_config(trial)
        run_pipeline(config, do_train=True, do_sample=False, do_eval=False, change_val=True, do_tune=do_tune)

        eval_seeds = 5
        total_score = 0.0
        for seed in range(eval_seeds):
            config['sample']['seed'] = seed
            run_pipeline(config, do_train=False, do_sample=True, do_eval=True, change_val=True, do_tune=do_tune)

            report_path = Path(config['parent_dir']) / 'results_eval.json'
            report = tddpm_lib.load_json(report_path)
            print('report', report)
            if 'f1-score' in report['metrics']['val']['macro avg']:
                total_score += report['metrics']['val']['macro avg']['f1-score']
            else:                
                total_score += report['results']['test'].get('acc') or report['results']['test'].get('f1', 0.0)

        shutil.rmtree(trial_dir)
        return total_score / eval_seeds

    # 建立 study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 保存最佳 trial 的 config 與模型
    best_trial = study.best_trial
    best_config = best_trial.user_attrs["config"]
    print('best_config: ', best_config)
    best_dir = Path("exp") / best_config['exp_name'] / "ddpm_tune_best"
    best_dir.mkdir(parents=True, exist_ok=True)

    # 儲存 config.toml
    best_config['parent_dir'] = str(best_dir)
    tddpm_lib.dump_config(best_config, best_dir / "config.toml")

    # # 儲存 model.pt
    # trial_dir = Path(best_trial.user_attrs['trial_dir'])
    # shutil.copy(trial_dir / "model.pt", best_dir / "model.pt")

    # 儲存參數重要性
    importance = optuna.importance.get_param_importances(study)
    tddpm_lib.dump_json(importance, best_dir / "importance.json")

    print(f"Best trial copied to: {best_dir}")

    # 最佳 config 再跑一次訓練與最終取樣
    run_pipeline(best_config, do_train=True, do_sample=True)

def run_pipeline(raw_config, do_train=True, do_sample=False, do_eval=False, change_val=False, do_tune=False):

    device = torch.device(raw_config.get('device', 'cuda:0'))
    save_path = os.path.join(raw_config['parent_dir'], 'result')
    os.makedirs(save_path, exist_ok=True)

    if do_train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val,
            save_path=save_path
        )
    if do_sample:
        sample(
            steps = raw_config['train']['main']['steps'],
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            save_path=save_path,
            do_tune=do_tune
        )
    if do_eval:
        print('pipeline eval')
        print('eval raw_config[parent_dir]: ', raw_config['parent_dir'])
        print('eval raw_config[real_data_path]: ', raw_config['real_data_path'])
        print('raw_config[eval][type][eval_model]: ', raw_config['eval']['type']['eval_model'])

        if not do_tune:
            print('Now is tractor')

            # logdir = 'train_log/slice__model_train_result'  # 若有固定邏輯可改成變數
            logdir = 'classifier_train_result/slice_model_train_result'
            model_paths = glob.glob(os.path.join(logdir, "model.*.pt"))
            print(f"load {model_paths}")
            model_paths.sort()
            model_name = os.path.basename(model_paths[-1])  # 最新模型
            
            if do_tune:
            # 呼叫 eval_func
                eval_func(
                    logdir=logdir,
                    model_name=model_name,
                    parent_dir=raw_config['parent_dir'],
                    device=device,
                    use_diffusion_data=False #True  # 如果你確定要測試 diffusion data
                )
            else:
                eval_func(
                    logdir=logdir,
                    model_name=model_name,
                    parent_dir=raw_config['parent_dir'],
                    device=device,
                    use_diffusion_data=True
                )
        elif raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=change_val,
            )
        elif raw_config['eval']['type']['eval_model'] == 'mlp':
            train_mlp(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=change_val,
                device=device
            )
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=change_val,
            )

    if not do_tune:
        plot_losses(
            parent_dir=save_path, 
            train_csv="train_loss.csv", 
            val_csv="validation_loss.csv", 
            test_csv="test_loss.csv"
        )
        
        # After the training loop
        plot_loss_history( # Plot the training loss history and save the plot as an image # add # for training result
            parent_dir=save_path,
            train_csv="train_loss.csv",
            plot_filename='loss_plot.png'
        )

# add # for training result
def plot_loss_history(parent_dir, train_csv, plot_filename):
    # read train.csv file
    csv_path = os.path.join(parent_dir, train_csv)
    loss_history = pd.read_csv(csv_path)

    # Plot the picture
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['step'], loss_history['train_loss'], label='Total Loss')
    plt.plot(loss_history['step'], loss_history['train_mloss'], label='Multinomial Loss')
    plt.plot(loss_history['step'], loss_history['train_gloss'], label='Gaussian Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()

    # Save the picture
    save_path = os.path.join(parent_dir, plot_filename)
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")

def plot_losses(parent_dir, train_csv, val_csv, test_csv):
    # Load each CSV file into a DataFrame
    train_df = pd.read_csv(os.path.join(parent_dir, train_csv))
    val_df = pd.read_csv(os.path.join(parent_dir, val_csv))
    test_df = pd.read_csv(os.path.join(parent_dir, test_csv))
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['step'], train_df['train_loss'], label='Train Total Loss', color='blue', linestyle='-')
    plt.plot(val_df['step'], val_df['val_loss'], label='Validation Total Loss', color='cyan', linestyle='-')
    plt.plot(test_df['step'], test_df['test_loss'], label='Test Total Loss', color='purple', linestyle='-')

    # Labeling the plot
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Losses Over Steps")
    plt.legend(loc="upper right")
    
    # Save and show the plot
    plt.savefig(os.path.join(parent_dir, 'train_val_test_losses_plot.png'))
    # plt.show()

def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

# add for npy classify
def load_npy_dataset_info(x_num_path, x_cat_path, y_path):
    X_num = np.load(x_num_path)  # shape: (N, slice_len, D_num)
    X_cat = np.load(x_cat_path)  # shape: (N, slice_len, D_cat)
    y = np.load(y_path)          # shape: (N,)

    if X_num.ndim == 2:
        # 兼容舊版資料格式 [N, D]，加一個 slice_len 維度
        X_num = X_num[:, np.newaxis, :]
    if X_cat.ndim == 2:
        X_cat = X_cat[:, np.newaxis, :]

    # 基本統計
    slice_len = X_num.shape[1]
    num_feats = X_num.shape[2] + X_cat.shape[2] - 1 # remove slice_id
    nclasses = int(np.max(y)) + 1
    numsamps = len(y)

    # 每個類別的樣本數
    samples_per_class = np.bincount(y)
    # print(f'slice_len {slice_len}, num_feats {num_feats}, nclasses {nclasses}, numsamps {numsamps}, samples_per_class {samples_per_class}')

    return {
        'numfeats': num_feats,
        'slice_len': slice_len,
        'numsamps': numsamps,
        'nclasses': nclasses,
        'samples_per_class': samples_per_class
    }

from classifier_models import ConvNN, TransformerNN, TransformerNN_v2
from classifier_models import megatron_ViT as ViT
from classifier_dataset import TracesDataset
import numpy as np
import pickle
from classifier_train import train_func
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--eval_type', type=str, default='synthetic', help='Evaluation metric: acc or f1')
    parser.add_argument('--eval_model', type=str, default='catboost', help='Model type for evaluation')
    parser.add_argument('--train_classifier', action='store_true', default=False, help="Train the TRACTOR model before pipeline")
    
    parser.add_argument("--ds_path", default="../logs/", help="Specify path where dataset files are stored")
    #--ds_path data\colosseum\pickle_to_csv
    parser.add_argument("--ds_file", nargs='+', required=True, help="Name of dataset pickle file containing training data and labels.")
    #--ds_file colosseum_dataset_globalnorm.pkl
    parser.add_argument("--norm_param_file", default="", help="Normalization parameters path.")
    #--norm_param_file colosseum_global_cols_maxmin.pkl
    parser.add_argument("--isNorm", default=False, action='store_true', help="Specify to load the normalized dataset." )
    #--isNorm
    parser.add_argument("--mode", default="train", choices=['train', 'test'], help="Testing the classifier model") # TODO visualize capture and then perform classification after loading model
    #--mode test
    parser.add_argument("--log_base_dir", default=".\classifier_train_result", help="Specify base directory for saving logs and models.") # add
    #--log_base_dir .\classifier_train_result\
    parser.add_argument("--exp_name", default='', help="Name of this experiment")
    #--exp_name slice_model_train_result
    parser.add_argument("--transformer", default=None, choices=['v1', 'v2', 'ViT'], help="Use Transformer based model instead of CNN, choose v1 or v2 ([CLS] token)")
    #--transformer v1
    parser.add_argument("--dropout", default=0.25, type=float, help="Only used for Visual Transformer (ViT)")
    parser.add_argument("--pos_enc",  action="store_true", default=False, help="Use positional encoder (only applied to transformer arch)")
    parser.add_argument("--patience", type=int, default=30, help="Num of epochs to wait before interrupting training with early stopping")
    parser.add_argument("--lrmax", type=float, default=1e-3,help="Initial learning rate ")
    parser.add_argument("--lrmin", type=float, default=1e-5, help="Final learning rate after scheduling ")
    parser.add_argument("--lrpatience", type=int, default=10, help="Patience before triggering learning rate decrease")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="[Deprecated] Enables GPU training")
    parser.add_argument("--slicelen", default=1, type=int, help="Specify the slices lengths while generating the dataset.")
    #--slicelen 1

    # Parse arguments (parser) and ignore unknown arguments
    args, _ = parser.parse_known_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available! Using {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")

    # ===== Select model =====
    # Select the version of the Transformer model
    transformer = {'v1': TransformerNN, 'v2': TransformerNN_v2, 'ViT': ViT}.get(args.transformer, None)

    # Load the training dataset
    print("--- Loading Train dataset...")
    ds_train = TracesDataset(dataset_pkls=args.ds_file, key='train', path=args.ds_path, sanitize=False, norm_par_path=args.norm_param_file)
    # Load the validation dataset
    print("--- Loading Validation dataset...")
    ds_test  = TracesDataset(dataset_pkls=args.ds_file, key='valid', path=args.ds_path, sanitize=False, norm_par_path=args.norm_param_file)
    
    print(np.bincount([lbl.item() for lbl in ds_train.obs_labels]))

    # Get basic information about the dataset
    ds_info = ds_train.info()

    print("--- DS INFO ---")
    print(ds_info)

    # ===== Set training parameters =====
    # Create a folder to store models and training records
    logdir = os.path.join(args.log_base_dir, args.exp_name) # modify # logdir = os.path.join(args.cp_path, args.exp_name)
    os.makedirs(logdir, exist_ok=True)

    # Load normalize parameters
    normp = pickle.load(open(os.path.join(args.ds_path, args.norm_param_file), "rb"))

    train_config = {
        # Set learning rate related parameters
        "lr": 1e-3,
        "lrmin": 1e-5,
        "lrpatience": 10,
        "batch_size": 512,
        "epochs": 350,
        # Set training related parameters
        "Nclass": ds_info['nclasses'],
        "slice_len": ds_info['slice_len'],
        "num_feats": ds_info['numfeats'],
        "logdir": logdir,
        "pos_enc": False,
        "patience": 30,
        "dropout": 0.25,
        # Set other parameters
        "global_model": ConvNN if args.transformer is None else transformer,
        "model_postfix": f"trans_{args.transformer}" if args.transformer else "cnn",
        "device": device
    }

    # If it is not test mode, start training
    if args.mode == 'train':

        # add for npy classify
        # ds_info = load_npy_dataset_info(
        #     x_num_path='exp/colosseum/ddpm_tune_best/X_num_train.npy',
        #     x_cat_path='exp/colosseum/ddpm_tune_best/X_cat_train.npy',
        #     y_path='exp/colosseum/ddpm_tune_best/y_train.npy'
        # )
        ds_info = load_npy_dataset_info(
            x_num_path='data/colosseum/combine/X_num_train.npy',
            x_cat_path='data/colosseum/combine/X_cat_train.npy',
            y_path='data/colosseum/combine/y_train.npy'
        )
        print("[INFO] Reconstructed ds_info from NPY:")
        print(ds_info)
        train_config.update({
            "Nclass": ds_info['nclasses'],
            "slice_len": ds_info['slice_len'],
            "num_feats": ds_info['numfeats'],
        })

        # Single machine training
        loss_result, model_name = train_func(train_config, ds_train, ds_test)

        import json
        from json_tricks import dump
        from classifier_train import prepare_train_config_for_saving
        train_config_to_save = prepare_train_config_for_saving(train_config)
        # Save config and dataset info
        with open(os.path.join(logdir, "train_config.json"), "w") as f:
            json.dump(train_config_to_save, f)
        with open(os.path.join(logdir, "ds_info.json"), "w") as f:
            dump(ds_info, f, allow_nan=True)
        torch.save(ds_test, os.path.join(logdir, "test_dataset.pt"))

    raw_config = tddpm_lib.load_config(args.config)
    
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.tune:
        tune_with_optuna(args.config, n_trials=args.n_trials, args=args)
    else:
        run_pipeline(raw_config, do_train=args.train, do_sample=args.sample, do_eval=args.eval, change_val=args.change_val)

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()