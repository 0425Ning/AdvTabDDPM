# AdvTabDDPM

This project provides a framework for training and evaluating traffic classification models using CNN/Transformer architectures and adversarial example generation.

---

## Environment Setup

### Recommended Versions
- **Python**: 3.9.18  
- **PyTorch**: 1.10.0  
- **CUDA Toolkit**: 10.2  

### Install Dependencies
All required Python packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Usage

The main script accepts various arguments to configure training, and evaluation.

---

### Example Commands

#### AdvTabDDPM Training
```bash
python AdvTabDDPM.py --config exp/colosseum/ddpm_tune_best/config.toml --train --ds_path data/colosseum/pickle_to_csv --ds_file colosseum_dataset_globalnorm.pkl --norm_param_file colosseum_global_cols_maxmin.pkl --isNorm --log_base_dir ./classifier_train_result/ --exp_name slice_model_train_result --transformer v1 --slicelen 1 --mode test
```

#### AdvTabDDPM Evaluation
```bash
python AdvTabDDPM.py --config exp/colosseum/ddpm_tune_best/config.toml --eval --ds_path data/colosseum/pickle_to_csv --ds_file colosseum_dataset_globalnorm.pkl --norm_param_file colosseum_global_cols_maxmin.pkl --isNorm --log_base_dir ./classifier_train_result/ --exp_name slice_model_train_result --transformer v1 --slicelen 1 --mode test
```

#### Classifier Training
```bash
python AdvTabDDPM.py --config exp/colosseum/ddpm_tune_best/config.toml --ds_path data/colosseum/pickle_to_csv --ds_file colosseum_dataset_globalnorm.pkl --norm_param_file colosseum_global_cols_maxmin.pkl --isNorm --log_base_dir ./classifier_train_result/ --exp_name slice_model_train_result --transformer v1 --slicelen 1
```

#### Classifier Training
```bash
python AdvTabDDPM.py --config exp/colosseum/ddpm_tune_best/config.toml --ds_path data/colosseum/pickle_to_csv --ds_file colosseum_dataset_globalnorm.pkl --norm_param_file colosseum_global_cols_maxmin.pkl --isNorm --log_base_dir ./classifier_train_result/ --exp_name slice_model_train_result --transformer v1 --slicelen 1 --mode test
```

## Command Line Arguments

This document lists and explains all available command line arguments for the project.

---

## Arguments Table

| Argument | Description |
| --- | --- |
| `--config FILE` | Path to config file (YAML/JSON). |
| `--train` | Enable training mode. |
| `--eval` | Run evaluation. |
| `--change_val` | Change validation set. |
| `--tune` | Enable hyperparameter tuning. |
| `--n_trials` | Number of trials for tuning (default: 50). |
| `--eval_type` | Evaluation metric (`acc` or `f1`, default: `synthetic`). |
| `--eval_model` | Model type for evaluation (`catboost`, etc.). |
| `--train_classifier` | Train TRACTOR classifier before pipeline. |
| `--ds_path` | Path where dataset files are stored. |
| `--ds_file` | Name(s) of dataset pickle file(s). |
| `--norm_param_file` | Path to normalization parameters file. |
| `--isNorm` | Use normalized dataset. |
| `--mode` | Run mode: `train` or `test`. |
| `--log_base_dir` | Base directory for saving logs and models. |
| `--exp_name` | Experiment name for result saving. |
| `--transformer` | Use Transformer-based model (`v1`, `v2`, `ViT`). |
| `--dropout` | Dropout rate (default: 0.25, for ViT). |
| `--pos_enc` | Use positional encoding (Transformer only). |
| `--patience` | Early stopping patience (default: 30). |
| `--lrmax` | Initial learning rate (default: 1e-3). |
| `--lrmin` | Final learning rate after scheduling (default: 1e-5). |
| `--lrpatience` | Patience before decreasing LR (default: 10). |
| `--use-gpu` | (Deprecated) Enable GPU training. |
| `--slicelen` | Slice length for dataset generation. |

---

### Notes
- `--config` should point to a valid YAML or JSON config file.  
- `--train`, and`--eval` are mutually exclusive main modes.  
- `--transformer` supports `v1`, `v2`, or `ViT`.  
- Some arguments (like `--use-gpu`) are deprecated.  

