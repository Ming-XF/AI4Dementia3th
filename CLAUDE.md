# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

EEG-based Alzheimer's disease diagnosis using deep learning. The codebase implements ~22 models spanning graph neural networks, CNNs, Transformers, and VAE-based architectures for classifying EEG connectivity/ time-series data. Currently runs binary (AD vs Normal) and 4-class classification experiments on Dementia400, Dementia2000, Dementia4000, C42B, and Beirut datasets.

## Common commands

### Train a single model
```bash
python main.py \
    --model "EEGNet" \
    --num_repeat 3 \
    --dataset 'Dementia400' \
    --data_dir "../data/Dementia400/Dementia400.npy" \
    --batch_size 16 \
    --num_epochs 200 \
    --frequency 128 --D 22 --num_kernels 16 --p1 4 --p2 8 --dropout 0.5 \
    --drop_last False \
    --schedule "cos" \
    --learning_rate 1e-3 \
    --do_train --do_evaluate --do_test
```

### Batch-train all models for a dataset
```bash
bash train_Dementia400.sh   # runs all scripts/dementia400/*.sh sequentially
```

### Run a specific model's ablation script
```bash
bash ablation_channel.sh    # loops over num_heads for BrainVAE on Dementia dataset
```

### Analyze results
```bash
python show.py --type dementia400 --path ./log_dir     # generates HTML comparison table + PNG
python threeShow.py --path ./log_dir/train_EEGNet_Dementia400.log --type 1   # single model 3-repeat metrics plot
```

### Environment
- Python with PyTorch 2.1.0 (CUDA), braindecode, mne, nilearn, einops, scikit-learn, wandb
- Install: `pip install -r requirements.txt`

## Architecture

### Entry point and flow

`main.py` is the sole entry point. It calls `init_config()` (argparse) → dispatches to `cross_subject()` or `within_subject()` based on `--within_subject`. Each repeat creates a `Trainer` subclass via `eval(args.model + 'Trainer')`, then calls `trainer.train()`.

### Key files and their roles

- **`config.py`** — Two responsibilities: (1) `init_config()` defines all CLI arguments via argparse; (2) `init_model_config(args, data_config)` maps `args.model` string to model config + instantiation via `eval()`. Every new model must be registered here.
- **`trainers.py`** — One `*Trainer` class per model. Each overrides `prepare_inputs_kwargs()` to shape model inputs (time_series, correlation matrices, labels). Most inherit from `DFaSTTrainer`, `BNTTrainer`, `SrCVIBTrainer`, or the base `Trainer`.
- **`utils/trainer.py`** — Base `Trainer` class. Handles dataset loading, optimizer/scheduler init via `init_components()`, the standard `train()` loop (epoch → `train_epoch()` → `evaluate()` → save best), and `binary_evaluate()`/`multiple_evaluate()` which compute Accuracy, AUC, Precision, Recall, F-score, Specificity, Sensitivity.
- **`data/dataset.py`** — `BaseDataset` with StratifiedKFold splitting, connectivity/correlation computation, and normalization utilities.
- **`data/dataloader.py`** — `init_StratifiedKFold_dataloader()` (single GPU) and `init_distributed_dataloader()` (multi-GPU).
- **`data/data_config.py`** — `DataConfig` dataclass holding all data-related hyperparameters.
- **`model/base/`** — `BaseModel(nn.Module)` with abstract `forward()` and `ModelOutputs` container (logits, loss, hidden_state). `BaseConfig` holds common model hyperparams.

### Model taxonomy

| Paradigm | Models |
|---|---|
| **Graph-based** (use correlation matrices as node features) | BNT, BrainNetCNN, FBNetGen, ALTER, GCDGCN, DFaST, DFaSTOnlySpatial, STAGIN |
| **CNN-based** (raw time-series input, freq-domain features) | EEGNet, DeepConvNet, ShallowConvNet, EEGChannelNet, CEEDNet, RACNN, LMDA, SteadyNet |
| **VAE / Information Bottleneck** (feature compression for low-SNR EEG) | SrCVIB, VIB, CVIB4LMDA |
| **Transformer** | Transformer, BrainNetworkTransformer |
| **Hybrid / Other** | TCANet (local+global), TCACNet (wavelet + CNN), SBLEST, AlzNetV3 |

### Trainer inheritance

- **`DFaSTTrainer`** (time_series → labels): DFaST, DFaSTOnlySpatial, LMDA, ShallowConvNet, DeepConvNet, EEGNet, EEGChannelNet, RACNN, TCANet, SteadyNet, CEEDNet
- **`BNTTrainer`** (correlation → labels): BNT, BrainNetCNN, Transformer, ALTER (adds time_series)
- **`SrCVIBTrainer`** (time_series + correlation, class-conditional VIB with z-statistics): SrCVIB, VIB, CVIB4LMDA
- **`STAGINTrainer`**: Custom `train_epoch()` with dynamic FC construction and gradient clipping
- **`GCDGCNTrainer`**: Two-stage training (pretrain < epoch 100, finetune after)
- **`TCACNetTrainer`**: Wavelet packet energy features, dual-loss (local+global)
- **`SBLESTTrainer`**: Non-neural — computes W projection via batch matrix ops
- **`FBNetGenTrainer`**: Uses both time_series and correlation, crops to window_size multiple

### Dataset classes

Each dataset (`Dementia400Dataset`, `C42BDataset`, `BeirutDataset`, etc.) extends `BaseDataset` and implements `load_data()` and `__getitem__()`. They load `.npy` files containing `timeseries`, `corr`, `labels`, `subject_id` keys. The `data/` directory also contains preprocessing scripts (e.g., `dementia_preprocess()` in `dementia400.py`) that convert raw `.mat` EEG files to `.npy`.

### Adding a new model

1. Create `model/NewModel/NewModel.py` with `NewModelConfig(BaseConfig)` and `NewModel(BaseModel)`
2. Create `model/NewModel/__init__.py` exporting the config and model classes
3. Add `from .NewModel import *` to `model/__init__.py`
4. Add `elif args.model == "NewModel":` branch in `config.py:init_model_config()`
5. Add `NewModelTrainer(Trainer)` in `trainers.py` (override `prepare_inputs_kwargs` at minimum)
6. Add training script: `scripts/<dataset>/train_NewModel_<Dataset>.sh`
7. Reference: `python main.py --model "NewModel" ...`

### Output structure

- **`log_dir/`** — Training logs (one per model-dataset combo). Log format uses `########## Repeat:N` delimiters between repeats, with per-epoch metrics on separate lines.
- **`output_dir/<ModelName>/`** — Saved model checkpoints (`<Model>-<task_id>.bin`) and config JSON.
- **`analysis/`** — Generated HTML comparison tables and PNG metric plots from `show.py` / `threeShow.py`.

### Key conventions

- Model instantiation and trainer selection use `eval()` based on string names — the `--model` argument must exactly match the class name prefix (e.g., `"EEGNet"` → `EEGNetTrainer`, `EEGNetConfig`, `EEGNet`).
- All models return a `ModelOutputs` object with `.logits` and `.loss` attributes.
- Binary classification uses `F.one_hot` labels (shape `[B, 2]`) with `labels[:, 1]` as the positive class.
- The `--num_repeat` flag controls StratifiedKFold splits (default 5); scripts typically use 3.
- wandb logging is present in the code but commented out — enable by uncommenting `wandb.init()` and `wandb.log()` calls.
- Seed is fixed at 42 via `set_seed()` in `main.py`.
