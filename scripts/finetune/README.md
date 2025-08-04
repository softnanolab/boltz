# Introduction
This folder contains all the code for finetuning Boltz.

## 1 Basic Stats
- Loading of the datamodule:
    ```python
    data_module = BoltzTrainingDataModule(data_config)
    ```
    - this takes ~40 seconds.
- Loading of the model:
    ```python
    cfg = hydra.utils.instantiate(raw_cfg) # instantiates the model 
    ```
    - this takes ~30 seconds.

## 2 Commands to Run `train.py`

If you don't want to use the default config, you can override it by passing the config path and name.
```bash
python scripts/finetune/train.py \
 --config-path=/home/path/to/experiment_dir \  # overrides the default config path
 --config-name=structure.yaml \
 trainer.precision=16  # overrides the precision in the yaml file
```