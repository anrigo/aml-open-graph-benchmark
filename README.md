# Installation
Recreate the environment with conda:
```
conda env create -f aml-env.yml
conda activate aml
```
If you can't, simply install python 3.10, wandb, pandas, tabulate, PyTorch 1.13, PyTorch Geometric 2.2.0 and Open Graph Benchmark 1.3.5.

In this example I'm using CUDA 11.6:
```
conda install 'python<3.11'
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 pyg ogb -c pytorch -c nvidia -c pyg
conda install wandb pandas tabulate
```
# Training and Evaluation
All the following commands will train the default model, the best one (see report). To change model replace the `model` variable in the functions you run.

You can also use the `--dw` option to disable Wandb logging.

For trainig choose a run name and run:
```
python train.py --run run_name
```

For evaluation, delete the checkpoints you don't want to test from `run_name/checkpoints`, then run:
```
python evaluate.py --run run_name
```

For hyperparameter tuning run:
```
python train.py run_name --tune
```

Put the resulting configuration in the `grid_search` function in the training script, then run the OGB evaluation:
```
python train.py run_name --ogb
```
