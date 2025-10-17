
# Training a simple GPT

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## quick start

```sh
python data/data-by-char/prepare.py
```

```sh
python train.py config/train_data_by_char.py
```

```sh
python sample.py --out_dir=out-data-by-char
```

## Parameter Explainations

### eval_iters

This defines how many iterations to run during an evaluation pass.  Every so often during training, the model pauses to evaluate how well it's doing.  When performing it's evaluation, it doesn't use the entire dataset as it would be too expensive; instead it uses a subset defined by this value.  The larger the value you set for eval_iters, the more accurate it's estimation will be, but the slower the evaluation will be.

### log_interval

