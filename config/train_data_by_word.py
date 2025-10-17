# Train a miniature character-level Shakespeare model

out_dir = 'out-data-by-char'
eval_interval = 250     # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10       # don't print too too often

always_save_checkpoint = False

wandb_log = False       # override via command line if you like
wandb_project = 'data-by-char'
wandb_run_name = 'mini-gpt'

dataset = 'data-by-char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64        # context of up to 256 previous characters

n_layer = 4
n_head = 4
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000   # make equal to max_iters
min_lr = 1e-4           # learning_rate / 10 usually
beta2 = 0.99            # make a bit bigger because number of tokens per iter is small

warmup_iters = 100      # not super necessary potentially
#device = 'cpu'          # run on cpu only
compile = False         # do not torch compile the model; requires extra setup on Windows