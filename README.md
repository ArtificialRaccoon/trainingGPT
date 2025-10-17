
# Training a simple GPT

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib
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

### eval_interval

This defines how many training iterations you should wait before sampling the results to determine the models performance.  A value of 250 means that every 250 training steps, the model will run an evaluation on itself.

#### Why do we need evaluation?

When your model is training, it's making predictions and comparing those predictions to the source/training data.  This comparison is measured by a loss function; a numerical score of how wrong the model currently is.

* If the model's predictions are close to the source, then you have a small loss
* If the model's predictions are far off from the source, then you have a greater loss

For a text model such as ours, the goal is to predict the next token in a sequence.  For example, since we are using a dataset based on Shakespeare the model may provide a sequence such as "To be or not to " and attempt to predict the next tokens.  If the loss is low, it might predict "be, the question is", whereas if the loss is quite high it might predict "see the forest through the trees".  

By evaluating the loss, we can observe our training and ensure it is getting better, rather than getting worse, over time.  

So far we have just discussed the training loss, however while training the model you should pay special attention to the validation loss as well.  Validation loss is a measure of how well the model is building new rules, rather than duplicating the data. If your training loss continues to go down, but your validation begins to rise, you've begun to regurgitate the same data - what is referred to as overfitting.

### eval_iters

This defines how many iterations to run during an evaluation pass.  Every so often during training, the model pauses to evaluate how well it's doing.  When performing it's evaluation, it doesn't use the entire dataset as it would be too expensive; instead it uses a subset of the trained data.  The larger the value you set for eval_iters, the larger the number of samples will be taken, and the more accurate it's estimation will be at the cost of a slower evaluation.

### log_interval

This simply controls how often the training will log to the console.  If set to a value of 1, a log will be written for each training iteration; if a value of 10 is used, it will only log every 10 iterations.

### gradient_accumulation_steps

When training the model, we are trying to reduce the loss in an effort to improve the predictions it is making.  The training process works by adjusting the weights assigned to the known tokens, and hoping that the loss is reduced.  This is achieved through the observation of the gradient; a vector/plotting of partial derivitives.  This plotting points towards the direction of the steepest change in the function; the sign of the gradient tells you the direction of change, where as the size of the value tells you the rate at which it changed.

If the gradient is positive, increasing the weight increases loss, and is considered a worse result.
If the gradient is negative, increasing the weight descreases loss, and is considered an improvement.
If the gradient is unchanged, it simply means changing the weight didn't affect the overall loss.

For more information on the mathematics behind the gradient, please check out this video from Google (https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent).

When training a model you would take a batch of data and compute the loss and gradients.  For example, you might select a batch of 256 chunks of data and perform your loss measurements.  The larger your batch, generally the more stable your training is.  A large batch size comes with a major caveat however - memory consumption.  The larger the batch, the more memory will be necessary.

If you would like to simulate a larger batch size without having the memory necessary, you can do so through gradient accumulation.  Instead of updating the model after a batch of 256 chunks, you instead calculate the gradient in several smaller batches and update the model only after those smaller batches complete.  For example, if you would like to simulate a batch size of 256, you could set your batch size to 8, and your gradient accumulation to 32 (8 x 32 = 256).  This would perform 32 separate batches of 8 samples each before updating the weights, simulating a batch size of 256 samples.

Very helpful when you're running up against the limitations of memory.  Using the gradient accumulation comes with a caveat however; the larger the accumulation the longer the training process will take.  For best results, you should try to balance the steps against the amount of memory you have available.

### batch_size

As discussed above, the batch size is the number of samples used to calculate the loss and gradient.  A value of 256 means that we are taking 256 samples against the model to process our calculations.  

### block_size

Block size is the length of context (tokens) per sample.  The larger the block size, the more input the model can consider when it predicts its next token.  

#### What is a token?

A token is simply a small chunk of text.  For character level models, this would be a single character; whereas in word level models a token is a single word.  

#### How are tokens used in training?

Lets assume that the block size is just 6 characters long.  We provide a set of tokens such as "Helena", and expect to get a sequence reminiscent of "All's Well That Ends Well", or of "A Midsummer Night’s Dream".  With a longer context, we could more accurately generate text which matches the intended characters personality.  Doubling the block size to 12, and providing "Helena: Use me " should result in text which more closely aligns to the lovesick Helena, rather than the strong willed Helena.

#### Why does the block size matter?

In short, the longer the block size the more context your model will have for learning.  When dealing with natural language processing, this is akin to a longer memory.  The model can build larger relationships between tokens, and effectively can "remember" what best fits as it continues.  Increasing the block size comes with a trade off however - memory consumption and processing time.

Simply doubling the block size can quadruple the memory consumption and training time (or worse).  Depending on the dataset you are training on, you may wish to reduce or increase the block size as apropriate.  For example, if working on paragraphs you may wish to have a block size of at least 1024, whereas a batch size of 128 for learning taxonomy/naming might be more apropriate.

