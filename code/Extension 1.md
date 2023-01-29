# Extension 1

### Import/Preprocess

We import packages necessary for modeling. We then load our data. We tokenize and pad our sequences.

### Glove Embeddings

We download Glove embeddings and create mapping. We then get embedding matrix from data.

### Modeling

We define embedding layer for our network. We then define train and evaluate functions. We then define LSTM network for classification. Finally, we implement training loop where we repeatedly call the LSTM model with a different output size for each modelling node.

### Training Loop

We train our model and assess performance. This section was used to experiment with hyper-parameter tuning. We also generate predictions for test dataset in our final iteration.

### Output from Test Dataset

We assess performance on test set and output our predictions. We also use our evaluation script to assess performance.

### Error Analysis

We examine our model's errors to determine categories on which it performs poorly.

### Execution

To execute this notebook, run each cell in order.