# CIS 5300 - Milestone #3 (Extension 1)

### Import/Preprocess

We import packages necessary for modeling. We then load our data. We tokenize and pad our sequences.

### Glove Embeddings

We download Glove embeddings and create mapping. We then get embedding matrix from data.

### Modeling

We define embedding layer for our network. We then define train and evaluate functions. We then define LSTM network for classification. Finally, we implement training loop where we repeatedly call the LSTM model with a different output size for each modelling node.

### Execution

To execute this notebook, run each cell in order.