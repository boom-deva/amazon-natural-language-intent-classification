# Simple Baseline ReadMe

## 1. Description of Simple Baseline 

In this `simple-baseline.py` script, we attempt to construct two such simple baselines: the majority class baseline and a KNN classifier baseline. 

The majority class baseline simply predicts the label that shows up most frequently in the training set for all examples. In the training set, the majority class is `label = 50` corresponding to `label_text = calendar_set`, which makes up a mere 7.035% of the dataset. Using this majority prediction, we have a training accuracy of 7.035% and testing accuracy of **7.028%**, which are incredibly low but consistent hence underfit. This would be far too trivial to beat.

As a result, we tried to explore a better but still naive baseline by running a Nearest Neighbors Classifier with $k = 1$ that uses Glove embeddings to compute the embedding of each word in an example text, then averages them to get a sentence level embedding for the example. We do this for all examples in both train and test sets. Then, for each testing example, we compute the cosine similarity of its sentence level embedding with the embedding of each of the training examples. The label of this testing example is then taken to be the same label as the training example with the highest cosine similarity. This baseline produced a testing accuracy of **74.65%**, which is somewhat more realistic as a simple baseline to beat.

## 2. Sample Output

### Majority Baseline Predictions
`majority_test_preds = [50, 50, 50, 50, 50 ... , 50, 50, 50, 50, 50]`

### KNN Baseline Predictions 
`test_preds = [48, 46, 1, 18, 40, ..., 33, 44, 44, 44, 44]`

### Actual (Gold) Test Labels
`df_test['labels'] = [48, 46, 1, 41, 40, ..., 33, 44, 44, 44, 44]`


## 3. How to Use This

This script runs in a linear fashion to return the scores for both simple baseline - no user inputs needed.