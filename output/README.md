# CIS 5300 - Final Project Predictions

### Example: Running evaluation script on output

predictions_df = pd.read_csv('predictions.csv')

pred = list(predictions_df['test_pred'])

gold = list(predictions_df['test_label'])

evaluate(pred, gold)

### Output of above example

85.23873570948219
