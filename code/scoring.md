### Description of evaluation metric:

For this project, we elect to use total classification accuracy as our evaluation metric. Because we are attempting to implement models that correctly classify input sequence into multiple intent classification labels, we require a metric that reports our overall performance toward this goal.

This metric allows us to measure the proportion of inputs which our models classify correctly across all classes. It is calculated using the simple formula below:


Total classification accuracy = (Number of correct classifications) / (Total number of classifications)


This metric has been successfully used in a wide variety of multiclass classification studies, including the below study which performs multiclass classification of intent using LSTM architectures:

G. Di Gennaro, A. Buonanno, A. Di Girolamo, A. Ospedale and F. A. N. Palmieri, "Intent classification in question-answering using LSTM architectures" in Progresses in Artificial Intelligence and Neural Systems, Singapore:Springer, vol. 184, pp. 115-124, 2020.
https://link.springer.com/chapter/10.1007/978-981-15-5093-5_11


A description of this accuracy metric for multiclass classification can be found in this Wikipedia article:

https://en.wikipedia.org/wiki/Accuracy_and_precision#In_multiclass_classification

### Example of running accuracy function:

pred = [1, 1, 2, 0, 2, 1, 0, 0]

gold = [0, 1, 1, 0, 2, 0, 1, 2]

evaluate(pred, gold)

### Output of above example:

37.5
