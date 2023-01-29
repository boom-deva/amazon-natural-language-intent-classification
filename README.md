# Amazon Natural Language Intent Classification

## Abstract
Our goal is to build a multi-classification NLP model that ingests input from a user and outputs a probabilistic prediction of each of the class labels that message/intention falls into. Our literature review surveyed various approaches from traditional Naive Bayes Classifiers, Support Vector Machines, and LSTM models to more sophisticated hierarchical models. Our extensions were based on such hierarchical approaches: the first extension was a hierarchical modeling tree, while our second extension created a hierarchy by combining target labels into parent classes to create a parent-child labels and made predictions using an LSTM. Our Hierarchical Modeling Tree had an accuracy of 0.8551 and F1 score of 0.8634, which underperformed our Parent-Child Hierarchical LSTM model which had a best accuracy of 0.8697 and F1 score of 0.8681 on the dev set after hyperparameter tuning. On the test set, we saw testing set accuracy scores of 0.8439 and 0.8524 respectively and F1 scores of 0.8672 and 0.8503 respectively, exceeding strong baseline performance slightly.

## Introduction

Classifying user intent has multitudinous applications for problems across industries. For example, in the consumer banking industry, automated chatbots take in queries from a user and return a number of suggested self-help links via the classification of their issues based on certain phrases in the input. In the airline industry, customers call to inquire about flight information and expect to be directed to the right source to execute their demands. More broadly though, virtual assistants such as Amazon Alexa are tasked with parsing language to then classify intentions in order to elicit the appropriate response. The Amazon Massive Intent Dataset provides a rich source of data to train an NLP model to accomplish this.

## Natural Language Models

### Model 1: Parent-Child Prediction using LSTM

We attempted to improve upon our strong baseline by implementing an extended LSTM model. Because the individual labels in our data contain prefixes that group them into parent classes, it seemed possible that our model could use these parent classes to learn more during its training process and thus increase performance.

Similar to the strong baseline model, the extended LSTM model takes in tokenized user utterances, gets their Glove embeddings, passes them through a bidirectional LSTM and fully connected layers, and outputs the predictions of intent.

The main difference between the strong baseline model and extended model is the way that loss is calculated. Instead of measuring loss solely based on the given labels, we place those that are related to each other into the same category and give them a parent label. To avoid confusion, we call the given labels “child labels” and the created labels “parent labels”. Since we are required to predict both parent and child labels, there are some minor changes to the baseline architecture: the input to the extended LSTM model is padded with an additional `<PAD>` token at the end of all text samples.

The output of the actual last token of the sentences produces the parent class prediction. The output of the $<$PAD$>$ token produces the child class prediction, which is the one used to compute classification accuracy. The loss of the extended model becomes the sum of the losses pertaining to the parent class prediction and child class prediction.

We experimented with hyperparameter tuning for this extended model by comparing different combinations of learning rate and batch size. The table found in the paper pdf shows our accuracy and F1 score for each combination using the development data.

We found that using a learning rate of 0.01 and a batch size of 128 yielded the best accuracy of 0.8697. The F1 score of the model with these hyperparameters is 0.8681. We found that there are not major issues with class imbalance for these data and that changes in our F1 scores are comparable to changes in our accuracy scores.

### Model 2: Hierarchical Modeling Tree
Looking at the performance of extension 1, we decided to exploit the hierarchical structure of our data through a more complex modeling approach. Similar to extension 1, we classified our labels into 18 "parent classes".

For the purpose of modeling, we devised a modeling tree, where we have a multi-class classification model at the root node, that classifies each sequence into one of the 18 parent classes. Depending on the classification of this model, we traverse down to the respective child node, which uses a different model trained specifically on data belonging to that parent class. The intuition behind such a structure was that each child model would be well-trained and better suited to preform predictions for similar data belonging to one class.

All of the models in the root node and the child node use the same LSTM structure that was specified in the published baseline. Input tokens are passed through an embedding layer that uses Glove embeddings. These are then fed into the LSTM layer, the output of which is passed through a linear layer which outputs one of the classes.

As we can see from the figure, our tree modeling structure has a root model at the top that classifies into one of the 18 parent classes, each of which have a model of their own.

We tested out our model on the development data using different combinations of hyperparameters: (i) Learning rate was changed uniformly across all models in the tree, while (ii) different combinations of batch sizes were used for the root model and the child models. The results of hyperparameter tuning are summarized in the paper.

After hyperparameter tuning, the best model achieved an accuracy of 0.8551 using a root batch size of 128, a child batch size of 64 and a learning rate of 0.01. The F1 score for this combination of hyperparameters was 0.8634. Contrary to our hypothesis, our extension 2 was not able to outperform extension 1 in the best case scenario when looking at the overall accuracy. However, the F1 score, while on par with that of extension 1, is relatively higher than the accuracy achieved by this model. This shows that this architecture was able to better counter the class imbalance within the dataset.

## Error Analysis

The two `text_label` classes that had the highest proportion of misclassifications were `general_quirky` (label 12) and `qa_factoid` (label 49). The former represents random requests while the latter represents requests for factual information.

Within these top 2 label misclassifications, we deduced 3 major categories of errors:

1. Slang and Faux Word Confusion
2. Oversensitivity/Overfitting to Food-Themed Words
3. Short and Vague Commands

Examples of Category 1 include phrases like `wakey wakey eggs and bakey`, where the only word that really made sense dictionary-wise was `eggs`. The model was unable to understand the terms `wakey` and `bakey` which were really meant to be clear signals for quirky requests, but our model focused on `eggs` as the only word that made sense and thought it was a recipe request and so predicted label 9 `cooking_recipe`. This is also an example of Category 2 errors.

A more prominent example of Category 2 error is the phrase `are jello shots calorie free` which was a clear question with a fixed yes/no factual answer. However, our model honed in on `jello shots` and thought the food term warranted a recipe request and likewise predicted label 9 as well.

As for Category 3, there were many examples of this where requests were far too terse that there just wasn't enough information or context for our model to go off of. For example, there were requests that were solely `none` or `the cosmos` or `sports` which were so vague that our model ended up honing into the only nouns that were available to make the closest categorical prediction.

## Conclusions
Our published baseline demonstrated the use of a simple LSTM structure for the purpose of multi-class classification. Our dataset had 60 class labels, and the published baseline was able to achieve an accuracy of 84.33\% on the test set.

For our extensions, we decided to exploit the hierarchical nature of our dataset through two modeling approaches: (i) an LSTM model that predicts the parent class along with the child class, and (ii) a modeling tree architecture that uses the parent class predictions to decide on which pre-trained child class model to use. The first extension was able to achieve an accuracy of 85.24\% on the test set, which outperformed the published baseline. Moreover, the second extension was able to achieve an accuracy of 0.8439, which was not able to improve on the results of extension 1. However, the F1 score for this model was 0.8672, which is higher than the F1 score of 0.8503 achieved by extension 1.

Extension 1 outperformed all of our models and gave the best results in our experiments. Looking at the mistakes this model made, we were able to identify three error classes: (i) Slang and Faux words, (ii) Oversensitivity/overfitting to food-themed words, and (iii) Short and vague commands.

For Extension 2, we dived deeper into understanding why it was not able to meet expectations. While the extension was still able to beat the published baseline, one of the reasons for it to not meet expectations could be the significantly larger number of hyperparamters used. For the purpose of this project, we used uniform hyperparameters across each child model. However, each child model could be further optimized to get the best hyperparameters. Moreover, diving deeper into each model showed that the root model was able to achieve around 90\% accuracy, which means that 10\% of the data points were not being classified by the correct child model, which could explain the lower accuracy.
