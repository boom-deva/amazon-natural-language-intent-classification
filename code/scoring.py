def evaluate(pred, gold):
    """

    Evaluate performance using accuracy metric

    INPUT:
    pred    – list of predicted classes from model
    gold    - list of corresponding "gold" classes/labels from data

    OUTPUT:
    acc     - accuracy percentage (float)

    """

    # counter for correct predictions
    n_correct = 0

    # total labels
    n = len(gold)

    # loop through labels and check correctness
    for i in range(n):
        if pred[i] == gold[i]:
            n_correct += 1

    # compute accuracy
    acc = 100. * n_correct / n

    return acc