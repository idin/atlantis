import numpy as np


def get_entropy(x):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(x, return_counts=True, axis=0)
    probability = count/len(x)
    entropy = np.sum((-1)*probability*np.log2(probability))
    return entropy


def get_joint_entropy(x, y):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    xy = np.c_[x, y]
    return get_entropy(xy)


def get_conditional_entropy(y, x):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return get_joint_entropy(y, x) - get_entropy(x)


def get_information_gain(y, x):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return get_entropy(y) - get_conditional_entropy(y, x)
