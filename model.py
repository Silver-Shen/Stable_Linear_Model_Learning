# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def column_wise_resampling(x, replacement = False, random_state = 0, **options):
    """
    Perform column-wise random resampling to break the joint distribution of p(x).
    In practice, we can perform resampling without replacement (a.k.a. permutation) to retain all the data points of feature x_j. 
    Moreover, if the practitioner has some priors on which features should be permuted,
    it can be passed through options by specifying 'sensitive_variables', by default it contains all the features
    """
    rng = np.random.RandomState(random_state)
    n, p = x.shape
    if 'sensitive_variables' in options:
        sensitive_variables = options['sensitive_variables']
    else:
        sensitive_variables = [i for i in range(p)] 
    x_decorrelation = np.zeros([n, p])
    for i in sensitive_variables:
        var = x[:, i]
        if replacement: # sampling with replacement
            x_decorrelation[:, i] = np.array([var[rng.randint(0, n)] for j in range(n)])
        else: # permutation     
            x_decorrelation[:, i] = var[rng.permutation(n)]
    return x_decorrelation

def decorrelation(x, solver = 'adam', hidden_layer_sizes = (5,5), max_iter = 500, random_state = 0):
    """
    Calcualte new sample weights by density ratio estimation
           q(x)   P(x belongs to q(x) | x) 
    w(x) = ---- = ------------------------ 
           p(x)   P(x belongs to p(x) | x)
    """
    n, p = x.shape
    x_decorrelation = column_wise_resampling(x, random_state = random_state)
    P = pd.DataFrame(x)
    Q = pd.DataFrame(x_decorrelation)
    P['src'] = 1 # 1 means source distribution
    Q['src'] = 0 # 0 means target distribution
    Z = pd.concat([P, Q], ignore_index=True, axis=0)
    labels = Z['src'].values
    Z = Z.drop('src', axis=1).values
    P, Q = P.values, Q.values
    # train a multi-layer perceptron to classify the source and target distribution
    clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    clf.fit(Z, labels)
    proba = clf.predict_proba(Z)[:len(P), 1]
    weights = (1./proba) - 1. # calculate sample weights by density ratio
    weights /= np.mean(weights) # normalize the weights to get average 1
    weights = np.reshape(weights, [n,1])
    return weights
