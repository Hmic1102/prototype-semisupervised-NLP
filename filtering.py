import numpy as np


def filtering(Prob):
    
    filtered_samples = []

    for softmax in Prob:
        max_softmax_score = np.max(softmax)
    
        rest_softmax_scores_sum = np.sum(softmax) - max_softmax_score
    
        if max_softmax_score < 0.5 * rest_softmax_scores_sum:
            continue
    
        filtered_samples.append(softmax)

