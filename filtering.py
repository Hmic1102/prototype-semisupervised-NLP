import numpy as np


def filtering(Prob,percent):
    
    filtered_samples = []

    for softmax in Prob:
        max_softmax_score = np.max(softmax)
    
        rest_softmax_scores_sum = np.sum(softmax) - max_softmax_score
    
        if max_softmax_score < percent * rest_softmax_scores_sum:
            continue
    
        filtered_samples.append(softmax)

