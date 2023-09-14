import numpy as np

# Step 1: Calculate the softmax scores (assuming `samples` is a list of lists with the scores)
samples = [...] # Replace with your 5000 samples with their scores

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

filtered_samples = []

for sample in samples:
    # Step 2: Apply softmax and find the highest softmax score
    softmax_scores = softmax(np.array(sample))
    max_softmax_score = np.max(softmax_scores)
    
    # Step 3: Calculate the sum of the rest of the softmax scores
    rest_softmax_scores_sum = np.sum(softmax_scores) - max_softmax_score
    
    # Step 4: Compare the highest softmax score with 50% of the sum of rest of the scores
    if max_softmax_score < 0.5 * rest_softmax_scores_sum:
        # Step 5: Filter out the sample
        continue
    
    filtered_samples.append(sample)

# Now, `filtered_samples` contains the samples that passed the filtering criterion
