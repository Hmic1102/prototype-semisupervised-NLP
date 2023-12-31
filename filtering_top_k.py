# -*- coding: utf-8 -*-
"""filtering_top_k.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dv7uU2C-cxuSrCsxx9ap27_DjUuAfkTK
"""

import numpy as np

def top_k_filter(Prob, samples, k):
  filtered_samples = []

  # top-k highest prob
  top_indices= np.argsort([np.max(softmax) for softmax in Prob])[-k:]

  # Select the top-k samples based on max softmax scores
  filtered_samples = [samples[i] for i in top_indices]

  return filtered_samples