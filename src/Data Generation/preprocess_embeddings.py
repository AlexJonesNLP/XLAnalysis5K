# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import normalize
from typing import List

def preprocess_embeddings(vecs: np.ndarray) -> np.ndarray:
    '''
    Preprocess embeddings before performing isomorphism computations
    Procedure consists of normalization, mean-centering, and re-normalization
    '''
    npvecs = np.vstack(vecs)
    
    # Step 1: Length normalize
    npvecs = normalize(npvecs, axis=1, norm='l2')
    # Step 2: Mean centering
    npvecs = npvecs - npvecs.mean(0)
    # Step 3: Length normalize again
    npvecs = normalize(npvecs, axis=1, norm='l2')
    
    # Return double-normalized and mean-centered vectors
    return npvecs
