# -*- coding: utf-8 -*-

import numpy as np
import math
import gudhi
import torch
from torch import nn
from typing import List

# MODIFIED from https://github.com/cambridgeltl/iso-study/blob/master/scripts/gh_script.py
def distance_matrix(embs: List[np.ndarray], metric='euclidean') -> np.ndarray:
    """
    Computes distance matrices from the embedding matrices for each document
    """ 
    
    embs_temp = np.vstack(embs)
    embs = torch.from_numpy(embs_temp)
    
    if metric=='euclidean':
        dist = torch.sqrt(2 - 2 * torch.clamp(torch.mm(embs, torch.t(embs)), -1., 1.))
    
    elif metric=='poincaré':
        dist = np.zeros((embs.shape[0], embs.shape[0]))
        for i in range(embs.shape[0]):
            for j in range(embs.shape[0]):
                num = (np.linalg.norm(embs[i,:] - embs[j,:]))**2
                denom = (1 - (np.linalg.norm(embs[i,:])**2)) * (1 - (np.linalg.norm(embs[j,:])**2))
                dist[i][j] = np.arccosh(1 + 2*(num/denom))
                
    if metric=='euclidean': 
      out = dist.cpu().numpy()
    else: 
      out = 2*(dist - np.min(dist))/np.ptp(dist)-1
    return out

# COPIED from https://github.com/cambridgeltl/iso-study/blob/master/scripts/gh_script.py
def compute_diagram(x, homo_dim=1):
    """
    This function computes the persistence diagram on the basis of the distance matrix
    and the homology dimension
    """
    rips_tree = gudhi.RipsComplex(x).create_simplex_tree(max_dimension=homo_dim)
    rips_diag = rips_tree.persistence()
    return [rips_tree.persistence_intervals_in_dimension(w) for w in range(homo_dim)]

# COPIED from https://github.com/cambridgeltl/iso-study/blob/master/scripts/gh_script.py
def compute_distance(x, y, homo_dim = 1):
    diag_x = compute_diagram(x, homo_dim=homo_dim)
    diag_y = compute_diagram(y, homo_dim=homo_dim)
    return min([gudhi.bottleneck_distance(x, y, e=0) for (x, y) in zip(diag_x, diag_y)])
