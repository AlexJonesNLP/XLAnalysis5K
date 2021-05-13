# -*- coding: utf-8 -*-

!python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps lang2vec
import lang2vec.lang2vec as l2v
import numpy as np
from tqdm import tqdm
from typing import List, Dict

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return u.dot(v)

def getTypologicalDistances(iso_lookup: Dict[str], dist_type='geo') -> List[float]:
  '''
  Compute geographic distances between languages using lang2vec
  dist_type: must be one of 'geo', 'syntax_knn', phonology_knn', 
  or 'inventory_knn'
  '''
  assert dist_type in ['geo', 'syntax_knn', 'phonology_knn', 'inventory_knn']

  # Generate geographic location vectors for each language
  typology_vecs = [np.array(l2v.get_features(lang, dist_type)[lang])
              for lang in tqdm(iso_lookup)]
  # Compute distances between vectors for all possible language pairs
  typology_dists = [(1-cosine_similarity(u,v)) for u,v in 
               tqdm(itertools.combinations(typology_vecs, 2))]
  return typology_dists
