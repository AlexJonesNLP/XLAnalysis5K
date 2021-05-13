# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import os
from typing import List, Dict

def readTexts(path_to_texts: str) -> List[List]:
  '''
  Reads in Bible texts from supplied file directories 
  '''
  print('Reading in texts . . .')
  out = []
  for filename in os.listdir(path_to_texts):
      path_to_file = os.path.join(path_to_texts, filename)
      with open(path_to_file, 'r') as f:
          txt = f.read().splitlines()
          if len(txt)>0:
              out.append(txt)
  return out

'''
Generate a list of the languages used
'''
def getLangs(path_to_texts: str) -> List[str]:
  return [f.replace('.txt', '').replace('-NT', '') for f in os.listdir(path_to_texts)]

def embedLaBSE(model, texts: List[List]) -> List[np.ndarray]:
  '''
  Produces LaBSE embeddings of the input texts using the input LaBSE model

  Returns lists of arrays of shape N x emb_dim, where N is the number of 
  sentences in the text and emb_dim is the embedding dimension of LaBSE (768)

  The list containing these arrays is the same length as 'texts'
  '''
  print('Getting LaBSE embeddings . . .')
  return [model.encode(text) for text in tqdm(texts)]

def embedLASER(model, texts: List[List], iso_lookup: Dict[str, str]) -> List[np.ndarray]:
  '''
  Produces LASER embeddings of the input texts using the input LASER model

  If LASER has training data for a given language, it uses whatever tokenizer
  it is assigned for that language; else, it embeds using the English tokenizer
  '''

  print('Getting LASER embeddings . . .')
  embeddings = []
  for i in tqdm(range(len(iso_lookup))):
    lang_id = iso_lookup[i]

    # If the language ID (ISO code) is in the dictionary
    if lang_id is not None:
      # Try embedding using the language-specific tokenizer
      try:
        embedding = model.embed_sentences(texts[i], lang_id)
      # If that fails, use the English tokenizer
      except:
        embedding = model.embed_sentences(texts[i], 'en')
    
    # If the language ID isn't in the dictionary
    else: 
      embedding = model.embed_sentences(texts[i], 'en')
    
    embeddings.append(embedding)
  
  return embeddings
