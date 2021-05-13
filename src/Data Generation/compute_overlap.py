# -*- coding: utf-8 -*-

from collections import Counter
from transformers import AutoTokenizer

def getChars(sents: list) -> set:
  '''
  Get multiset of characters in text
  '''
  chars = []
  for s in sents:
    for char in list(s):
      chars.append(char)
  return chars

def getCharOverlap(chars1: list, chars2: list) -> float:
  '''
  Get the multiset character-level overlap between texts 1 and 2, corresponding
  to chars1 and chars2, respectively
  '''
  size_chars1, size_chars2 = len(chars1), len(chars2)
  # Multiset analog to set intersection
  intersect = list((Counter(chars1)&Counter(chars2)).elements())
  return len(intersect) / (size_chars1+size_chars2-len(intersect))

labse_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')

# Just a utility function so we don't have to write out parameters repeatedly
def _tokenize(sentences: list) -> torch.tensor:
    return labse_tokenizer(sentences, 
                           padding=True, 
                           truncation=True, 
                           max_length=64, 
                           return_tensors='pt')['input_ids'] # We're interested in just the IDs

def getTokenIDs(tokens: torch.tensor) -> set:
    '''
    Get list of tokens IDs from an array of tokenized sentences
    '''
    token_ids = []
    for tokenized_sentence in tokens:
        for token_id in tokenized_sentence:
            token_ids.append(token_id.item())
    return token_ids

def getTokenOverlap(tokens1: list, tokens2: list) -> float:
    '''
    Get the multiset token-level overlap between texts 1 and 2, corresponding
    to tokens1 and tokens2, respectively
    '''
    # List/multiset version of Jaccard coefficient
    size_tokens1, size_tokens2 = len(tokens1), len(tokens2)
    intersect = list((Counter(tokens1)&Counter(tokens2)).elements())
    return len(intersect) / (size_tokens1+size_tokens2-len(intersect))
