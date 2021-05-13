# -*- coding: utf-8 -*-

import numpy as np
import itertools
from tqdm import tqdm
import os
import torch
from torch import nn
import faiss

'''

Params
******
src_emb: array of size number_of_source_sentences X embedding_dimension
tgt_emb: array of size number_of_target_sentences X embedding_dimension
k: number of neighbors to return
batch_size: batch size

Returns
*******
cos_sims: cosine similarity scores for each of k nearest neighbors for each source sentence
inds: target indices of k nearest neighbors for each source sentence

Modeled off of LASER source code: https://github.com/facebookresearch/LASER/blob/master/source/mine_bitexts.py

'''
def knnSearch(src_emb, tgt_emb, k=1, batch_size=1):
    GPU = faiss.StandardGpuResources() # enables GPU for similarity search with FAISS
    emb_dim = src_emb.shape[1] # Embedding dimension
    num_src_sents = src_emb.shape[0]
    num_tgt_sents = tgt_emb.shape[0]
    cos_sims = np.zeros((num_src_sents, k), dtype=np.float32)
    inds = np.zeros((num_src_sents, k), dtype=np.int64)
    for s_min in range(0, num_src_sents, batch_size):
        s_max = min(s_min + batch_size, num_src_sents)
        src_sims = []
        src_inds = []
        for t_min in range(0, num_tgt_sents, batch_size):
            t_max = min(t_min + batch_size, num_tgt_sents)
            idx = faiss.IndexFlatIP(emb_dim)
            idx = faiss.index_cpu_to_gpu(GPU, 0, idx)
            idx.add(tgt_emb[t_min : t_max])
            src_sim, src_ind = idx.search(src_emb[s_min : s_max], min(k, t_max-t_min))
            src_sims.append(src_sim)
            src_inds.append(src_ind + t_min)
            del idx
        src_sims = np.concatenate(src_sims, axis=1)
        src_inds = np.concatenate(src_inds, axis=1)
        sorted_inds = np.argsort(-src_sims, axis=1)
        for i in range(s_min, s_max):
            for j in range(k):
                cos_sims[i, j] = src_sims[i-s_min, sorted_inds[i-s_min, j]]
                inds[i, j] = src_inds[i-s_min, sorted_inds[i-s_min, j]]
    return cos_sims, inds

'''
Retrieves k-nearest neighbor indices and similarity means for margin scoring
If forward==True: finds neearest neighbors and indices for all source sentences
If backward==True: finds nearest neighbors and indices for all target sentences
In the approach implemented in our paper, we perform both forward and backward search
'''
def directedMeansAndInds(src_emb, tgt_emb, forward=False, backward=False, k=1, batch_size=1):
    assert forward != backward, "Please choose either forward or backward"
    if forward:
        cos_sims, inds = knnSearch(src_emb, tgt_emb, min(tgt_emb.shape[0], k), batch_size)
        return cos_sims.mean(axis=1), inds
    elif backward:
        cos_sims, inds = knnSearch(tgt_emb, src_emb, min(src_emb.shape[0], k), batch_size)
        return cos_sims.mean(axis=1), inds

'''

Params
******
pred_tuples: predicted sentence pairs
gold_tuples: ground-truth sentence pairs

Returns
*******
Unweighted F1, precision, recall

'''

def computeF1(pred_tuples, gold_tuples):
    tp = 0 # true positives
    fp = 0 # false positives
    prec = 0
    rec = 0
    f1 = 0
    epsilon = 1e-8 # To prevent division by zero
    for pair in pred_tuples:
        if pair in gold_tuples:
            tp += 1
        else:
            fp += 1 
    prec = tp / (len(pred_tuples) + epsilon)
    rec = tp / len(gold_tuples)
    f1 = 2*prec*rec / (prec+rec+epsilon)
    return f1, prec, rec

'''

Params
******
src_embs: array of size (number_of_source_sentences * embedding_dimension)
tgt_embs: array of size (number_of_source_sentences * embedding_dimension)
batch_size: batch size
num_neighbors: number of neighbors
average: whether to return an average margin score across aligned sentences

Returns
*******
concat_pairs: list of mined sentence pairs
margin_scores: list of scores corresponding to mined pairs

'''
def mineSentencePairs(src_embs: list, tgt_embs: list, batch_size=100, num_neighbors=4, average=False):

    # Retrieve means and indices in the forward direction . . .
    fwd_means, fwd_inds = directedMeansAndInds(src_embs, tgt_embs, forward=True, k=num_neighbors, batch_size=batch_size)
    # . . . and in the backward direction
    bwd_means, bwd_inds = directedMeansAndInds(src_embs, tgt_embs, backward=True, k=num_neighbors, batch_size=batch_size)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # We'll sneak this in as an extra capability of this function so we can compute 
    # average margin scores over force-aligned sentence pairs (DV #2)
    margin_scores_aligned = []
    if average:
        for i in range(fwd_inds.shape[0]):
            aligned_margin = (src_embs[i].dot(tgt_embs[i])) / np.average((fwd_means[i], bwd_means[i]))
            margin_scores_aligned.append(aligned_margin)
        return np.average(margin_scores_aligned)
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    # else . . .
    fwd_margin_scores = np.zeros(fwd_inds.shape)
    for i in range(fwd_inds.shape[0]):
        for j in range(fwd_inds.shape[1]):
            tgt_ind = fwd_inds[i,j]
            # Compute ratio margin score between each source sentence and each of its k-nearest neighbors
            margin_score = (src_embs[i].dot(tgt_embs[tgt_ind])) / np.average((fwd_means[i], bwd_means[tgt_ind]))
            # Store the result
            fwd_margin_scores[i,j] = margin_score
    
    # We will store the source index, target index, and margin score for the best
    # pairs found using forward search
    best = np.zeros((fwd_inds.shape[0], 3))
    # Take pair that maximizes margin score for each source sentence
    best_inds = fwd_inds[np.arange(src_embs.shape[0]), fwd_margin_scores.argmax(axis=1)]
    for i in range(fwd_inds.shape[0]):
        best_score, ind = (np.max(fwd_margin_scores[i]), np.argmax(fwd_margin_scores[i]))
        best[i] = ((i+1, best_inds[i]+1, best_score)) # Assumption is that GROUND TRUTH VALUES ARE 1-INDEXED!!!

    # Repeat process in backward direction (finding matches in source text for target sentences)
    bwd_margin_scores = np.zeros(bwd_inds.shape)
    for i in range(bwd_inds.shape[0]):
        for j in range(bwd_inds.shape[1]):
            tgt_ind = bwd_inds[i,j]
            margin_score = (tgt_embs[i].dot(src_embs[tgt_ind])) / np.average((bwd_means[i], fwd_means[tgt_ind]))
            bwd_margin_scores[i,j] = margin_score
            
    bwd_best = np.zeros((bwd_inds.shape[0], 3))
    best_inds = bwd_inds[np.arange(tgt_embs.shape[0]), bwd_margin_scores.argmax(axis=1)]
    for i in range(bwd_inds.shape[0]):
        best_score, ind = (np.max(bwd_margin_scores[i]), np.argmax(bwd_margin_scores[i]))
        bwd_best[i] = ((best_inds[i]+1, i+1, best_score))
    
    # Best triples (src_idx, tgt_idx, margin_score) from forward/backward searches
    fwd_best = [tuple(best[i]) for i in range(best.shape[0])]
    bwd_best = [tuple(bwd_best[i]) for i in range(bwd_best.shape[0])]

    pairs_and_scores = []
    # Take INTERSECTION of forward and backward searches
    pairs_and_scores = list(set(fwd_best) & set(bwd_best))

    pairs_and_scores = list(dict.fromkeys(pairs_and_scores))
    concat_pairs = [(triplet[0], triplet[1]) for triplet in pairs_and_scores] # Store indices only
    concat_pairs_int = []
    for tup in concat_pairs:
        concat_pairs_int.append((int(tup[0]), int(tup[1]))) # Ground-truth indices are ints, so change type
    concat_pairs = concat_pairs_int

    margin_scores = [triplet[2] for triplet in pairs_and_scores] # Store scores only
                                    
    return concat_pairs, margin_scores
