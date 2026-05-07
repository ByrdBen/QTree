import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors, cm 
import pickle
import os as os

def get_transitions(PCA_list, map_list, qdim, rdim, cdim):
    """""""""""""""
    
    """""""""""""""
    # Setting dimension size for proper indexing
    n_bare = qdim * cdim

    # Ensure that we can treat rows of our PCA as a normalized distribution
    def _norm_rows(M):
        s = M.sum(axis=1, keepdims=True)
        return M / np.where(s > 0, s, 1)

    # Trim down the number of states which correspond to 
    # only the bare states assigned within a specific rung
    def extract_bare_subspace(PCA_list, map_list):
        return [PCA_list[k][map_list[k], :] for k in range(len(PCA_list))]

    # 
    def pca_transition_metric(PCA_list, qdim, n_bare):
        relevant = [j for j in range(n_bare) if j % qdim in (0, 1)]
        
        dMs = [_norm_rows(PCA_list[k+1]) - _norm_rows(PCA_list[k])
             for k in range(len(PCA_list) - 1)]

        # Distance between successive elements of dM
        frob     = np.array([np.linalg.norm(dM[relevant], 'fro')   for dM in dMs])
        # Normalized rows of dM
        row_norm = np.array([np.linalg.norm(dM[relevant], axis=1)  for dM in dMs])  # (n_transitions, n_relevant)
        # Normalized columns of dM
        col_norm = np.array([np.linalg.norm(dM[relevant], axis=0)  for dM in dMs])  # (n_transitions, n_bare)
    
        return frob, row_norm, col_norm, relevant
        
    # Look at subspace of states corresponding to assigned bare states
    bare_states = extract_bare_subspace(PCA_list, map_list)

    # Grabbing results 
    res = pca_transition_metric(bare_states, qdim, n_bare)

    return res

def identify_crossings(col_norm, row_norm, relevant, qdim, threshold):
    crossings = []
    
    for k in range(len(col_norm)):
        if col_norm[k].max() < threshold:
            continue
        
        j_partner = np.argmax(col_norm[k])           # bare branch index, direct
        
        j_rel_idx = np.argmax(row_norm[k])           # index into relevant list
        j_rel = relevant[j_rel_idx]                  # bare branch index of primary branch
        
        crossings.append({
          'n_r': k,
          'branch':  (j_rel     % qdim, j_rel     // qdim),
          'partner': (j_partner % qdim, j_partner // qdim),
        })

    return crossings
                