"""

This script contains a selection of sample acquisition methods for active learning.
All functions here operate on model predictions in the form of logits_N_K_C = [N, num_inference_samples, num_classes].
Here N is a molecule, K are the number of sampled predictions (i.e., 10 for a 10-model ensemble), and C = 2 ([0, 1]):

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

# ============================================================================
# Modifications by Yukun Luo:
#
# 1. Sample-level data tracking for all acquisition strategies:
#    - Saves per-batch CSV with: smiles, probability, mutual_information,
#      uncertainty, Tanimoto_hits, Tanimoto_nonhits, Tanimoto_diff
#    - Output path: {output_dir}/obtained_data/seed{seed}_obtained_data_{dataset}/
#
# 2. Novel acquisition strategies:
#    - Hybridcombo9010/8020/955: Fixed-ratio exploitation + uncertainty mixing
#    - policy1-policy4: Time-triggered switching strategies over AL cycles
# ============================================================================

import numpy as np
import torch
from torch import Tensor
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as ECFPbitVec
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import Chem
import math
import pandas as pd
import os

class Acquisition:
    def __init__(self, method: str, seed: int = 42, **kwargs):

        self.acquisition_method = {'random': self.random_pick,
                                   'uncertainty': greedy_exploration,
                                   'exploitation': greedy_exploitation,
                                   'bala': bala,
                                   'bald': bald,
                                   'similarity': similarity_search,
                                   'Hybridcombo9010': hybridcombo9010,
                                   'Hybridcombo8020': hybridcombo8020,
                                   'Hybridcombo955': hybridcombo955,
                                   'policy1': policy1,
                                   'policy2': policy2,
                                   'policy3': policy3,
                                   'policy4': policy4,
                                   }

        assert method in self.acquisition_method.keys(), f"Specified 'method' not available. " \
                                                         f"Select from: {self.acquisition_method.keys()}"

        self.method = method
        self.params = kwargs
        self.rng = np.random.default_rng(seed=seed)
        self.iteration = 0
        
        

    def acquire(self, ds_screen, cycle, seed, ensemble_size, architecture, INFERENCE_BATCH_SIZE, dataset, logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray[str], n: int, timestamp=None, output_dir='results', hits_discovered=None, test_precision=None) -> \
            np.ndarray[str]:

        self.iteration += 1
    
        return self.acquisition_method[self.method](logits_N_K_C=logits_N_K_C, smiles=smiles,  hits=hits, nonhits=nonhits, n=n, ds_screen=ds_screen, 
                                                    cycle=cycle, seed=seed, ensemble_size=ensemble_size, architecture=architecture, INFERENCE_BATCH_SIZE=INFERENCE_BATCH_SIZE, dataset=dataset,
                                                    iteration=self.iteration, timestamp=timestamp, output_dir=output_dir, test_precision=test_precision, **self.params)

    def __call__(self, *args, **kwargs) -> np.ndarray[str]:
        return self.acquire(*args, **kwargs)

    def random_pick(self, hits: np.ndarray[str], nonhits: np.ndarray, logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, radius: int = 2, nBits: int = 1024, return_smiles: bool = True, output_dir='results', **kwargs) -> np.ndarray:
        """ select n random samples """
        picks_idx = self.rng.integers(0, len(smiles), n)
        I = mutual_information(logits_N_K_C)
        entropy_mean_N = mean_sample_entropy(logits_N_K_C)
        mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
        total_I = mutual_information(logits_N_K_C)
        # Save sample data to CSV
        picks_index = picks_idx
        picks_idx_np = np.array([smiles[i] for i in picks_index])

        probs = mean_probs_hits[picks_idx].cpu().numpy()
        I = total_I[picks_idx].cpu().numpy()
        I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
        entropy = entropy_mean_N[picks_idx].cpu().numpy()

        similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
            hits=hits,
            nonhits=nonhits,
            smiles=picks_idx_np,
            radius=radius,
            nBits=nBits,
        )
        
        
        # Update DataFrame with new similarity metrics
        df = pd.DataFrame({
            'smiles': picks_idx_np,
            'probability': probs,
            'mutual_information': I,
            'uncertainty': entropy,
            'Tanimoto_hits': similarity_hits,
            'Tanimoto_nonhits': similarity_nonhits,
            'Tanimoto_diff': similarity_diff,
        })
        

        
        timestamp = kwargs.get('timestamp', '')
        dataset = kwargs.get('dataset', '')
        seed = kwargs.get('seed', '')
        folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
        os.makedirs(folder, exist_ok=True)
        filename = f"{self.method}_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"{self.method}_samples_{dataset}_batch{n}.csv"
        filepath = os.path.join(folder, filename)
        header = not os.path.exists(filepath)
        df.to_csv(filepath, mode='a', index=False, header=header)
        with open(filepath, 'a') as f:
            f.write('-' * 10 + '\n')  # Add separator line

        return smiles[picks_idx] if return_smiles else picks_idx


def logits_to_pred(logits_N_K_C: Tensor, return_prob: bool = True, return_uncertainty: bool = True) -> (Tensor, Tensor):
    """ Get the probabilities/class vector and sample uncertainty from the logits """

    mean_probs_N_C = torch.mean(torch.exp(logits_N_K_C), dim=1)
    uncertainty = mean_sample_entropy(logits_N_K_C)

    if return_prob:
        y_hat = mean_probs_N_C
    else:
        y_hat = torch.argmax(mean_probs_N_C, dim=1)

    if return_uncertainty:
        return y_hat, uncertainty
    else:
        return y_hat


def logit_mean(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """ Logit mean with the logsumexp trick - Kirch et al., 2019, NeurIPS """

    return torch.logsumexp(logits_N_K_C, dim=dim, keepdim=keepdim) - math.log(logits_N_K_C.shape[dim])


def entropy(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Calculates the Shannon Entropy """

    return -torch.sum((torch.exp(logits_N_K_C) * logits_N_K_C).double(), dim=dim, keepdim=keepdim)


def mean_sample_entropy(logits_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Calculates the mean entropy for each sample given multiple ensemble predictions - Kirch et al., 2019, NeurIPS"""

    sample_entropies_N_K = entropy(logits_N_K_C, dim=dim, keepdim=keepdim)
    entropy_mean_N = torch.mean(sample_entropies_N_K, dim=1)

    return entropy_mean_N


def mutual_information(logits_N_K_C: Tensor) -> Tensor:
    """ Calculates the Mutual Information - Kirch et al., 2019, NeurIPS """

    # this term represents the entropy of the model prediction (high when uncertain)
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)

    # This term is the expectation of the entropy of the model prediction for each draw of model parameters
    mean_entropy_N = entropy(logit_mean(logits_N_K_C, dim=1), dim=-1)

    I = mean_entropy_N - entropy_mean_N

    return I


def _compute_similarity_metrics(
    hits: np.ndarray[str],
    nonhits: np.ndarray,
    smiles: np.ndarray[str],
    radius: int,
    nBits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp_smiles = [ECFPbitVec(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in smiles]

    if hits is None or len(hits) == 0:
        similarity_hits = np.zeros(len(smiles), dtype=np.float16)
        similarity_diff = np.zeros(len(smiles), dtype=np.float16)
    else:
        fp_hits = [ECFPbitVec(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in hits]
        m_hits = np.zeros([len(hits), len(smiles)], dtype=np.float16)
        for i in range(len(hits)):
            m_hits[i] = BulkTanimotoSimilarity(fp_hits[i], fp_smiles)
        similarity_hits = np.max(m_hits, axis=0)
        similarity_diff = None

    if nonhits is None or len(nonhits) == 0:
        similarity_nonhits = np.zeros(len(smiles), dtype=np.float16)
    else:
        fp_nonhits = [ECFPbitVec(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in nonhits]
        m_nonhits = np.zeros([len(nonhits), len(smiles)], dtype=np.float16)
        for i in range(len(nonhits)):
            m_nonhits[i] = BulkTanimotoSimilarity(fp_nonhits[i], fp_smiles)
        similarity_nonhits = np.max(m_nonhits, axis=0)

    if similarity_diff is None:
        similarity_diff = abs(similarity_hits - similarity_nonhits)

    return similarity_hits, similarity_nonhits, similarity_diff


def _max_similarity_to_hits(
    hits: np.ndarray[str],
    smiles: np.ndarray[str],
    radius: int,
    nBits: int,
) -> np.ndarray:
    if hits is None or len(hits) == 0:
        return np.zeros(len(smiles), dtype=np.float16)

    fp_hits = [ECFPbitVec(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in hits]
    fp_smiles = [ECFPbitVec(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in smiles]
    m = np.zeros([len(hits), len(smiles)], dtype=np.float16)
    for i in range(len(hits)):
        m[i] = BulkTanimotoSimilarity(fp_hits[i], fp_smiles)
    return np.max(m, axis=0)


def greedy_exploitation(hits: np.ndarray[str], nonhits: np.ndarray, logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, radius: int = 2, nBits: int = 1024, output_dir='results', **kwargs) -> np.ndarray[str]:
    """ Get the n highest predicted samples """
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    picks_idx = torch.argsort(mean_probs_hits, descending=True)[:n]

    total_I = mutual_information(logits_N_K_C)
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])

    probs = mean_probs_hits[picks_idx].cpu().numpy()
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)
    entropy = entropy_mean_N[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )

    # Update DataFrame with new similarity metrics
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
    })
    
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"exploitation_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"exploitation_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')  # Add separator line

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]


def greedy_exploration(hits: np.ndarray[str], nonhits: np.ndarray, logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, radius: int = 2, nBits: int = 1024, output_dir='results', **kwargs) -> np.ndarray[str]:
    """ Get the n most samples with the most uncertainty"""
    
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    picks_idx = torch.argsort(entropy_mean_N, descending=True)[:n]
    
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    total_I = mutual_information(logits_N_K_C)
    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])
    
    probs = mean_probs_hits[picks_idx].cpu().numpy()
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = entropy_mean_N[picks_idx].cpu().numpy()
    
    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )
    
    # Update DataFrame with new similarity metrics
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
    })
    
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"uncertainty_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"uncertainty_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')  # Add separator line
    
    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]


def bala(logits_N_K_C: Tensor, smiles: np.ndarray[str],  hits: np.ndarray[str], nonhits: np.ndarray, radius: int = 2, nBits: int = 1024, n: int = 1, output_dir='results', **kwargs) -> np.ndarray[str]:
    """ Get the n molecules with the lowest Mutual Information """
    I = mutual_information(logits_N_K_C)
    picks_idx = torch.argsort(I, descending=False)[:n]

    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    total_I = mutual_information(logits_N_K_C)
    # Save sample data to CSV
    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])

    probs = mean_probs_hits[picks_idx].cpu().numpy()
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = entropy_mean_N[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )
    
    # Update DataFrame with new similarity metrics
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
    })
    
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"bala_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"bala_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')  # Add separator line

    return smiles[picks_idx.cpu()]


def bald(logits_N_K_C: Tensor, smiles: np.ndarray[str],  hits: np.ndarray[str], nonhits: np.ndarray, radius: int = 2, nBits: int = 1024, n: int = 1, output_dir='results', **kwargs) -> np.ndarray[str]:
    """ Get the n molecules with the highest Mutual Information (classic bald) """
    I = mutual_information(logits_N_K_C)
    picks_idx = torch.argsort(I, descending=True)[:n]

    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    total_I = mutual_information(logits_N_K_C)
    # Save sample data to CSV
    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])

    probs = mean_probs_hits[picks_idx].cpu().numpy()
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = entropy_mean_N[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )
    
    # Update DataFrame with new similarity metrics
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
    })
    
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"bald_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"bald_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')  # Add separator line

    return smiles[picks_idx.cpu()]


def similarity_search(logits_N_K_C: Tensor, hits: np.ndarray[str], nonhits: np.ndarray, smiles: np.ndarray[str], n: int, radius: int = 2, nBits: int = 1024,
                      output_dir='results', **kwargs) -> np.ndarray[str]:
    """ 1. Compute the similarity of all screen smiles to all hit smiles
        2. take the n screen smiles with the highest similarity to any hit """

    hit_similarity = _max_similarity_to_hits(hits=hits, smiles=smiles, radius=radius, nBits=nBits)

    # get the n highest similarity smiles to any hit
    picks_idx = np.argsort(hit_similarity)[::-1].copy()[:n]

    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    total_I = mutual_information(logits_N_K_C)
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)

    # Save sample data to CSV
    picks_index = picks_idx
    picks_idx_np = np.array([smiles[i] for i in picks_index])

    probs = mean_probs_hits[picks_idx].cpu().numpy()
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = entropy_mean_N[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )
    
    # Update DataFrame with new similarity metrics
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
    })
    
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"similarity_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"similarity_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')  # Add separator line
    return smiles[picks_idx]


def policy1(logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray, 
               n: int = 1, radius: int = 2, nBits: int = 1024, iteration: int = 0, output_dir='results', **kwargs) -> np.ndarray[str]:
    """
    First 5 cycles: exploitation, then uncertainty
    """
    if iteration < 5:
        # exploitation
        picks_idx = torch.argsort(torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1], descending=True)[:n]
        strategy = ['exploitation'] * n
    else:
        # uncertainty
        entropy_mean_N = mean_sample_entropy(logits_N_K_C)
        picks_idx = torch.argsort(entropy_mean_N, descending=True)[:n]
        strategy = ['uncertainty'] * n

    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    probs = mean_probs_hits[picks_idx].cpu().numpy()
    total_I = mutual_information(logits_N_K_C)
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = mean_sample_entropy(logits_N_K_C)[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )

    # Save analysis CSV
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
        'strategy': strategy,
    })
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"policy_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"policy_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\\n')  # Separator

    return smiles[picks_idx.cpu()]


def policy2(logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray, 
               n: int = 1, radius: int = 2, nBits: int = 1024, iteration: int = 0, output_dir='results', **kwargs) -> np.ndarray[str]:
    """
    Similar to policy1, but uses BALD after first 5 cycles (highest mutual information I).
    """
    if iteration < 5:
        # exploitation
        picks_idx = torch.argsort(torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1], descending=True)[:n]
        strategy = ['exploitation'] * n
    else:
        # BALD (highest mutual information)
        I = mutual_information(logits_N_K_C)
        picks_idx = torch.argsort(I, descending=True)[:n]
        strategy = ['bald'] * n

    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    probs = mean_probs_hits[picks_idx].cpu().numpy()
    total_I = mutual_information(logits_N_K_C)
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = mean_sample_entropy(logits_N_K_C)[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )

    # Save analysis CSV
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
        'strategy': strategy,
    })
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"policy2_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"policy2_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')  # Separator

    return smiles[picks_idx.cpu()]

def _hybridcombo_uncertainty_ratio(
    logits_N_K_C: Tensor,
    smiles: np.ndarray[str],
    hits: np.ndarray[str],
    nonhits: np.ndarray,
    n: int,
    radius: int,
    nBits: int,
    output_dir: str,
    strategy_prefix: str,
    explore_ratio: float,
    **kwargs,
) -> np.ndarray[str]:
    """Mix exploitation with uncertainty using a fixed exploration ratio."""

    n_explore = int(n * explore_ratio)
    if n > 0 and n_explore == 0 and explore_ratio > 0:
        n_explore = 1
    n_exploit = n - n_explore

    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    exploit_idx = torch.argsort(mean_probs_hits, descending=True)[:n_exploit]

    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    uncertainty_rank = torch.argsort(entropy_mean_N, descending=True)
    if n_explore > 0:
        selected_mask = torch.zeros(len(smiles), dtype=torch.bool, device=uncertainty_rank.device)
        selected_mask[exploit_idx] = True
        uncertainty_candidates = uncertainty_rank[~selected_mask[uncertainty_rank]]
        uncertainty_idx = uncertainty_candidates[:n_explore]
    else:
        uncertainty_idx = torch.empty(0, dtype=torch.long, device=uncertainty_rank.device)

    picks_idx = torch.cat([exploit_idx, uncertainty_idx])

    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])
    probs = mean_probs_hits[picks_idx].cpu().numpy()
    total_I = mutual_information(logits_N_K_C)
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)
    entropy = entropy_mean_N[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )

    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
    })

    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"{strategy_prefix}_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"{strategy_prefix}_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')

    return smiles[picks_idx.cpu()]


def hybridcombo955(logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray,
                   n: int = 1, radius: int = 2, nBits: int = 1024, output_dir='results', **kwargs) -> np.ndarray[str]:
    """95/5 fixed mix: exploitation + uncertainty."""
    return _hybridcombo_uncertainty_ratio(
        logits_N_K_C=logits_N_K_C,
        smiles=smiles,
        hits=hits,
        nonhits=nonhits,
        n=n,
        radius=radius,
        nBits=nBits,
        output_dir=output_dir,
        strategy_prefix='hybridcombo955',
        explore_ratio=0.05,
        **kwargs,
    )


def hybridcombo9010(logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray,
                    n: int = 1, radius: int = 2, nBits: int = 1024, output_dir='results', **kwargs) -> np.ndarray[str]:
    """90/10 fixed mix: exploitation + uncertainty."""
    return _hybridcombo_uncertainty_ratio(
        logits_N_K_C=logits_N_K_C,
        smiles=smiles,
        hits=hits,
        nonhits=nonhits,
        n=n,
        radius=radius,
        nBits=nBits,
        output_dir=output_dir,
        strategy_prefix='hybridcombo9010',
        explore_ratio=0.10,
        **kwargs,
    )


def hybridcombo8020(logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray,
                    n: int = 1, radius: int = 2, nBits: int = 1024, output_dir='results', **kwargs) -> np.ndarray[str]:
    """80/20 fixed mix: exploitation + uncertainty."""
    return _hybridcombo_uncertainty_ratio(
        logits_N_K_C=logits_N_K_C,
        smiles=smiles,
        hits=hits,
        nonhits=nonhits,
        n=n,
        radius=radius,
        nBits=nBits,
        output_dir=output_dir,
        strategy_prefix='hybridcombo8020',
        explore_ratio=0.20,
        **kwargs,
    )


def policy3(logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray, 
                  n: int = 1, radius: int = 2, nBits: int = 1024, iteration: int = 0, output_dir='results', **kwargs) -> np.ndarray[str]:
    """
    cycle 0–4：exploitation
    cycle 5–7：uncertainty
    cycle ≥8：exploitation
    """
    
    # - `cycle` (0-based) is the intended AL round index.
    # - `iteration` is incremented in `Acquisition.acquire()` and is effectively 1-based.
    # Use `cycle` when available to avoid off-by-one strategy switches.
    t = kwargs.get('cycle', None)
    t = int(t) if t is not None else int(iteration) - 1

    if t < 5:
        # First 5 cycles: exploitation
        picks_idx = torch.argsort(torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1], descending=True)[:n]
        strategy = ['exploitation'] * n
    elif t < 8:
        # cycles 5-7: uncertainty
        entropy_mean_N = mean_sample_entropy(logits_N_K_C)
        picks_idx = torch.argsort(entropy_mean_N, descending=True)[:n]
        strategy = ['uncertainty'] * n
    else:
        # cycle >= 8: exploitation
        picks_idx = torch.argsort(torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1], descending=True)[:n]
        strategy = ['exploitation'] * n

    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    probs = mean_probs_hits[picks_idx].cpu().numpy()
    total_I = mutual_information(logits_N_K_C)
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = mean_sample_entropy(logits_N_K_C)[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )

    # Save analysis CSV
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
        'strategy': strategy,
    })
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"policy3_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"policy3_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')
    return smiles[picks_idx.cpu()]


def policy4(logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], nonhits: np.ndarray, 
                   n: int = 1, radius: int = 2, nBits: int = 1024, iteration: int = 0, output_dir='results', **kwargs) -> np.ndarray[str]:
    """
    cycle 0–4：exploitation
    cycle 5–7：BALD
    cycle ≥8：exploitation
    """
    # Keep the same round indexing behavior as `dynamicexphit`.
    t = kwargs.get('cycle', None)
    t = int(t) if t is not None else int(iteration) - 1

    if t < 5:
        # First 5 cycles: exploitation
        picks_idx = torch.argsort(torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1], descending=True)[:n]
        strategy = ['exploitation'] * n
    elif t < 8:
        # cycles 5-7: BALD
        I = mutual_information(logits_N_K_C)
        picks_idx = torch.argsort(I, descending=True)[:n]
        strategy = ['bald'] * n
    else:
        # cycle >= 8: exploitation
        picks_idx = torch.argsort(torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1], descending=True)[:n]
        strategy = ['exploitation'] * n

    picks_index = picks_idx.cpu().numpy()
    picks_idx_np = np.array([smiles[i] for i in picks_index])
    mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    probs = mean_probs_hits[picks_idx].cpu().numpy()
    total_I = mutual_information(logits_N_K_C)
    I = total_I[picks_idx].cpu().numpy()
    I = np.maximum(I, 0.0)  # Clamp near-zero negative values to 0
    entropy = mean_sample_entropy(logits_N_K_C)[picks_idx].cpu().numpy()

    similarity_hits, similarity_nonhits, similarity_diff = _compute_similarity_metrics(
        hits=hits,
        nonhits=nonhits,
        smiles=picks_idx_np,
        radius=radius,
        nBits=nBits,
    )

    # Save analysis CSV
    df = pd.DataFrame({
        'smiles': picks_idx_np,
        'probability': probs,
        'mutual_information': I,
        'uncertainty': entropy,
        'Tanimoto_hits': similarity_hits,
        'Tanimoto_nonhits': similarity_nonhits,
        'Tanimoto_diff': similarity_diff,
        'strategy': strategy,
    })
    timestamp = kwargs.get('timestamp', '')
    dataset = kwargs.get('dataset', '')
    seed = kwargs.get('seed', '')
    folder = os.path.join(output_dir, f"obtained_data/seed{seed}_obtained_data_{dataset}")
    os.makedirs(folder, exist_ok=True)
    filename = f"policy4_samples_{dataset}_batch{n}_{timestamp}.csv" if timestamp else f"policy4_samples_{dataset}_batch{n}.csv"
    filepath = os.path.join(folder, filename)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)
    with open(filepath, 'a') as f:
        f.write('-' * 10 + '\n')
    return smiles[picks_idx.cpu()]