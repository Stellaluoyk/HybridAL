import os
import warnings
from collections import Counter
import torch
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.neighbors import NearestNeighbors
import gc
from active_learning.data_prep import MasterDataset, load_hdf5, get_data, split_data, similarity_vectors
from config import ROOT_DIR
warnings.simplefilter(action='ignore', category=FutureWarning)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Cluster import Butina

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Cluster import Butina
from scipy.spatial.distance import pdist


def smiles_to_ecfp_bitmat(smiles_list, radius=2, nBits=1024):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps  = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits) for m in mols]
    bitmat = np.array([[int(b) for b in fp.ToBitString()] for fp in fps], dtype=bool)
    return fps, bitmat

def build_supercluster_labels_butina(smiles_screen, sim_super=0.40, min_super=32):
    """
    Butina coarse clustering. Returns length-n labels (-1 = unclustered).
    sim_super is a similarity threshold; Butina uses distance = 1 - sim_super
    """
    fps, _ = smiles_to_ecfp_bitmat(smiles_screen)

    # Scalar distance function: 1 - Tanimoto
    def tanimoto_dist(x, y):
        return 1.0 - DataStructs.TanimotoSimilarity(x, y)

    dist_thresh = 1.0 - sim_super
    clusters = Butina.ClusterData(
        fps, len(fps),
        distThresh=dist_thresh,
        isDistData=False,
        distFunc=tanimoto_dist,
        reordering=True
    )
    clusters = [list(c) for c in clusters if len(c) >= min_super]

    labels = np.full(len(smiles_screen), -1, dtype=int)
    for cid, idxs in enumerate(clusters):
        for i in idxs:
            labels[i] = cid
    return labels


def to_bool_binary(arr):
    """Convert input to boolean matrix (nonzero -> True)."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.dtype != np.bool_:
        return arr != 0
    return arr


def intra_cluster_average_link(x_screen, super_cluster_labels, tani_cutoffs):
    """
    Average-link clustering within each super-cluster.
    Note: tani_cutoffs are distance thresholds (e.g. 0.6, 0.4).
    """
    x_screen = to_bool_binary(x_screen)  # Ensure boolean matrix for Jaccard distance
    n_samples = x_screen.shape[0]
    final_clusters = np.full((n_samples, len(tani_cutoffs)), -1, dtype=int)

    unique_super_clusters = np.unique(super_cluster_labels)
    unique_super_clusters = unique_super_clusters[unique_super_clusters >= 0]

    print(f"Clustering within {len(unique_super_clusters)} super-clusters")

    for super_cluster_id in unique_super_clusters:
        cluster_indices = np.where(super_cluster_labels == super_cluster_id)[0]
        cluster_size = len(cluster_indices)
        print(f"Processing super-cluster {super_cluster_id}, size: {cluster_size}")

        if cluster_size < 2:
            # Singleton cluster: assign same label for all cutoffs
            for idx in cluster_indices:
                final_clusters[idx, :] = super_cluster_id * 1000  # arbitrary fixed ID
            continue

        X = x_screen[cluster_indices].astype(bool)

        # Jaccard condensed distance vector
        y = pdist(X, metric='jaccard')  # skip squareform

        # Average linkage, pass y directly
        Z = hierarchy.linkage(y, method='average')

        for cutoff_idx, cutoff in enumerate(tani_cutoffs):
            sub_labels = hierarchy.cut_tree(Z, height=cutoff)[:, 0]  # shape (m, 1) → (m,)
            # Map sub-cluster labels back to original indices, offset by super_cluster_id*1000 to avoid conflicts
            for i_local, sub_id in enumerate(sub_labels):
                final_clusters[cluster_indices[i_local], cutoff_idx] = super_cluster_id * 1000 + int(sub_id)

    # Unclustered (-1) stays -1
    return final_clusters





if __name__ == '__main__':

    # Process the data
    SCREEN_RATIO = 5 / 6  # 5:1 screen-to-test split
    for dataset in ['MAPK1', 'MTORC1', 'VDR']:
    # for dataset in ['PKM2']:

        df = get_data(dataset=dataset)
        n = len(df)
        screen_size = int(n * SCREEN_RATIO)
        test_size = n - screen_size
        df_screen, df_test = split_data(df, screen_size=screen_size, test_size=test_size, dataset=dataset)

        MasterDataset(name='screen', df=df_screen, overwrite=True, dataset=dataset)
        MasterDataset(name='test', df=df_test, overwrite=True, dataset=dataset)

        df_screen = pd.read_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/screen.csv'))
        df_test = pd.read_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/test.csv'))

        similarity_vectors(df_screen, df_test, dataset=dataset)

    # Perform clustering for each dataset
    # for dataset, tani_cutoffs in zip(['CYP3A4', 'PKM2', 'TP53'], [[0.8, 0.6], [0.2, 0.4], [0.8, 0.6]]):
    # ==== Utility: checkpoint/resume ==== 
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def ckpt_exists(path):
        return os.path.exists(path)

    def save_ckpt(obj, path):
        ensure_dir(os.path.dirname(path))
        torch.save(obj, path)

    def load_ckpt(path):
        return torch.load(path)

    for dataset, tani_cutoffs in zip(['PPARG', 'ESR1_ant'], [[0.8, 0.6], [0.8, 0.6]]):
        ds_screen = MasterDataset('screen', representation='ecfp', dataset=dataset)
        x_screen, y_screen, smiles_screen = ds_screen.all()
        smiles_index = torch.load(f'data/{dataset}/screen/smiles_index')
        min_supercluster_size = 64
        min_subcluster_size = 16
        n = len(smiles_screen)
        subcluster_mu = np.mean(y_screen.tolist()) * min_subcluster_size
        subcluster_sigma = np.std(y_screen.tolist()) * np.sqrt(min_subcluster_size)
        # PKM2: memory-optimized two-stage clustering
        if dataset == 'PKM2':
            print("Starting memory-optimized two-stage clustering...")

            # === Butina super-clusters (checkpoint) ===
            super_labels_path = f'data/{dataset}/screen/super_cluster_labels.pt'
            if ckpt_exists(super_labels_path):
                super_cluster_labels = load_ckpt(super_labels_path)
                print("[ckpt] Loaded super-cluster labels")
            else:
                super_cluster_threshold = 0.2   # Coarse similarity threshold
                min_supercluster_size_adjusted = 32
                super_cluster_labels = build_supercluster_labels_butina(
                    smiles_screen,
                    sim_super=super_cluster_threshold,
                    min_super=min_supercluster_size_adjusted
                )
                save_ckpt(super_cluster_labels, super_labels_path)
                print("[ckpt] Saved super-cluster labels")

            # === Stage 2: average-link cut tree (checkpoint) ===
            cut_clusters_path = f'data/{dataset}/screen/cut_clusters.pt'
            if ckpt_exists(cut_clusters_path):
                cut_clusters = load_ckpt(cut_clusters_path)
                print("[ckpt] Loaded cut_clusters")
            else:
                cut_clusters = intra_cluster_average_link(x_screen, super_cluster_labels, tani_cutoffs)
                save_ckpt(cut_clusters, cut_clusters_path)
                print("[ckpt] Saved cut_clusters")

            print("Two-stage clustering complete")

            # Diagnostics
            num_super = np.sum(super_cluster_labels >= 0)
            print(f"Super-clusters: {len(np.unique(super_cluster_labels[super_cluster_labels >= 0]))}")
            print(f"Unclustered (-1) count: {np.sum(super_cluster_labels == -1)}")
            print(f"Molecules in super-clusters: {num_super}")
            
        else:
            D = load_hdf5(f'data/{dataset}/screen/tanimoto_distance_vector')
            linkage = hierarchy.average(D)
            del D
            cut_clusters = hierarchy.cut_tree(linkage, height=tani_cutoffs)
        # cut = np.concatenate((np.array([range(n)]).T, cut_clusters), axis=1)

        # # Find the big superclusters
        # super_clusters = [clust for clust, cnt in Counter(cut[:, 1]).items() if cnt >= min_supercluster_size]
        # cut = cut[[True if i in super_clusters else False for i in cut[:, 1]]]
        cut_path = f'data/{dataset}/screen/cut.pt'
        sub_clusters_path = f'data/{dataset}/screen/sub_clusters.pt'
        super_clusters_path = f'data/{dataset}/screen/super_clusters.pt'
        if ckpt_exists(cut_path) and ckpt_exists(sub_clusters_path) and ckpt_exists(super_clusters_path):
            cut = load_ckpt(cut_path)
            sub_clusters = load_ckpt(sub_clusters_path)
            super_clusters = load_ckpt(super_clusters_path)
            print("[ckpt] Loaded cut, sub_clusters, super_clusters")
        else:
            cut_all = np.concatenate((np.arange(n)[:, None], cut_clusters), axis=1)

            # Filter out unclustered (-1)
            valid_mask = (cut_all[:, 1] >= 0)
            cut = cut_all[valid_mask]

            # Find the big superclusters
            super_clusters = [clust for clust, cnt in Counter(cut[:, 1]).items() if cnt >= min_supercluster_size]
            cut = cut[np.isin(cut[:, 1], super_clusters)]

            # find the subclusters
            sub_clusters = [clust for clust, cnt in Counter(cut[:, -1]).items() if cnt >= min_subcluster_size]

            save_ckpt(cut, cut_path)
            save_ckpt(sub_clusters, sub_clusters_path)
            save_ckpt(super_clusters, super_clusters_path)
            print("[ckpt] Saved cut, sub_clusters, super_clusters")

        starting_clusters_path = f'data/{dataset}/screen/starting_clusters'
        if ckpt_exists(starting_clusters_path):
            cluster_smiles_with_hits = torch.load(starting_clusters_path)
            print("[ckpt] Loaded starting_clusters")
            # Still need cluster0_idx for downstream mapping
            final_assigned_idx = set(cut[:, 0].astype(int))
            all_idx = set(range(n))
            cluster0_idx = np.array(sorted(all_idx - final_assigned_idx))
        else:
            # put the subclusters and superclusters together
            cluster_smiles = []
            for sub_clust in sub_clusters:
                super_clust = cut[:, 1][cut[:, 2] == sub_clust][0]
                cluster_smiles.append([
                    smiles_screen[cut[:, 0][np.where(cut[:, 2] == sub_clust)]],
                    smiles_screen[cut[:, 0][np.where(cut[:, 1] == super_clust)]]
                ])
            # Keep as list to avoid dimension mismatch when stacking object arrays
            cluster_smiles_list = cluster_smiles
            # === Build Cluster 0 ===
            final_assigned_idx = set(cut[:, 0].astype(int))   # All samples assigned to valid clusters
            all_idx = set(range(n))
            cluster0_idx = np.array(sorted(all_idx - final_assigned_idx))
            cluster0_smiles = smiles_screen[cluster0_idx]

            if cluster0_idx.size > 0:
                # Insert first row at list stage, convert to 2D object array later
                cluster_smiles_list = [[cluster0_smiles, cluster0_smiles]] + cluster_smiles_list

            # Convert to 2D object array: (num_clusters, 2)
            cluster_smiles_with_hits = np.array(cluster_smiles_list, dtype=object)

            for i in cluster_smiles_with_hits:
                print(len(i[0]), len(i[1]), len(i[0]) / len(i[1]))

            only_child = []
            for i in range(len(cluster_smiles_with_hits)):
                supercluster = cluster_smiles_with_hits[i][1]
                subcluster = cluster_smiles_with_hits[i][0]

                contains = 0
                if len(np.intersect1d(supercluster, subcluster)) > 0:
                    contains = 1

                for j in range(len(cluster_smiles_with_hits)):
                    if i != j and len(np.intersect1d(cluster_smiles_with_hits[j][1], subcluster)) > 0:
                        contains += 1

                only_child.append(contains)

            cluster_smiles_with_hits = cluster_smiles_with_hits[np.where(np.array(only_child) == 1)]
            print(len(cluster_smiles_with_hits))

            torch.save(cluster_smiles_with_hits, starting_clusters_path)
            print("[ckpt] Saved starting_clusters")


        starting_clusters = torch.load(f'data/{dataset}/screen/starting_clusters')
        mol_to_supercluster = {}
        for idx, supercluster_id in zip(cut[:, 0], cut[:, 1]):
            smi = smiles_screen[idx]
            mol_to_supercluster[smi] = int(supercluster_id)
        
        # # Mark unclustered molecules as cluster -1
        # all_smiles_set = set(smiles_screen)
        # clustered_smiles_set = set([smiles_screen[idx] for idx in cut[:, 0]])
        # unclustered_smiles = all_smiles_set - clustered_smiles_set
        # for smi in unclustered_smiles:
        #     mol_to_supercluster[smi] = -1

        # 1) Build old_id -> new_id mapping for retained super-clusters (starting from 1)
        uniq_supers = np.unique(cut[:, 1]).astype(int)
        super_id_remap = {old_id: new_id for new_id, old_id in enumerate(uniq_supers, start=1)}

        # 2) Build molecule -> cluster mapping
        mol_to_supercluster = {}

        # Valid super-clusters (renumbered 1..K)
        for idx_int, super_id_old in zip(cut[:, 0].astype(int), cut[:, 1].astype(int)):
            smi = smiles_screen[idx_int]
            mol_to_supercluster[smi] = int(super_id_remap[super_id_old])

        # Cluster 0 (unclustered)
        for idx_int in cluster0_idx:
            smi = smiles_screen[idx_int]
            mol_to_supercluster[smi] = 0


        mol2super_path = f'data/{dataset}/screen/mol_to_supercluster'
        if ckpt_exists(mol2super_path):
            print("[ckpt] mol_to_supercluster exists, skipping")
        else:
            torch.save(mol_to_supercluster, mol2super_path)
            print("[ckpt] Saved mol_to_supercluster")

        print('Total molecules:', len(smiles_screen))
        print('Cluster 0 unclustered count:', len(cluster0_idx))
        print('After super-cluster filter:', len(super_clusters), ' count:', sum([cnt for clust, cnt in Counter(cut[:, 1]).items() if cnt >= min_supercluster_size]))
        print('After sub-cluster filter:', len(sub_clusters), ' count:', sum([cnt for clust, cnt in Counter(cut[:, -1]).items() if cnt >= min_subcluster_size]))
        print('Final clustered:', len(mol_to_supercluster))