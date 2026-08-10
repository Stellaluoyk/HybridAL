"""

This script contains the main active learning loop that runs all experiments.

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

# ============================================================================
# Modifications by Yukun Luo:
# - Added per-cycle test set prediction probability export to CSV
# - Added CKA (Centered Kernel Alignment) analysis for ensemble feature
#   consistency (compute_cosine_consistency, compute_classwise_cka)
# - Added training loss logging to JSON
# - Per-cycle test precision tracking for adaptive strategies
# ============================================================================

from math import ceil
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import WeightedRandomSampler
from active_learning.nn import Ensemble, RfEnsemble
from active_learning.data_prep import MasterDataset
from active_learning.data_handler import Handler
from active_learning.utils import Evaluate, to_torch_dataloader
from active_learning.utils import compute_cosine_consistency, compute_classwise_cka
from active_learning.acquisition import Acquisition, logits_to_pred
from active_learning.utils import molecular_graph_featurizer, visualize_ensemble_umap
import os
import torch_geometric
import random
from datetime import datetime
import imageio
import glob

INFERENCE_BATCH_SIZE = 512
TRAINING_BATCH_SIZE = 64

def active_learning(n_start: int = 64, acquisition_method: str = 'uncertainty', max_screen_size: int = 1088,
                    batch_size: int = 32, architecture: str = 'mlg', seed: int = 0, bias: str = 'random',
                    optimize_hyperparameters: bool = False, ensemble_size: int = 10, retrain: bool = True,
                    anchored: bool = True, dataset: str = 'CYP3A4', scrambledx: bool = False,
                    scrambledx_seed: int = 1, output_dir: str = 'results',
                    init_pos_count: int = None) -> pd.DataFrame:
    """
    :param n_start: number of molecules to start out with
    :param acquisition_method: acquisition method, as defined in active_learning.acquisition
    :param max_screen_size: we stop when this number of molecules has been screened
    :param batch_size: number of molecules to add every cycle
    :param architecture: 'gcn', 'mlp', or 'rf'
    :param seed: int 1-20
    :param bias: 'random', 'small', 'large'
    :param optimize_hyperparameters: Bool
    :param ensemble_size: number of models in the ensemble, default is 10
    :param scrambledx: toggles randomizing the features
    :param scrambledx_seed: seed for scrambling the features
    :return: dataframe with results
    """
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Load the datasets
    representation = 'ecfp' if architecture in ['mlp', 'rf'] else 'graph'
    ds_screen = MasterDataset('screen', representation=representation, dataset=dataset, scramble_x=scrambledx,
                              scramble_x_seed=scrambledx_seed)
    ds_test = MasterDataset('test', representation=representation, dataset=dataset)
    #print(type(ds_screen))
    # Initiate evaluation trackers
    eval_test, eval_screen, eval_train = Evaluate(), Evaluate(), Evaluate()
    handler = Handler(
        n_start=n_start,
        seed=seed,
        bias=bias,
        dataset=dataset,
        init_pos_count=init_pos_count
    )

    # Define some variables
    hits_discovered, total_mols_screened, all_train_smiles = [], [], []
    max_screen_size = len(ds_screen) if max_screen_size is None else max_screen_size
    
    
    previous_hits_count = 0  

    # build test loader
    x_test, y_test, smiles_test = ds_test.all()
    test_loader = to_torch_dataloader(x_test, y_test,
                                      batch_size=INFERENCE_BATCH_SIZE,
                                      shuffle=False, pin_memory=True)

    n_cycles = ceil((max_screen_size - n_start) / batch_size)
    # exploration_factor = 1 / lambd^x. To achieve a factor of 1 at the last cycle: lambd = 1 / nth root of 2
    lambd = 1 / (2 ** (1/n_cycles))

    ACQ = Acquisition(method=acquisition_method, seed=seed, lambd=lambd)
    all_cycle_features = []
    all_loss_dicts = []
    # While max_screen_size has not been achieved, do some active learning in cycles
    for cycle in tqdm(range(n_cycles+1)):

        # Get the train and screen data for this cycle
        train_idx, screen_idx = handler()
        x_train, y_train, smiles_train = ds_screen[train_idx]
        x_screen, y_screen, smiles_screen = ds_screen[screen_idx]

        # Update some tracking variables
        all_train_smiles.append(','.join(smiles_train.tolist()))
        hits_discovered.append(sum(y_train).item())
        
        
        
        
        # previous_hits = hits_discovered[cycle-1] if cycle > 0 else 0
        # new_hits_this_cycle = current_hits - previous_hits
        
        
        # new_hits_for_dyal = new_hits_this_cycle

        hits = smiles_train[np.where(y_train == 1)]
        nonhits = smiles_train[np.where(y_train == 0)]
        hits_idx = np.where(y_train == 1)[0].tolist()
        nonhits_idx = np.where(y_train == 0)[0].tolist()
        total_mols_screened.append(len(y_train))

        if len(train_idx) >= max_screen_size:
            break
        if architecture != 'rf':
            # If only one class in y_train, do not use sampler
            unique_classes = set(y_train.tolist()) if hasattr(y_train, 'tolist') else set(y_train)
            use_sampler = len(unique_classes) > 1

            train_loader = to_torch_dataloader(x_train, y_train,
                                               batch_size=INFERENCE_BATCH_SIZE,
                                               shuffle=False, pin_memory=True)

            if use_sampler:
                # Get class weight to build a weighted random sampler to balance out this data
                class_weights = [1 - sum((y_train == 0) * 1) / len(y_train), 1 - sum((y_train == 1) * 1) / len(y_train)]
                weights = [class_weights[i] for i in y_train]
                sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)
                train_loader_balanced = to_torch_dataloader(x_train, y_train,
                                                           batch_size=TRAINING_BATCH_SIZE,
                                                           sampler=sampler,
                                                           shuffle=False, pin_memory=True)
            else:
                train_loader_balanced = to_torch_dataloader(x_train, y_train,
                                                           batch_size=TRAINING_BATCH_SIZE,
                                                           shuffle=True, pin_memory=True)

            screen_loader = to_torch_dataloader(x_screen, y_screen,
                                                batch_size=INFERENCE_BATCH_SIZE,
                                                shuffle=False, pin_memory=True)

            # Initiate and train the model (optimize if specified)
            print("Training model")
            if retrain or cycle == 0:
                M = Ensemble(seed=seed, ensemble_size=ensemble_size, architecture=architecture, anchored=anchored)
                print(M)
                if cycle == 0 and optimize_hyperparameters:
                    M.optimize_hyperparameters(x_train, y_train)
                train_result = M.train(train_loader_balanced, verbose=False)
                loss_dict = train_result.get('loss', {})

            # Do inference of the train/test/screen data
            print("Train/test/screen inference")
            train_logits_N_K_C = M.predict(train_loader)
            eval_train.eval(train_logits_N_K_C, y_train)

            test_logits_N_K_C = M.predict(test_loader)
            eval_test.eval(test_logits_N_K_C, y_test)
            # Export test set positive class prediction probabilities per cycle
            test_probs_N_C = logits_to_pred(test_logits_N_K_C, return_prob=True, return_uncertainty=False)
            test_pos_probs = test_probs_N_C[:, 1].detach().cpu().numpy()
            test_pred_dir = os.path.join(output_dir, f"test_predictions/test_{dataset}")
            os.makedirs(test_pred_dir, exist_ok=True)
            test_pred_filename = (
                f"test_probs_{dataset}_{acquisition_method}_batch{batch_size}"
                f"seed{seed}_cycle{cycle}_{timestamp}.csv"
            )
            test_pred_path = os.path.join(test_pred_dir, test_pred_filename)
            df_test_probs = pd.DataFrame({
                "smiles": smiles_test,
                "probability": test_pos_probs
            })
            df_test_probs.to_csv(test_pred_path, index=False)

            # Track current cycle test precision
            current_test_precision = eval_test.precision[-1]

            screen_logits_N_K_C = M.predict(screen_loader)
            eval_screen.eval(screen_logits_N_K_C, y_screen)
            # # Extract ensemble features and compute consistency metrics
            # try:
            #     test_features_N_K_D = M.predict_features(test_loader)  # [N,K,D]
            #     # Per-sample cosine consistency
            #     cosine_consistency = compute_cosine_consistency(test_features_N_K_D)
            #     # Class-wise CKA (positive/negative)
            #     labels_tensor = y_test if isinstance(y_test, torch.Tensor) else torch.as_tensor(y_test)
            #     labels_tensor = labels_tensor.view(-1).to(test_features_N_K_D.device)
            #     classwise_cka = compute_classwise_cka(test_features_N_K_D, labels_tensor)
            #     # Append results to eval_test for later CSV export
            #     # Save per-sample file
            #     df_cos = pd.DataFrame({
            #         'cosine_consistency': cosine_consistency
            #     })
            #     cos_dir = os.path.join(output_dir, f"consistency/consistency_{dataset}")
            #     os.makedirs(cos_dir, exist_ok=True)
            #     # df_cos.to_csv(os.path.join(cos_dir, f"test_cosine_cycle{cycle}_seed{seed}.csv"), index=False)

            #     # Compute mean cosine for positive and negative samples
            #     if isinstance(y_test, torch.Tensor):
            #         y_np = y_test.cpu().numpy()
            #     else:
            #         y_np = np.array(y_test)
            #     mask_neg = (y_np == 0)
            #     mask_pos = (y_np == 1)
            #     mean_cos_neg = float(np.mean(cosine_consistency[mask_neg])) if mask_neg.any() else float('nan')
            #     mean_cos_pos = float(np.mean(cosine_consistency[mask_pos])) if mask_pos.any() else float('nan')

            #     # Compute gradient norm statistics
            #     # grad_norm_pos = 0.0
            #     # grad_norm_neg = 0.0
            #     # if loss_norms_dict:
            #     #     # Average gradient norms across all models
            #     #     pos_norms = []
            #     #     neg_norms = []
            #     #     for model_id, norms in loss_norms_dict.items():
            #     #         if 'pos' in norms and norms['pos']:
            #     #             pos_norms.extend(norms['pos'])
            #     #         if 'neg' in norms and norms['neg']:
            #     #             neg_norms.extend(norms['neg'])
            #     #     grad_norm_pos = float(np.mean(pos_norms)) if pos_norms else 0.0
            #     #     grad_norm_neg = float(np.mean(neg_norms)) if neg_norms else 0.0

            #     # Save class-wise CKA, mean cosine, and gradient norms
            #     df_cka = pd.DataFrame([{ 
            #         'cycle': cycle,
            #         'cka_pos': classwise_cka.get('pos', 0.0),
            #         'cka_neg': classwise_cka.get('neg', 0.0), 
            #         'mean_cosine_pos': mean_cos_pos,
            #         'mean_cosine_neg': mean_cos_neg,
            #         # 'grad_norm_pos': grad_norm_pos,
            #         # 'grad_norm_neg': grad_norm_neg,
            #     }])
            #     cka_path = os.path.join(cos_dir, f"test_mean_cka_cosine_seed{seed}.csv")
            #     cka_header = not os.path.exists(cka_path)
            #     df_cka.to_csv(cka_path, mode='a', index=False, header=cka_header)
            # except Exception as e:
            #     print(f"Feature consistency metrics failed: {e}")
            # Save model parameters
            # # model_dir = os.path.join(output_dir, f"model/model_{dataset}")
            # # os.makedirs(model_dir, exist_ok=True)
            # # torch.save(M.state_dict(), os.path.join(model_dir, f"{dataset}_{acquisition_method}_batch{batch_size}_seed{seed}_cycle_{cycle}_{timestamp}.pth"))

            # all_loss_dicts.append({'cycle': cycle, 'loss': loss_dict})

        else:
            print("Training model")
            if retrain or cycle == 0:
                M = RfEnsemble(seed=seed, ensemble_size=ensemble_size)
                print(M)
                # if cycle == 0 and optimize_hyperparameters:
                #     M.optimize_hyperparameters(x_train, y_train)
                M.train(x_train, y_train, verbose=False)

            # Do inference of the train/test/screen data
            print("Train/test/screen inference")
            train_logits_N_K_C = M.predict(x_train)
            eval_train.eval(train_logits_N_K_C, y_train)

            test_logits_N_K_C = M.predict(x_test)
            eval_test.eval(test_logits_N_K_C, y_test)
            # Export test set positive class prediction probabilities per cycle
            test_probs_N_C = logits_to_pred(test_logits_N_K_C, return_prob=True, return_uncertainty=False)
            test_pos_probs = test_probs_N_C[:, 1].detach().cpu().numpy()
            test_pred_dir = os.path.join(output_dir, f"test_predictions/test_{dataset}")
            os.makedirs(test_pred_dir, exist_ok=True)
            test_pred_filename = (
                f"test_probs_{dataset}_{acquisition_method}_batch{batch_size}_"
                f"seed{seed}_cycle{cycle}_{timestamp}.csv"
            )
            test_pred_path = os.path.join(test_pred_dir, test_pred_filename)
            df_test_probs = pd.DataFrame({
                "smiles": smiles_test,
                "probability": test_pos_probs
            })
            df_test_probs.to_csv(test_pred_path, index=False)
            
            # Track current cycle test precision
            current_test_precision = eval_test.precision[-1]

            screen_logits_N_K_C = M.predict(x_screen)
            eval_screen.eval(screen_logits_N_K_C, y_screen)
                
        # If this is the second to last cycle, update the batch size, so we end at max_screen_size
        if len(train_idx) + batch_size > max_screen_size:
            batch_size = max_screen_size - len(train_idx)

        # Select the molecules to add for the next cycle
        print("Sample acquisition")
        picks = ACQ.acquire(ds_screen, cycle, seed, ensemble_size, architecture, INFERENCE_BATCH_SIZE, dataset, screen_logits_N_K_C, smiles_screen, hits=hits, nonhits=nonhits,
                            n=batch_size, timestamp=timestamp, output_dir=output_dir, hits_discovered=hits_discovered[-1], test_precision=current_test_precision
                            )
        handler.add(picks)
        
        # 
        

    import json
    log_dir = os.path.join(output_dir, f"logs/logs_{dataset}")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"loss_{dataset}_{architecture}_{acquisition_method}_batch{batch_size}_seed{seed}_{bias}_{timestamp}.json"), "w") as f:
        json.dump(all_loss_dicts, f, indent=2)
    # Add all results to a dataframe
    train_results = eval_train.to_dataframe("train_")
    test_results = eval_test.to_dataframe("test_")
    screen_results = eval_screen.to_dataframe('screen_')
    results = pd.concat([train_results, test_results, screen_results], axis=1)
    results['hits_discovered'] = hits_discovered
    results['total_mols_screened'] = total_mols_screened
    results['all_train_smiles'] = all_train_smiles

    return results
