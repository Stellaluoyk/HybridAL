"""

This script contains all models:

    - MLP: a simple feed forward multi-layer perceptron. Supports weight anchoring - Pearce et al. (2018)
    - GCN: a simple graph convolutional NN - Kipf & Welling (2016). Supports weight anchoring - Pearce et al. (2018)
    - Model: A wrapper class that contains a train(), and predict() loop
    - Ensemble: Class that ensembles n Model classes. Contains a train() method and an predict() method that outputs
        logits_N_K_C, defined as [N, num_inference_samples, num_classes]. Also has an optimize_hyperparameters() method.

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm, GATConv, GINConv, NNConv
from tqdm.auto import trange
from sklearn.ensemble import RandomForestClassifier
from active_learning.hyperopt import optimize_hyperparameters


class MLP(torch.nn.Module):
    def __init__(self, in_feats: int = 1024, n_hidden: int = 1024, n_out: int = 2, n_layers: int = 3, seed: int = 42,
                 lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4,
                 weight_decay: float = 0):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        torch.manual_seed(seed)

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(in_feats if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def get_features(self, x: Tensor) -> Tensor:
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)
        return x  # Return hidden layer features
    
    def forward(self, x: Tensor) -> Tensor:
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 1024, num_conv_layers: int = 5, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 3, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GCNConv(n_hidden, n_hidden))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()
    
    def get_features(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = F.elu(self.atom_embedding(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)
        return x  # Return pooled global features

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x

class GAT(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 1024, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 3, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GATConv(n_hidden, n_hidden, add_self_loops=True, negative_slope=0.2,
                                      heads=8, concat=False))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 1024, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 3, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        SimpleMLP = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden))

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GINConv(nn=SimpleMLP))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x


class Model(torch.nn.Module):
    def __init__(self, architecture: str, **kwargs):
        super().__init__()
        assert architecture in ['gcn', 'mlp', 'gat', 'gin']
        self.architecture = architecture
        if architecture == 'mlp':
            self.model = MLP(**kwargs)
        elif architecture == 'gcn':
            self.model = GCN(**kwargs)
        elif architecture == 'gin':
            self.model = GIN(**kwargs)
        else:
            self.model = GAT(**kwargs)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device_type = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(self.device_type)
        self.loss_fn = torch.nn.NLLLoss()

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr,
                                          weight_decay=self.model.weight_decay)

        # Save initial weights in the model for the anchored regularization and move them to the gpu
        if self.model.anchored:
            self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
            self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> dict:

        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        scaler = torch.cuda.amp.GradScaler()
        
        # Track loss norms (approximated via loss ratio)
        loss_norms = {'pos': [], 'neg': []}

        for _ in bar:
            running_loss = 0
            items = 0
            epoch_loss_norms_pos = []
            epoch_loss_norms_neg = []

            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):

                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y = batch.y
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x, y = batch[0].to(self.device), batch[1].to(self.device)
                        y_hat = self.model(x)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    
                    # Compute loss for positive and negative classes separately
                    y_squeezed = y.squeeze()
                    pos_mask = (y_squeezed == 1)
                    neg_mask = (y_squeezed == 0)
                    
                    loss_pos = None
                    loss_neg = None
                    
                    if pos_mask.any():
                        loss_pos = self.loss_fn(y_hat[pos_mask], y_squeezed[pos_mask])
                    if neg_mask.any():
                        loss_neg = self.loss_fn(y_hat[neg_mask], y_squeezed[neg_mask])
                    
                    # Compute exact gradient norms (before optimization)
                    def grad_norm_of(loss_term):
                        if loss_term is None:
                            return 0.0
                        grads = torch.autograd.grad(
                            loss_term, 
                            [p for p in self.model.parameters() if p.requires_grad],
                            retain_graph=True,            # will backward total loss later
                            allow_unused=True, 
                            create_graph=False
                        )
                        total = 0.0
                        for g in grads:
                            if g is not None:
                                total += g.detach().float().pow(2).sum()
                        return float(total.sqrt().item())

                    # if idx == 0:  # only compute on first batch to avoid redundancy
                    #     pos_gnorm = grad_norm_of(loss_pos) if pos_mask.any() else 0.0
                    #     neg_gnorm = grad_norm_of(loss_neg) if neg_mask.any() else 0.0
                        
                    #     epoch_loss_norms_pos.append(pos_gnorm)
                    #     epoch_loss_norms_neg.append(neg_gnorm)
                    
                    # Compute total loss and optimize
                    loss = self.loss_fn(y_hat, y_squeezed)

                    if self.model.anchored:
                        # Calculate the total anchored L2 loss
                        l2_loss = 0
                        for param_name, params in self.model.named_parameters():
                            anchored_param = self.model.anchor_weights[param_name]

                            l2_loss += (self.model.l2_lambda / len(y)) * torch.mul(params - anchored_param,
                                                                                   params - anchored_param).sum()

                        # Add anchored loss to regular loss according to Pearce et al. (2018)
                        loss = loss + l2_loss

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)
            
            # Log loss norms for this epoch
            if epoch_loss_norms_pos:
                loss_norms['pos'].append(np.mean(epoch_loss_norms_pos))
            if epoch_loss_norms_neg:
                loss_norms['neg'].append(np.mean(epoch_loss_norms_neg))
            
            self.epoch += 1
        
        return {'train_loss': self.train_loss, 'loss_norms': loss_norms}

    def predict(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats


class Ensemble(torch.nn.Module):

    def __init__(self, ensemble_size: int = 10, seed: int = 0, architecture: str = 'mlp', **kwargs) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, ensemble_size)
        # self.models = {i: Model(seed=s, architecture=architecture, **kwargs) for i, s in enumerate(self.seeds)}
        self.models = torch.nn.ModuleDict({
                    str(i): Model(seed=s, architecture=architecture, **kwargs) for i, s in enumerate(self.seeds)
                })
    def get_all_features(self, dataloader: DataLoader) -> Tensor:
        """Extract hidden layer features for all samples"""
        features_list = []
        with torch.no_grad():
            for batch in dataloader:
                for model in self.models.values():
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch = batch.to(model.device)
                        feat = model.model.get_features(batch.x.float(), 
                                                      batch.edge_index,
                                                      batch.batch)
                    else:
                        x = batch[0].to(model.device)
                        feat = model.model.get_features(x)
                    features_list.append(feat.cpu())
        return torch.cat(features_list, dim=0)

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader, **kwargs) -> dict:
        loss_dict = {}
        loss_norms_dict = {}
        for i, m in self.models.items():
            result = m.train(dataloader, **kwargs)  # result for each sub-model
            if isinstance(result, dict):
                loss_dict[i] = result.get('train_loss', [])
                loss_norms_dict[i] = result.get('loss_norms', {'pos': [], 'neg': []})
            else:
                loss_dict[i] = result
                loss_norms_dict[i] = {'pos': [], 'neg': []}
        return {'loss': loss_dict, 'loss_norms': loss_norms_dict}

    def predict(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C

    def predict_features(self, dataloader: DataLoader) -> Tensor:
        """Return hidden features for each sample and each ensemble member.

        Output shape: [N, K, D]
        - N: number of samples in dataloader (in-order across batches)
        - K: ensemble size
        - D: feature dimension from model.model.get_features
        """
        # First pass with model "0" to determine total N and feature dim D
        model_keys = list(self.models.keys())
        assert len(model_keys) > 0, "Ensemble has no models"

        # Collect per-model features as a list of [N, D]
        per_model_features = []
        for key in model_keys:
            m = self.models[key]
            feats_list = []
            with torch.no_grad():
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch = batch.to(m.device)
                        feat = m.model.get_features(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(m.device)
                        feat = m.model.get_features(x)
                    feats_list.append(feat.cpu())
            per_model_features.append(torch.cat(feats_list, dim=0))  # [N, D]

        # Stack along K -> [K, N, D] then permute to [N, K, D]
        feats_K_N_D = torch.stack(per_model_features, dim=0)
        feats_N_K_D = feats_K_N_D.permute(1, 0, 2).contiguous()
        return feats_N_K_D

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"

class RfEnsemble():
    """ Ensemble of RFs"""
    def __init__(self, ensemble_size: int = 10, seed: int = 0, **kwargs) -> None:
        self.ensemble_size = ensemble_size
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, ensemble_size)
        self.models = {i: RandomForestClassifier(random_state=s, class_weight="balanced", **kwargs) for i, s in enumerate(self.seeds)}

    def train(self, x, y, **kwargs) -> None:
        for i, m in self.models.items():
            m.fit(x, y)

    def predict(self, x, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        # logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)
        eps = 1e-10  # we need to add this so we don't get divide by zero errors in our log function

        y_hats = []
        for m in self.models.values():

            y_hat = torch.tensor(m.predict_proba(x) + eps)
            if y_hat.shape[1] == 1:  # if only one class if predicted with the RF model, add a column of zeros
                y_hat = torch.cat((y_hat, torch.zeros((y_hat.shape[0], 1))), dim=1)
            y_hats.append(y_hat)

        logits_N_K_C = torch.stack(y_hats, 1)

        logits_N_K_C = torch.log(logits_N_K_C)

        return logits_N_K_C

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} RF Classifiers"
