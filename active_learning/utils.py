from typing import Union, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, average_precision_score, matthews_corrcoef
import umap.umap_ as umap
import matplotlib.pyplot as plt

structural_smarts = {
    # chirality
    "Specified chiral carbon": "[$([#6X4@](*)(*)(*)*),$([#6X4@H](*)(*)*)]",  # Matches carbons whose chirality is specified (clockwise or anticlockwise) Will not match molecules whose chirality is unspecified b ut that could otherwise be considered chiral. Also,therefore won't match molecules that would be chiral due to an implicit connection (i.e.i mplicit H).
    # connectivity
    "Quaternary Nitrogen": "[$([NX4+]),$([NX4]=*)]",  # Hits non-aromatic Ns.
    "S double-bonded to Carbon": "[$([SX1]=[#6])]",  # Hits terminal (1-connected S)
    "Triply bonded N": "[$([NX1]#*)]",
    "Divalent Oxygen": "[$([OX2])]",
    # chains and branching
    "Long_chain groups": "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",  # Aliphatic chains at-least 8 members long.
    "Carbon_isolating": "[$([#6+0]);!$(C(F)(F)F);!$(c(:[!c]):[!c])!$([#6]=,#[!#6])]",  # This definition is based on that in CLOGP, so it is a charge-neutral carbon, which is not a CF3 or an aromatic C between two aromati c hetero atoms eg in tetrazole, it is not multiply bonded to a hetero atom.
    # rotation
    "Rotatable bond": "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]",  # An atom which is not triply bonded and not one-connected i.e.terminal connected by a single non-ring bond to and equivalent atom. Note that logical operators can be applied to bonds ("-&!@"). Here, the overall SMARTS consists of two atoms and one bond. The bond is "site and not ring". *#* any atom triple bonded to any atom. By enclosing this SMARTS in parentheses and preceding with $, this enables us to use $(*#*) to write a recursive SMARTS using that string as an atom primitive. The purpose is to avoid bonds such as c1ccccc1-C#C which wo be considered rotatable without this specification.
    # cyclic features
    "Bicyclic": "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",  # Bicyclic compounds have 2 bridgehead atoms with 3 arms connecting the bridgehead atoms.
    "Ortho": "*-!:aa-!:*",  # Ortho-substituted ring
    "Meta": "*-!:aaa-!:*",  # Meta-substituted ring
    "Para": "*-!:aaaa-!:*",  # Para-substituted ring
    "Acylic-bonds": "*!@*",
    "Single bond and not in a ring": "*-!@*",
    "Non-ring atom": "[!R]",
    "Ring atom": "[R]",
    "Macrocycle groups": "[r;!r3;!r4;!r5;!r6;!r7]",
    "S in aromatic 5-ring with lone pair": "[sX2r5]",
    "Aromatic 5-Ring O with Lone Pair": "[oX2r5]",
    # "N in 5-sided aromatic ring": "[nX2r5]",
    "Spiro-ring center": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",  # rings size 4-6
    "N in 5-ring arom": "[$([nX2r5]:[a-]),$([nX2r5]:[a]:[a-])]",  # anion
    "CIS or TRANS double bond in a ring": "*/,\[R]=;@[R]/,\*",  # An isomeric SMARTS consisting of four atoms and three bonds.
    "CIS or TRANS double or aromatic bond in a ring": "*/,\[R]=,:;@[R]/,\*",
    "Unfused benzene ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # To find a benzene ring which is not fused, we write a SMARTS of 6 aromatic carbons in a ring where each atom is only in one ring:
    "Multiple non-fused benzene rings": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "Fused benzene rings": "c12ccccc1cccc2",
}

functional_group_smarts = {
    # carbonyl
    "Carbonyl group": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",  # Hits either resonance structure
    "Aldehyde": "[CX3H1](=O)[#6]",  # -al
    "Amide": "[NX3][CX3](=[OX1])[#6]",  # -amide
    "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",  # Hits carbamic esters, acids, and zwitterions
    "Carboxylate Ion": "[CX3](=O)[O-]",  # Hits conjugate bases of carboxylic, carbamic, and carbonic acids.
    "Carbonic Acid or Carbonic Ester": "[CX3](=[OX1])(O)O",  # Carbonic Acid, Carbonic Ester, or combination
    "Carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
    "Ester Also hits anhydrides": "[#6][CX3](=O)[OX2H0][#6]",  # won't hit formic anhydride.
    "Ketone": "[#6][CX3](=O)[#6]",  # -one
    # ether
    "Ether": "[OD2]([#6])[#6]",
    # hydrogen atoms
    "Mono-Hydrogenated Cation": "[+H]",  # Hits atoms that have a positive charge and exactly one attached hydrogen:  F[C+](F)[H]
    "Not Mono-Hydrogenated": "[!H1]",  # Hits atoms that don't have exactly one attached hydrogen.
    # amide
    "Amidinium": "[NX3][CX3]=[NX3+]",
    "Cyanamide": "[NX3][CX2]#[NX1]",
    # amine
    "Primary or secondary amine, not amide": "[NX3;H2,H1;!$(NC=O)]",  # Not ammonium ion (N must be 3-connected), not ammonia (H count can't be 3). Primary or secondary is specified by N's H-count (H2 &amp; H1 respectively).  Also note that "&amp;" (and) is the dafault opperator and is higher precedence that "," (or), which is higher precedence than ";" (and). Will hit cyanamides and thioamides
    "Enamine": "[NX3][CX3]=[CX3]",
    "Enamine or Aniline Nitrogen": "[NX3][$(C=C),$(cc)]",
    # azo
    "Azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",  # 5 member aromatic heterocycle w/ 2double bonds. contains N &amp; another non C (N,O,S)  subclasses are furo-, thio-, pyrro-  (replace
    # hydrazine
    "Hydrazine H2NNH2": "[NX3][NX3]",
    # hydrazone
    "Hydrazone C=NNH2": "[NX3][NX2]=[*]",
    # imine
    "Substituted imine": "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",  # Schiff base
    "Substituted or un-substituted imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Iminium": "[NX3+]=[CX3]",
    # imide
    "Unsubstituted dicarboximide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "Substituted dicarboximide": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
    # nitrate
    "Nitrate group": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",  # Also hits nitrate anion
    # nitrile
    "Nitrile": "[NX1]#[CX2]",
    # nitro
    "Nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",   #Hits both forms.
    # hydroxyl (includes alcohol, phenol)
    "Hydroxyl": "[OX2H]",
    "Hydroxyl in Alcohol": "[#6][OX2H]",
    "Enol": "[OX2H][#6X3]=[#6]",
    "Phenol": "[OX2H][cX3]:[c]",
    # thio groups (thio-, thi-, sulpho-, marcapto-)
    "Carbo-Thioester": "S([#6])[CX3](=O)[#6]",
    "Thio analog of carbonyl": "[#6X3](=[SX1])([!N])[!N]",  # Where S replaces O.  Not a thioamide.
    "Thiol, Sulfide or Disulfide Sulfur": "[SX2]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    # sulfide
    "Sulfide": "[#16X2H0]",  # -alkylthio  Won't hit thiols. Hits disulfides.
    "Mono-sulfide": "[#16X2H0][!#16]",  # alkylthio- or alkoxy- Won't hit thiols. Won't hit disulfides.
    "Two Sulfides": "[#16X2H0][!#16].[#16X2H0][!#16]",  # Won't hit thiols. Won't hit mono-sulfides. Won't hit disulfides.
    "Sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",  # Hits all sulfones, including heteroatom-substituted sulfones:  sulfonic acid, sulfonate, sulfuric acid mono- &amp; di- esters, sulfamic acid, sulfamate, sulfonamide... Hits Both Depiction Forms.
    "Sulfonamide": "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",  # (sulf drugs)  Won't hit sulfamic acid or sulfamate. Hits Both Depiction Forms.
    # sulfoxide
    "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",  # ( sulfinyl, thionyl ) Analog of carbonyl where S replaces C. Hits all sulfoxides, including heteroatom-substituted sulfoxides, dialkylsulfoxides carbo-sulfoxides, sulfinate, sulfinic acids... Hits Both Depiction Forms. Won't hit sulfones.
    # halide (-halo -fluoro -chloro -bromo -iodo)
    "Any carbon attached to any halogen": "[#6][F,Cl,Br,I]",
    # Halogen
    "Halogen": "[F,Cl,Br,I]",
    # Three_halides groups
    "Three_halides groups": "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",  # Hits SMILES that have three halides.
}


eval_smarts = {
    # chains and branching
    "Long_chain groups": "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",  # Aliphatic chains at-least 8 members long.
    # cyclic features
    "Bicyclic": "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",  # Bicyclic compounds have 2 bridgehead atoms with 3 arms connecting the bridgehead atoms.
    "Ortho": "*-!:aa-!:*",  # Ortho-substituted ring
    "Meta": "*-!:aaa-!:*",  # Meta-substituted ring
    "Para": "*-!:aaaa-!:*",  # Para-substituted ring
    "Macrocycle groups": "[r;!r3;!r4;!r5;!r6;!r7]",
    "Spiro-ring center": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",  # rings size 4-6
    "CIS or TRANS double bond in a ring": "*/,\[R]=;@[R]/,\*",  # An isomeric SMARTS consisting of four atoms and three bonds.
    "Unfused benzene ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # To find a benzene ring which is not fused, we write a SMARTS of 6 aromatic carbons in a ring where each atom is only in one ring:
    "Multiple non-fused benzene rings": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "Fused benzene rings": "c12ccccc1cccc2",
    # Functional groups:
    # carbonyl
    "Carbonyl group": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",  # Hits either resonance structure
    "Aldehyde": "[CX3H1](=O)[#6]",  # -al
    "Amide": "[NX3][CX3](=[OX1])[#6]",  # -amide
    "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",  # Hits carbamic esters, acids, and zwitterions
    "Carboxylate Ion": "[CX3](=O)[O-]",  # Hits conjugate bases of carboxylic, carbamic, and carbonic acids.
    "Carbonic Acid or Carbonic Ester": "[CX3](=[OX1])(O)O",  # Carbonic Acid, Carbonic Ester, or combination
    "Carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
    "Ester Also hits anhydrides": "[#6][CX3](=O)[OX2H0][#6]",  # won't hit formic anhydride.
    "Ketone": "[#6][CX3](=O)[#6]",  # -one
    # ether
    "Ether": "[OD2]([#6])[#6]",
    # hydrogen atoms
    "Mono-Hydrogenated Cation": "[+H]",  # Hits atoms that have a positive charge and exactly one attached hydrogen:  F[C+](F)[H]
    # amide
    "Amidinium": "[NX3][CX3]=[NX3+]",
    "Cyanamide": "[NX3][CX2]#[NX1]",
    # amine
    "Primary or secondary amine, not amide": "[NX3;H2,H1;!$(NC=O)]",  # Not ammonium ion (N must be 3-connected), not ammonia (H count can't be 3). Primary or secondary is specified by N's H-count (H2 &amp; H1 respectively).  Also note that "&amp;" (and) is the dafault opperator and is higher precedence that "," (or), which is higher precedence than ";" (and). Will hit cyanamides and thioamides
    "Enamine or Aniline Nitrogen": "[NX3][$(C=C),$(cc)]",
    # azo
    "Azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",  # 5 member aromatic heterocycle w/ 2double bonds. contains N &amp; another non C (N,O,S)  subclasses are furo-, thio-, pyrro-  (replace
    # hydrazine
    "Hydrazine H2NNH2": "[NX3][NX3]",
    # hydrazone
    "Hydrazone C=NNH2": "[NX3][NX2]=[*]",
    # imine
    "Substituted or un-substituted imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Iminium": "[NX3+]=[CX3]",
    # imide
    "Unsubstituted dicarboximide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "Substituted dicarboximide": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
    # nitrate
    "Nitrate group": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",  # Also hits nitrate anion
    # nitrile
    "Nitrile": "[NX1]#[CX2]",
    # nitro
    "Nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",   #Hits both forms.
    # hydroxyl (includes alcohol, phenol)
    "Hydroxyl": "[OX2H]",
    "Enol": "[OX2H][#6X3]=[#6]",
    "Phenol": "[OX2H][cX3]:[c]",
    # thio groups (thio-, thi-, sulpho-, marcapto-)
    "Carbo-Thioester": "S([#6])[CX3](=O)[#6]",
    "Thiol, Sulfide or Disulfide Sulfur": "[SX2]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    # sulfide
    "Sulfide": "[#16X2H0]",  # -alkylthio  Won't hit thiols. Hits disulfides.
    "Mono-sulfide": "[#16X2H0][!#16]",  # alkylthio- or alkoxy- Won't hit thiols. Won't hit disulfides.
    "Two Sulfides": "[#16X2H0][!#16].[#16X2H0][!#16]",  # Won't hit thiols. Won't hit mono-sulfides. Won't hit disulfides.
    "Sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",  # Hits all sulfones, including heteroatom-substituted sulfones:  sulfonic acid, sulfonate, sulfuric acid mono- &amp; di- esters, sulfamic acid, sulfamate, sulfonamide... Hits Both Depiction Forms.
    "Sulfonamide": "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",  # (sulf drugs)  Won't hit sulfamic acid or sulfamate. Hits Both Depiction Forms.
    # sulfoxide
    "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",  # ( sulfinyl, thionyl ) Analog of carbonyl where S replaces C. Hits all sulfoxides, including heteroatom-substituted sulfoxides, dialkylsulfoxides carbo-sulfoxides, sulfinate, sulfinic acids... Hits Both Depiction Forms. Won't hit sulfones.
    # halide (-halo -fluoro -chloro -bromo -iodo)
    # Halogen
    "Halogen": "[F,Cl,Br,I]",
    # Three_halides groups
    "Three_halides groups": "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",  # Hits SMILES that have three halides.
}


def atom_featurizer(mol, structural_feats: bool = True, functional_feats: bool = True):

    x = []
    for atom in mol.GetAtoms():
        try:
            x_ = atom_props(atom)
        except:
            return mol
        x.append(x_)
    x = torch.tensor(x)

    if structural_feats:
        x_struc = match_patterns(mol, structural_smarts)
        x = torch.cat((x, x_struc), dim=1)

    if functional_feats:
        x_func = match_patterns(mol, functional_group_smarts)
        x = torch.cat((x, x_func), dim=1)

    return x


def check_featurizability(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for atom in mol.GetAtoms():
            try:
                x_ = atom_props(atom)
            except:
                return False
    except:
        return False

    return True


def molecular_graph_featurizer(smiles: str, y=None, structural_feats: bool = True, functional_feats: bool = True):

    y = torch.tensor([y]).to(torch.long)

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # RDKIT Atom featurization
    x = atom_featurizer(mol, structural_feats, functional_feats)

    # Edge featurization
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices += [[i, j], [j, i]]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]

    if torch.isnan(x).any():
        return smiles
        # raise ValueError(f"Featurizing {smiles} gave nan(s)")

    graph = Data(x=x, edge_index=edge_index, smiles=smiles, y=y)

    return graph


def atom_props(atom):

    x = []

    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se']
    symbols = [0] * 12
    symbols[atom_types.index(atom.GetSymbol())] = 1
    x += symbols

    degrees = [0] * 6  # {1, 2, 3, 4, 5, 6}
    degrees[atom.GetDegree()-1] = 1
    x += degrees

    total_degree = [0] * 6  # {1, 2, 3, 4, 5, 6}
    total_degree[atom.GetTotalDegree()-1] = 1
    x += total_degree

    explicit_valance = [0] * 6  # {1, 2, 3, 4, 5, 6}
    explicit_valance[atom.GetExplicitValence()-1] = 1
    x += explicit_valance

    implicit_valence = [0] * 4  # {0, 1, 2, 3}
    implicit_valence[atom.GetImplicitValence()] = 1
    x += implicit_valence

    GetTotalValence = [0] * 6  # {1, 2, 3, 4, 5, 6}
    GetTotalValence[atom.GetImplicitValence()-1] = 1
    x += GetTotalValence

    implicit_Hs = [0] * 4  # {0, 1, 2, 3}
    implicit_Hs[atom.GetNumImplicitHs()] = 1
    x += implicit_Hs

    total_Hs = [0] * 4  # {0, 1, 2, 3}
    total_Hs[atom.GetTotalNumHs()] = 1
    x += total_Hs

    formal_charge = [0] * 5  # {-1, 0, 1, 2, 3}
    formal_charge[atom.GetFormalCharge()+1] = 1
    x += formal_charge

    hybridization = [0] * 6
    possible_hybridizations = ['SP', 'SP2', 'SP3', 'SP2D', 'SP3D', 'SP3D2']
    hybridization[possible_hybridizations.index(atom.GetHybridization().name)] = 1
    x += hybridization

    return x


def match_patterns(mol, smarts: dict) -> Tensor:
    """

    :param mol: RDKIT mol object
    :param smarts: dict: {"name": "SMARTS"}
    :return: torch.Tensor (n_atoms x n_patterns): one hot tensor of functional group membership
    """

    x = torch.zeros(len(smarts), len(mol.GetAtoms()))
    for i, pattern in enumerate(smarts.values()):
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        # collapse tuple of tuple into unique list of atom indices
        atoms = list(set(sum(matches, ())))
        x[i][atoms] = 1

    return x.T


def smiles_to_ecfp(smiles: list[str], radius: int = 2, nbits: int = 1024, silent: bool = True, to_array: bool = True) \
        -> np.ndarray:
    """ Get a Numpy array of ECFPs from a list of SMILES strings """
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    from rdkit.Chem import MolFromSmiles
    from rdkit.DataStructs import ConvertToNumpyArray

    if type(smiles) is str:
        smiles = [smiles]

    fp = [GetMorganFingerprintAsBitVect(MolFromSmiles(s), radius, nBits=nbits) for s in tqdm(smiles, disable=silent)]

    if not to_array:
        return fp

    output = []
    for f in fp:
        arr = np.zeros((1,))
        ConvertToNumpyArray(f, arr)
        output.append(arr)

    return np.asarray(output)


class Evaluate:
    def __init__(self):
        self.binary_accuracy = [0]
        self.balanced_accuracy = [0]
        self.precision = [0]
        self.tpr = [0]
        self.roc_auc = [0]
        self.tn, self.fp, self.fn, self.tp = [0], [0], [0], [0]
        self.tnr = [0]
        self.prc_auc = [0]
        self.mcc = [0]
        self.ef5 = [0]

    def eval(self, logits_N_K_C: torch.Tensor, y: torch.Tensor):
        
        y = y.cpu() if type(y) is torch.Tensor else torch.tensor(y)
        y_hat = torch.mean(torch.exp(logits_N_K_C), dim=1)
        y_hat = y_hat.cpu() if type(y_hat) is torch.Tensor else torch.tensor(y_hat)
        has_positive = torch.any(y == 1).item()
        has_negative = torch.any(y == 0).item()

        y_hat_bin = torch.argmax(y_hat, dim=1)
        y_hat = y_hat[:, 1]

        # calc_binary_accuracy
        acc = torch.sum(y_hat_bin == y) / len(y)
        self.binary_accuracy.append(acc.item())

        # calc_balanced_accuracy
        balanced_acc = balanced_accuracy_score(y, y_hat_bin)
        self.balanced_accuracy.append(balanced_acc)

        # calc roc-auc
        try:
            if has_positive and has_negative:
                roc_auc = roc_auc_score(y, y_hat)
                self.roc_auc.append(roc_auc)
            else:
                self.roc_auc.append(0)
        except:
            self.roc_auc.append(0)

        # calc_precision
        try:
            self.precision.append(precision_score(y, y_hat_bin, zero_division=0))
        except:
            self.precision.append(0)

        # calc recall
        try:
            self.tpr.append(recall_score(y, y_hat_bin, zero_division=0))
        except:
            self.tpr.append(0)

        # calc confusion
        cm = confusion_matrix(y, y_hat_bin, labels=[0, 1]).ravel()
        tn, fp, fn, tp = cm
        self.tn.append(tn)
        self.fp.append(fp)
        self.fn.append(fn)
        self.tp.append(tp)

        # calc TNR (specificity)
        try:
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            self.tnr.append(tnr)
        except:
            self.tnr.append(0)

        # calc PRC AUC (average precision score)
        try:
            if has_positive:
                prc_auc = average_precision_score(y, y_hat)
                self.prc_auc.append(prc_auc)
            else:
                self.prc_auc.append(0)
        except:
            self.prc_auc.append(0)

        # calc MCC (Matthews Correlation Coefficient)
        try:
            mcc = matthews_corrcoef(y, y_hat_bin)
            self.mcc.append(mcc)
        except:
            self.mcc.append(0)

        # calc EF@5 (Enrichment Factor at 5%)
        try:
            # Sort predictions and get top 5%
            n_samples = len(y_hat)
            n_top = max(1, int(0.05 * n_samples))  # 5% of samples
            top_indices = torch.argsort(y_hat, descending=True)[:n_top]
            
            # Count hits in top 5%
            hits_in_top = torch.sum(y[top_indices] == 1).item()
            total_hits = torch.sum(y == 1).item()
            
            # Calculate EF@5
            if total_hits > 0:
                ef5 = (hits_in_top / n_top) / (total_hits / n_samples)
            else:
                ef5 = 0
            self.ef5.append(ef5)
        except:
            self.ef5.append(0)

    def __repr__(self):
        return f"Binary accuracy:    {self.binary_accuracy[-1]:.4f}\n" \
               f"Balanced accuracy:  {self.balanced_accuracy[-1]:.4f}\n" \
               f"ROC AUC:            {self.roc_auc[-1]:.4f}\n" \
               f"PRC AUC:            {self.prc_auc[-1]:.4f}\n" \
               f"Precision:          {self.precision[-1]:.4f}\n" \
               f"True positive rate: {self.tpr[-1]:.4f}\n" \
               f"True negative rate: {self.tnr[-1]:.4f}\n" \
               f"MCC:                {self.mcc[-1]:.4f}\n" \
               f"EF@5:               {self.ef5[-1]:.4f}\n" \
               f"Hits:               {self.tp[-1]}\n" \
               f"Misses:             {self.fn[-1]}\n" \
               f"False positives:    {self.fp[-1]}\n" \
               f"True negatives:     {self.tn[-1]}\n"

    def to_dataframe(self, colnames: str = ''):
        df = pd.DataFrame({
            'cycle': list(range(len(self.tp))),
            'binary_accuracy': self.binary_accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'auroc': self.roc_auc,
            'auprc': self.prc_auc,
            'precision': self.precision,
            'recall': self.tpr,
            'specificity': self.tnr,
            'mcc': self.mcc,
            'ef5': self.ef5,
            'tp': self.tp,
            'fn': self.fn,
            'fp': self.fp,
            'tn': self.tn,
        })
        df.columns = [f"{colnames}{i}" for i in df.columns]
        return df


def mol_descriptor(smiles: list, scale: bool = True):
    from rdkit.Chem.QED import qed
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from sklearn import preprocessing as pre
    # smiles = smiles[0]
    X = []
    for smi in tqdm(smiles):
        m = Chem.MolFromSmiles(smi)
        x = np.array([Descriptors.TPSA(m),
                      Descriptors.MolLogP(m),
                      Descriptors.MolWt(m),
                      Descriptors.FpDensityMorgan2(m),
                      Descriptors.HeavyAtomMolWt(m),
                      Descriptors.MaxPartialCharge(m),
                      Descriptors.MinPartialCharge(m),
                      Descriptors.NumRadicalElectrons(m),
                      Descriptors.NumValenceElectrons(m),
                      rdMolDescriptors.CalcFractionCSP3(m),
                      rdMolDescriptors.CalcNumRings(m),
                      rdMolDescriptors.CalcNumRotatableBonds(m),
                      rdMolDescriptors.CalcNumLipinskiHBD(m),
                      rdMolDescriptors.CalcNumLipinskiHBA(m),
                      rdMolDescriptors.CalcNumHeterocycles(m),
                      rdMolDescriptors.CalcNumHeavyAtoms(m),
                      rdMolDescriptors.CalcNumAromaticRings(m),
                      rdMolDescriptors.CalcNumAtoms(m),
                      qed(m)])
        X.append(x)

    if scale:
        return pre.MinMaxScaler().fit_transform(np.array(X))
    return np.array(X)


def tsne():

    from active_learning.data_prep import MasterDataset
    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.DataStructs import BulkTanimotoSimilarity
    from sklearn import preprocessing as pre

    ds_test = MasterDataset('test', representation='ecfp')
    ds_screen = MasterDataset('screen', representation='ecfp')

    x_test, y_test, smiles_test = ds_test.all()
    x_screen, y_screen, smiles_screen = ds_screen.all()

    x = np.concatenate((x_test, x_screen))
    y = np.concatenate((y_test, y_screen))
    smiles = np.concatenate((smiles_test, smiles_screen))
    split = ['test'] * len(x_test) + ['screen'] * len(x_screen)

    selected = [1 if i in smiles_screen else 0 for i in smiles]

    X_raw = mol_descriptor(smiles, scale=False)
    X_scaled = pre.MinMaxScaler().fit_transform(X_raw)

    x_df = pd.DataFrame(X_raw, columns=["TPSA", "MolLogP", "MolWt", "FpDensityMorgan2", "HeavyAtomMolWt",
                                    "MaxPartialCharge", "MinPartialCharge", "NumRadicalElectrons",
                                    "NumValenceElectrons", "CalcFractionCSP3", "CalcNumRings", "CalcNumRotatableBonds",
                                    "CalcNumLipinskiHBD", "CalcNumLipinskiHBA", "CalcNumHeterocycles",
                                    "CalcNumHeavyAtoms", "CalcNumAromaticRings", "CalcNumAtoms", "qed"])

    # fps = smiles_to_ecfp(smiles, to_array=False)
    # S = np.array([BulkTanimotoSimilarity(fps[i], fps) for i in tqdm(range(len(fps)))])
    # D = 1 - S

    perplexity = 50
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=500)
    results = tsne.fit_transform(X_scaled)
    df = pd.DataFrame({"x": results[:, 0], "y": results[:, 1], "label": y, "split": split, 'smiles': smiles, 'selected': selected})
    df = pd.concat([df, x_df], axis=1)
    df.to_csv(f'tsne_perp{perplexity}_500.csv', index=False)


def get_tanimoto_matrix(smiles: list[str], radius: int = 2, nBits: int = 1024, verbose: bool = True,
                        scaffolds: bool = False, zero_diag: bool = True, as_vector: bool = False):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""
    from active_learning.data_prep import smi_to_scaff

    # Make a fingerprint database
    db_fp = {}
    for smi in smiles:
        if scaffolds:
            m = Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False))
        else:
            m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len], dtype=np.float16)  # We use 16-bit floats to prevent giant matrices
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len), disable=not verbose):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]], db_fp[smiles[j]])
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    if zero_diag:
        np.fill_diagonal(m, 0)
    if as_vector:
        from scipy.spatial.distance import squareform
        m = squareform(m)

    return m


def to_torch_dataloader(x: Union[list, np.ndarray], y: Optional[np.ndarray] = None, **kwargs) -> \
        Union[DataLoader, pyg_DataLoader]:

    if type(x) is np.ndarray:
        assert y is not None, 'No y values provided'
        return DataLoader(TensorDataset(Tensor(x), Tensor(y).unsqueeze(1).type(torch.LongTensor)), **kwargs)
    else:
        return pyg_DataLoader(x, **kwargs)


def scramble_features(x: np.ndarray, seed: int = 1) -> np.ndarray:
    """ shuffle an array across the whole array (every number can end up anywhere)

    :param x: numpy array
    :param seed: random seed
    :return: shuffled array
    """

    rng = np.random.default_rng(seed=seed)

    # Flatten the array, shuffle, and reshape it back
    flat_x = x.flatten()
    rng.shuffle(flat_x)
    x = flat_x.reshape(x.shape)

    return x


def scramble_graphs(graphs, seed=1):

    def scramble_single_graph(g, seed):
        g.x = torch.tensor(scramble_features(np.array(g.x), seed=seed))
        g.edge_index = torch.tensor(scramble_features(np.array(g.edge_index), seed=seed))

        return g

    graphs = [scramble_single_graph(g, seed=seed) for g in graphs]

    return graphs


def visualize_ensemble_umap(features, hits=None, nonhits=None, save_path=None, cycle=None, global_xlim=None, global_ylim=None):
    """
    UMAP visualization that highlights hits vs nonhits and saves each cycle plot with consistent axes.

    :param features: np.ndarray, shape (n_samples, n_features)
    :param hits: list of int, indices of hit samples
    :param nonhits: list of int, indices of nonhit samples
    :param save_path: where to save the plot
    :param cycle: int, cycle number to include in title
    :param global_xlim: tuple, x-axis limits
    :param global_ylim: tuple, y-axis limits
    """
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)
    hits = np.array(hits, dtype=int) if hits is not None else np.array([], dtype=int)
    nonhits = np.array(nonhits, dtype=int) if nonhits is not None else np.array([], dtype=int)

    plt.figure(figsize=(10, 8))
    if hits is not None:
        plt.scatter(embedding[hits, 0], embedding[hits, 1],
                    c='red', s=10, label='Hits', alpha=0.7, marker='o')
    if nonhits is not None:
        plt.scatter(embedding[nonhits, 0], embedding[nonhits, 1],
                    c='blue', s=10, label='Non-hits', alpha=0.5, marker='x')

    if global_xlim:
        plt.xlim(global_xlim)
    if global_ylim:
        plt.ylim(global_ylim)

    plt.title(f"UMAP Feature Space - Cycle {cycle}" if cycle is not None else "UMAP Feature Space")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def compute_cosine_consistency(features_N_K_D: torch.Tensor) -> np.ndarray:
    """
    Compute per-sample feature consistency across ensemble members (mean pairwise cosine similarity).

    :param features_N_K_D: [N, K, D] feature tensor
    :return: numpy array of shape [N], mean pairwise cosine similarity per sample (excluding diagonal)
    """
    assert features_N_K_D.dim() == 3, "features should be [N, K, D]"
    N, K, D = features_N_K_D.shape
    if K == 1:
        return np.ones(N)

    # Normalize to unit vectors
    feats = features_N_K_D / (features_N_K_D.norm(dim=2, keepdim=True) + 1e-12)
    # Compute KxK cosine similarity matrix per sample, average off-diagonal
    sims = torch.einsum('nkd,njd->nkj', feats, feats)  # [N, K, K]
    diag_sum = torch.diagonal(sims, dim1=1, dim2=2).sum(dim=1)  # [N]
    total_sum = sims.sum(dim=(1, 2))  # [N]
    offdiag_sum = total_sum - diag_sum
    mean_sim = offdiag_sum / (K * (K - 1))
    return mean_sim.cpu().numpy()


def _center_gram_matrix(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    unit = torch.ones((n, n), device=K.device, dtype=K.dtype)
    identity = torch.eye(n, device=K.device, dtype=K.dtype)
    H = identity - unit / n
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between two representations X and Y.
    X: [N, D1], Y: [N, D2]
    """
    Kx = X @ X.T
    Ky = Y @ Y.T
    Kx_center = _center_gram_matrix(Kx)
    Ky_center = _center_gram_matrix(Ky)
    hsic = (Kx_center * Ky_center).sum()
    norm_x = torch.linalg.norm(Kx_center)
    norm_y = torch.linalg.norm(Ky_center)
    if norm_x.item() == 0 or norm_y.item() == 0:
        return 0.0
    return (hsic / (norm_x * norm_y)).item()


def compute_classwise_cka(features_N_K_D: torch.Tensor, labels_N: torch.Tensor) -> dict:
    """
    Compute class-wise (negative=0, positive=1) ensemble representation consistency (CKA), averaged over all ensemble member pairs.

    :param features_N_K_D: [N, K, D]
    :param labels_N: [N] binary labels (0/1)
    :return: { 'neg': float, 'pos': float }
    """
    assert features_N_K_D.dim() == 3, "features should be [N, K, D]"
    N, K, D = features_N_K_D.shape
    device = features_N_K_D.device
    y = labels_N.to(device)

    results = {}
    for cls_val, key in [(0, 'neg'), (1, 'pos')]:
        idx = (y == cls_val).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < 2 or K < 2:
            results[key] = 0.0
            continue
        feats_cls = features_N_K_D[idx]  # [Nc, K, D]
        # Compute linear CKA for all pairs of K members and average
        cka_vals = []
        for a in range(K):
            Xa = feats_cls[:, a, :]  # [Nc, D]
            for b in range(a + 1, K):
                Xb = feats_cls[:, b, :]  # [Nc, D]
                cka_vals.append(linear_cka(Xa, Xb))
        results[key] = float(np.mean(cka_vals)) if len(cka_vals) > 0 else 0.0

    return results

