
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm.auto import tqdm
from active_learning.screening import active_learning
from config import ROOT_DIR
import itertools
import argparse
import warnings
from datetime import datetime


warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")
warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.*")

PARAMETERS = {'max_screen_size': [1088],
              'n_start': [64],
              'init_pos_count': [10],
              'batch_size': [64],
              'architecture': ['gcn', 'mlp', 'gat', 'gin', 'rf'],
              'dataset': ['PKM2', 'TP53', 'CYP3A4'],
              'seed': list(range(5)),
              'bias': ['random', 'real', 'small', 'large'],
              'acquisition': ['random', 'exploitation', 'uncertainty', 'similarity', 'bald', 'bala'],
              }


def build_ratio_token(init_pos_count: int, n_start: int) -> str:
    if init_pos_count is None:
        return 'pdefault'
    return f'p{init_pos_count}of{n_start}'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='Output directory', default=os.path.join(ROOT_DIR, 'results'))
    parser.add_argument('-acq', help='Acquisition strategy key', default='exploitation')
    parser.add_argument('-bias', help='Initial bias ("random", "real", "small", "large")', default='real')
    parser.add_argument('-arch', help='Model architecture ("mlp", "gcn", "gat", "gin", "rf")', default='mlp')
    parser.add_argument('-dataset', help='Dataset name', default='CYP3A4')
    parser.add_argument('-retrain', help='Retrain model every cycle', default='True')
    parser.add_argument('-batch_size', help='Molecules selected per cycle', type=int, default=64)
    parser.add_argument('-n_start', help='Initial training set size', type=int, default=64)
    parser.add_argument('-max_screen_size', help='Total molecules to screen (default: whole library)', type=int, default=None)
    parser.add_argument('-init_pos_count', help='Positive samples in starting set when bias=real', type=int, default=10)
    parser.add_argument('-seed', help='Number of random seeds, generates 0..N-1', type=int, default=5)
    parser.add_argument('-anchored', help='Anchor the weights', default='True')
    parser.add_argument('-scrambledx', help='Scramble the features', default='False')
    parser.add_argument('-scrambledx_seed', help='Seed for scrambling the features', type=int, default=1)
    args = parser.parse_args()

    PARAMETERS['acquisition'] = [args.acq]
    PARAMETERS['bias'] = [args.bias]
    PARAMETERS['dataset'] = [args.dataset]
    PARAMETERS['retrain'] = [eval(args.retrain)]
    PARAMETERS['architecture'] = [args.arch]
    PARAMETERS['batch_size'] = [args.batch_size]
    PARAMETERS['n_start'] = [args.n_start]
    PARAMETERS['init_pos_count'] = [args.init_pos_count]
    if args.max_screen_size is not None:
        PARAMETERS['max_screen_size'] = [args.max_screen_size]
    PARAMETERS['anchored'] = [eval(args.anchored)]
    PARAMETERS['scrambledx'] = [eval(args.scrambledx)]
    PARAMETERS['scrambledx_seed'] = [args.scrambledx_seed]

    if args.seed is not None:
        PARAMETERS['seed'] = list(range(args.seed))

    os.makedirs(args.o, exist_ok=True)
    ratio_token = build_ratio_token(PARAMETERS['init_pos_count'][0], PARAMETERS['n_start'][0])
    date_prefix = datetime.now().strftime('%y%m%d')
    run_dir = f'{args.o}/{date_prefix}_{args.dataset}_{args.acq}_{args.batch_size}_{ratio_token}'
    os.makedirs(run_dir, exist_ok=True)
    LOG_FILE = f'{run_dir}/{args.dataset}_{args.acq}_{args.batch_size}_{ratio_token}_results.csv'

    experiments = [dict(zip(PARAMETERS.keys(), v)) for v in itertools.product(*PARAMETERS.values())]

    for experiment in tqdm(experiments):

        results = active_learning(n_start=experiment['n_start'],
                                  bias=experiment['bias'],
                                  acquisition_method=experiment['acquisition'],
                                  max_screen_size=experiment['max_screen_size'],
                                  batch_size=experiment['batch_size'],
                                  architecture=experiment['architecture'],
                                  seed=experiment['seed'],
                                  retrain=experiment['retrain'],
                                  anchored=experiment['anchored'],
                                  dataset=experiment['dataset'],
                                  scrambledx=experiment['scrambledx'],
                                  scrambledx_seed=experiment['scrambledx_seed'],
                                  init_pos_count=experiment['init_pos_count'],
                                  optimize_hyperparameters=False,
                                  output_dir=run_dir)

        results['acquisition_method'] = experiment['acquisition']
        results['architecture'] = experiment['architecture']
        results['n_start'] = experiment['n_start']
        results['batch_size'] = experiment['batch_size']
        results['init_pos_count'] = experiment['init_pos_count']
        results['seed'] = experiment['seed']
        results['bias'] = experiment['bias']
        results['retrain'] = experiment['retrain']
        results['scrambledx'] = experiment['scrambledx']
        results['scrambledx_seed'] = experiment['scrambledx_seed']
        results['dataset'] = experiment['dataset']

        results.to_csv(LOG_FILE, mode='a', index=False, header=False if os.path.isfile(LOG_FILE) else True)
