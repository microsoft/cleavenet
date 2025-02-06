import argparse
import os

import pandas as pd
import cleavenet

from cleavenet.utils import mmps

parser = argparse.ArgumentParser()
parser.add_argument("--path-to-sequence-csv", type=str, default='/data/',
                    help="path to csv file where each line should be a new peptide sequence")
parser.add_argument("--path-to-zscores", type=str, default=None,
                    help="if you want to measure our model predictions against your data, provide a path to csv file \
                          where each line should be a corresponding z-score, see 'splits/y_all.csv' for an example. If \
                          using different MMPS, the first row of this file should correspond to the MMPs in each row. "
                         "The default is to assign each column to MMPs in the order we analyzed the data")
parser.add_argument("--no-csv-header", action='store_true',
                    help="If using data splits from kukreja as described in the README, we store the MMP headers separately. Use this flag to indicate that")
parser.add_argument("--save-dir", type=str, default='outputs/',
                    help="directory to save model outputs too")
parser.add_argument("--model-architecture", type=str, default='transformer',
                    help="'transformer' or 'lstm, for most use cases the default should be used'")
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

input_df = pd.read_csv(args.path_to_sequence_csv, names=['sequence']).set_index('sequence')
eval_sequences = input_df.index.to_list()

true_scores=None
true_mmps=mmps
if args.path_to_zscores is not None:
    if args.no_csv_header:
        true_scores = pd.read_csv(args.path_to_zscores, names=mmps).to_numpy()
    else:
        true_scores = pd.read_csv(args.path_to_zscores)
        true_mmps = true_scores.columns.to_list()
        true_scores = true_scores.to_numpy()
data_dir = cleavenet.utils.get_data_dir()
data_path = os.path.join(data_dir, "kukreja.csv")

# Load in dataloader
kukreja = cleavenet.data.DataLoader(data_path, seed=0, task='generator', model='autoreg', test_split=0.2,
                                            dataset='kukreja')

k_pred_zscores, k_std_zscores = cleavenet.models.prediction(data_path,
                                                            eval_sequences,
                                                            args.save_dir,
                                                            checkpoint_dir='weights/',
                                                            predictor_model_type=args.model_architecture,
                                                            true_zscores=true_scores,
                                                            true_mmps=true_mmps)
