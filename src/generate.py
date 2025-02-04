import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


import cleavenet
from cleavenet.data import custom_round
from cleavenet.utils import mmps

parser = argparse.ArgumentParser()
parser.add_argument("--num-seqs", default=100, type=int,
                    help="number of sequences to be generated")
parser.add_argument("--output-dir", default="generated", type=str,
                    help="Directory to store outputs ")
parser.add_argument("--repeat-penalty", default=1.0, type=float,
                    help="Repeat penalty factor, for no penalty use 1")
parser.add_argument("--temperature", default=1.0, type=float,
                    help="Sampling temperature, for standard sampling use 1. For less diverse use 0.7, for more diverse use >1")
parser.add_argument("--z-scores", default=None, type=str,
                    help="File containing z-scores for conditional generation")
args = parser.parse_args()

tf.config.list_physical_devices('GPU')

# Load in dataloader
data_dir = cleavenet.utils.get_data_dir()
data_path = os.path.join(data_dir, "kukreja.csv")
kukreja = cleavenet.data.DataLoader(data_path, seed=0, task='generator', model='autoreg', test_split=0.2,
                                            dataset='kukreja')

# From dataloader get necessary variables
start_id = kukreja.char2idx[kukreja.START]
vocab_size = len(kukreja.char2idx)

# Load model
model, checkpoint_path = cleavenet.models.load_generator_model(model_type='transformer',
                                                               training_scheme='rounded')
# Fake run to load data in model (have to do this for conditional models since run in eager mode)
conditioning_tag_fake = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
generated_seq = cleavenet.models.inference(model, kukreja, causal=True, seq_len=11,
                                           penalty=1, # no penalty
                                           verbose=True,
                                           conditioning_tag=conditioning_tag_fake,
                                           temperature=1 # no temp
                                           )
model.summary()

if args.z_scores is not None:
    cond_z_scores = pd.read_csv(args.z_scores)
    assert ([mmp in cond_z_scores.columns for mmp in mmps])
    cond_z_scores = cond_z_scores[mmps]
    for mmp in mmps:
        cond_z_scores[mmp] = cond_z_scores[mmp].apply(lambda x: custom_round(x, base=0.1))  # round to nearest 0.1
    conditioning_tag = cond_z_scores.values.tolist()
else:
    conditioning_tag = [[start_id]] # unconditional generation

tokenized_seqs = []
untokenized_seqs = []

for i in range(len(conditioning_tag)):
    for j in tqdm(range(args.num_seqs)):
        model.built=True
        model.load_weights(checkpoint_path)  # Load model weights
        # Generate using loaded weights
        generated_seq = cleavenet.models.inference(model, kukreja, causal=True, seq_len=11,
                                                   penalty=args.repeat_penalty,
                                                   verbose=False,
                                                   conditioning_tag=[conditioning_tag[i]], temperature=args.temperature)
        tokenized_seqs.append(generated_seq)
        untokenized_seqs.append(''.join(kukreja.idx2char[generated_seq]))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

save_file = os.path.join(args.output_dir, 'generated_samples_penalty_'+str(args.repeat_penalty)+'_temp_'+str(args.temperature)+'.csv')
with open(save_file, 'a') as f:
    for seq in untokenized_seqs:
        f.write(seq)
        f.write('\n')
