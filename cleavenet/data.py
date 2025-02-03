import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cleavenet.utils import mmps

def pad(sequence, max_seq_len=10, pad_token='-'):
    seq_len = len(sequence)
    if seq_len <= max_seq_len:
        sequence += pad_token * (max_seq_len-seq_len)
    elif seq_len > max_seq_len:
        print("Increase max seq len for padding")
        import pdb; pdb.set_trace()
    return sequence


def custom_round(x, base=5):
    return base * round(float(x)/base)


class DataLoader(object):
    """
    DataLoader class for loading peptide sequence data, given a CSV file (data_path)
    Creates 1 test/train split for Dataset, or loads saved splits
    Saves a char2idx dict mapping for each model
    """
    def __init__(self, data_path, seed=0, task='predictor', model='bert', test_split=0.2, dataset='kukreja',
                 use_dataloader=None, rounded=False):
        self.seed = seed
        self.model = model  # bert, autoregressive, regression
        self.dataset = dataset
        if rounded:
            self.dataset += '_rounded'
        self.data_path = data_path
        self.test_split = test_split
        self.task = task
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Set save paths
        self.out_path = os.path.join('splits/', self.dataset+'/') # One split per dataset
        if os.path.exists(self.out_path):
            print("Splits previously written to file")
            self.X = list(get_data(self.out_path + 'X_all.csv', names=['sequence']).index)
            self.y = get_data(self.out_path + 'y_all.csv', index_col=None, names=mmps).values
            self.sequences = self.X
            if test_split > 0:
                self.X_train = list(get_data(self.out_path + 'X_train.csv', names=['sequence']).index)
                self.y_train = get_data(self.out_path + 'y_train.csv', index_col=None, names=mmps).values
                self.X_test = list(get_data(self.out_path + 'X_test.csv', names=['sequence']).index)
                self.y_test = get_data(self.out_path + 'y_test.csv', index_col=None, names=mmps).values
        else:
            os.makedirs(self.out_path)
            print('Splits directory created', self.out_path)
            #Load data
            data = self.load_zscore_data()
            self.sequences = data.index.to_list()
            if dataset == 'kukreja':
                # replace gaps in data in kukreja
                # these are artificially created in csv processing
                self.sequences = [seq.replace(' ', '') for seq in self.sequences]
            if rounded:
                data = data.apply(lambda x: custom_round(x, base=0.5)) # round to nearest 0.5
            # Assign data
            self.X = self.sequences  # sequences
            self.y = data.values
            np.savetxt(self.out_path + 'X_all.csv', self.X, delimiter=",", fmt='%s')
            np.savetxt(self.out_path + 'y_all.csv', self.y, delimiter=",")
            if test_split > 0:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                        test_size=self.test_split,
                                                                                        random_state=self.seed)
                np.savetxt(self.out_path + 'X_train.csv', self.X_train, delimiter=",", fmt='%s')
                np.savetxt(self.out_path + 'y_train.csv', self.y_train, delimiter=",")
                np.savetxt(self.out_path + 'X_test.csv', self.X_test, delimiter=",", fmt='%s')
                np.savetxt(self.out_path + 'y_test.csv', self.y_test, delimiter=",")

        # Create vocab
        if not use_dataloader:
            self.char2idx, self.idx2char = self.create_vocab()
            print("Vocab: \n", self.char2idx)
        else:
            self.char2idx = use_dataloader.char2idx
            self.idx2char = use_dataloader.idx2char
            print("Vocab: \n", self.char2idx)

        # If using generator
        if task == 'generator':
            train_data = pd.DataFrame(self.y_train, columns=mmps)
            train_data['sequence'] = self.X_train
            train_data = train_data.set_index('sequence')
            self.X_train = train_data.index.to_list()
            self.y_train = train_data.values

            test_data = pd.DataFrame(self.y_test, columns=mmps)
            test_data['sequence'] = self.X_test
            test_data = test_data.set_index('sequence')
            self.X_test = test_data.index.to_list()
            self.y_test = test_data.values


    def load_zscore_data(self):
        """
        Load z-scored protease-substrate data from path
        """
        df_z = pd.read_csv(self.data_path, header=0)
        df_z = df_z.dropna(how='all', axis=1)
        if self.dataset == 'kukreja' or self.dataset == 'kukreja_rounded':
            df_z = df_z.drop('construct', axis=1)
        elif self.dataset == 'bhatia': # Remove all seqs not == 10 residues in length (this is outside training task)
            new_sequences = df_z['sequence'].to_list()
            new_sequences = [seq[2:-2] for seq in new_sequences] # crop in middle
            seq_lens = [len(seq) for seq in new_sequences]
            df_z['sequence'] = new_sequences
            df_z['len'] = seq_lens
            df_z = df_z.drop('len', axis=1)
            print(df_z)
        df_z = pd.pivot_table(df_z, index=["sequence"])  # average z-scores for any duplicated sequences
        print("WARNING: PIVOT TABLE WILL RE-ORDER COLUMNS IN DATAFRAME- ENSURE LABELING IS MAPPED CORRECTLY")
        return df_z


    def create_vocab(self):
        # Find unique characters
        seqs_joined = "".join(self.sequences)

        if self.task == 'regression':
            self.PAD = '-'
            vocab = [self.PAD] # set 0 to pad token
            vocab += sorted(set(seqs_joined))
            # Add CLS for transformer
            if self.model == 'transformer':
                self.CLS = '!' # classifier token
                vocab += self.CLS
        else:
            vocab = sorted(set(seqs_joined))
        # For generator task add special tokens
        if self.task == 'generator':
            if self.model == 'bert':
                self.MASK = '#'
                vocab += self.MASK
            elif self.model == 'autoreg':
                self.START = '$'
                self.STOP = '*'
                vocab += self.START
                vocab += self.STOP

        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        return char2idx, idx2char


def tokenize_sequences(sequences, dataloader):
    """
    Tokenize sequences
    """
    char2idx, idx2char = dataloader.char2idx, dataloader.idx2char
    tokenized = [np.array([char2idx[aa] for aa in sequence]) for sequence in sequences]
    return np.stack(tokenized) # This will break if sequences are different lengths


def untokenize_sequences(sequences, dataloader):
    """
    Untokenize sequences
    """
    char2idx, idx2char = dataloader.char2idx, dataloader.idx2char
    untokenized = [np.array(idx2char[sequence]) for sequence in sequences]
    return np.stack(untokenized) # This will break if sequences are different lengths


def get_batch(x, y, batch_size, dataloader, test=False, transformer=False):
    """
    Randomly sample batch of training data (x,y), and tokenize inputs (x) for training
    """
    #x = np.array(x)
    if transformer:
        cls_idx = dataloader.char2idx[dataloader.CLS]

    if test==True:
        batch_X = x
        batch_Y = y
    else:
        # Randomly choose the indices for the examples in the training batch
        selected_inds = np.random.choice(len(x), size=batch_size)
        batch_X = [x[s] for s in selected_inds]
        batch_Y = [y[s] for s in selected_inds]

    # Tokenize inputs
    tokenized_batch_X = tokenize_sequences(batch_X, dataloader)
    if transformer:
        tokenized_batch_X = np.stack([np.append(np.array(cls_idx), s) for s in tokenized_batch_X])
    return tokenized_batch_X, batch_Y


def get_data(path, index_col=0, names=None):
    """Transform data from .csv file to np.array
    Input:
        path (str): path to .csv
    """
    if names is not None:
        data = pd.read_csv(path, index_col=index_col, names=names)
    else:
        data = pd.read_csv(path, index_col=index_col)
    return data


def read_csv(path):
    data = pd.read_csv(path, index_col=0, names=['sequences'])
    return list(data.index)


def get_masked_batch(seqs, batch_size, rng, dataloader):
    """ Corrupt inputs with a BERT-like training scheme

    Args:
        seqs (np.array): input sequences to be divided into kmers
        batch_size (int)
        rng: ranodom seed

        Returns:
        masked_seq (array): corrupted sequences
        target_seq (array): original sequences
        mask (array): bool array of corrupted locations
    """
    # Initiate arrays to store input and output sequences for all training sequences
    target_seq = []
    masked_seq = []
    mask_output = []

    vocab_words = list(dataloader.char2idx.keys())
    mask_id = dataloader.char2idx[dataloader.MASK]

    batch_inds = np.random.choice(len(seqs), size=batch_size)
    batch_seqs = [seqs[i] for i in batch_inds]
    batch_seqs = tokenize_sequences(batch_seqs, dataloader)

    for seq in batch_seqs:
        # calculate seq length
        n = len(seq)
        # For now randomly mask 15% of sequence with mask token only
        num_mask = 2 #int(0.15 * n) # ~ this is approx 2 for a 10mer
        # generate mask
        mask_arr = np.random.choice(n, num_mask, replace=False)  # Generates array of len num_mask
        index_arr = np.arange(0, n)  # index array [1...seq_len]
        mask = np.isin(index_arr, mask_arr, invert=False).reshape(index_arr.shape)  # mask bools indices specified by mask_arr
        # Mask inputs
        # 80% of time w/ mask token
        # 10% random token
        # 10% with same token
        masked_seq_temp = seq.copy()
        for masked_index in mask_arr:
            #print(masked_index)
            if rng.random() < 0.8:
                masked_token = mask_id
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = seq[masked_index]
                # 10% of the time, replace with random token
                else:
                    masked_token = rng.randint(0, len(vocab_words) - 1)
            masked_seq_temp[masked_index] = masked_token
        masked_seq.append(masked_seq_temp)
        mask_output.append(mask)
        target_seq.append(seq)
    return np.array(masked_seq), np.array(target_seq), np.array(mask_output) # input (x), target(y), mask


def get_autoreg_batch(seqs, batch_size, dataloader, width=11, conditioning_tag=None, rng=0, randomize_tag=False):
    """
    Autoregressive inputs/target
    seqs: input sequences (list of strings)
    batch_size: batch size (int)
    dataloader: saved dataloader with dictionary (Dataloader)
    width: context window (for short seqs, this is the entire length (10) + start or stop token (1)
    conditioning_tag: array of z scores in same order as sequences (y_train), or None
    rng: random seed 

    Returns:
        src_seqs (array): [START] + seqs[0:10]
        target_seq (array): seqs shifted by 1; seqs[1:10] + [END]
    """
    start_idx = dataloader.char2idx[dataloader.START]
    stop_idx = dataloader.char2idx[dataloader.STOP]

    batch_inds = np.random.choice(len(seqs), size=batch_size)
    batch_seqs = [seqs[i] for i in batch_inds]
    batch_seqs = tokenize_sequences(batch_seqs, dataloader)

    # Add special chars
    batch_seqs = [np.append(np.append(np.array(start_idx), s), np.array(stop_idx)) for s in batch_seqs]
    slice_index = np.random.randint(0, len(batch_seqs[0])-width, size=len(batch_seqs)) # if context window less than length (not needed here)
    
    target_seq = [s[slice_index[i]+1:slice_index[i]+width+1] for i, s in enumerate(batch_seqs)]
    if conditioning_tag is not None:
        batch_tags = [np.array(conditioning_tag[i]) for i in batch_inds]
        if randomize_tag:  
            batch_tags = [np.array(b) if rng.random() < 0.5 else np.array([start_idx]) for b in batch_tags]
        batch_seqs = [s[1:-1] for s in batch_seqs]
        return (np.array(batch_seqs), batch_tags), np.array(target_seq)
    else:
        # sample around window width
        src_seqs = [s[slice_index[i]:slice_index[i]+width] for i, s in enumerate(batch_seqs)]
        return np.array(src_seqs), np.array(target_seq)