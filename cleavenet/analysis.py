import os

from collections import Counter
import jellyfish
import numpy as np
import pandas as pd
import scipy.stats as st

import cleavenet
from cleavenet.utils import mmps

def confidence_score(predictions, mmps):
    "calculate mean, standard error for each mmp"
    means = np.zeros((predictions.shape[1], len(mmps)))
    st_devs = np.zeros((predictions.shape[1], len(mmps)))

    for i, m in enumerate(mmps):
        means[:,i] = np.mean(predictions[:, :, i], axis=0)  # Take mean and std across batches
        st_devs[:,i] = np.std(predictions[:, :, i], axis=0)

    return means, st_devs


def confidence_interval_1_sided(std, num_samples, confidence_level=0.80):
    z_score = st.norm.ppf(confidence_level)
    ci = z_score * (std/np.sqrt(num_samples))
    return ci


def save_to_dataframe(x_all, y_all, mmp_idx, y_hat_all, uncertainty, z_cutoff=0, write_top_scores=False,
                      find_matches=False, dataloader=None, mmp=None, save_path=None, threshold=0, top=50):
    labels = pd.DataFrame(data=x_all, index=np.arange(len(x_all)))
    if y_all is not None:
        labels['True Z-scores'] = y_all
    labels['Pred Z-scores'] = y_hat_all[:,mmp_idx]
    labels['uncertainty'] = uncertainty[:,mmp_idx]
    # make df with true and predicted labels only
    if y_all is not None:
        scores = labels.iloc[:, -3:]
    else:
        scores = labels.iloc[:, -2:]
    if y_all is not None:
        scores['Cleaved true'] = scores['True Z-scores'].apply(lambda x: 1 if x > z_cutoff else z_cutoff)
        scores['Cleaved pred'] = scores['Pred Z-scores'].apply(lambda x: 1 if x > z_cutoff else z_cutoff)
    if write_top_scores:
        # Efficient (raw top 100 z-scores)
        sequences = ["".join(dataloader.idx2char[s]) for s in x_all]
        scores['sequences'] = sequences
        scores = scores.set_index('sequences')
        if threshold > 0:
            scores_thresh = scores[scores['uncertainty'] <= threshold]
            save_file1 = '_top'+str(top)+'_cleaved_threshold.csv'
            save_file2 = '_weighted_top'+str(top)+'_cleaved_threshold.csv'
            scores_sorted = scores_thresh.sort_values(by=['Pred Z-scores'], ascending=False)[:top]
        else:
            save_file1 = '_top'+str(top)+'_cleaved.csv'
            save_file2 = '_weighted_top'+str(top)+'_cleaved.csv'
            scores_sorted = scores.sort_values(by=['Pred Z-scores'], ascending=False)[:top]
        if find_matches:
            exact_match_freq, active_match_freq, s_lev = exact_match(list(scores_sorted.index), dataloader)
            # save all to dataframe
            scores_sorted['exact matches'] = exact_match_freq
            scores_sorted['exact active'] = active_match_freq
            scores_sorted['lev to z1'] = s_lev
        scores_sorted['mmp'] = [mmp] * len(scores_sorted)
        scores_sorted.to_csv(os.path.join(save_path, mmp+save_file1))

        # Selective (relative to mean top z-score)
        all_mmp_idx = list(range(y_hat_all.shape[1]))
        y_hat_weighted = np.zeros((y_hat_all.shape))
        # Weight scores based on mean of all other MMP scores (z-score the z-scores)
        for idx in all_mmp_idx:
            idx_to_mean = [i for i in all_mmp_idx if i is not idx]
            y_hat_weighted[:,idx] = y_hat_all[:,idx] - np.mean(y_hat_all[:, idx_to_mean], axis=1)
        scores_weighted = pd.DataFrame(y_hat_weighted[:,mmp_idx], columns=['Weighted Z-scores'])
        scores_weighted['uncertainty'] = uncertainty[:,mmp_idx]
        scores_weighted['sequences'] = sequences
        scores_weighted = scores_weighted.set_index('sequences')
        if threshold > 0:
            scores_weighted_thresh = scores_weighted[scores_weighted['uncertainty'] <= threshold]
            weighted_scores_sorted = scores_weighted_thresh.sort_values(by=['Weighted Z-scores'], ascending=False)[:top]
        else:
            weighted_scores_sorted = scores_weighted.sort_values(by=['Weighted Z-scores'], ascending=False)[:top]
        if find_matches:
            exact_match_freq, active_match_freq, s_lev = exact_match(list(weighted_scores_sorted.index), dataloader)
            # save all to dataframe
            weighted_scores_sorted['exact matches'] = exact_match_freq
            weighted_scores_sorted['exact active'] = active_match_freq
            weighted_scores_sorted['lev to z1'] = s_lev
        weighted_scores_sorted['mmp'] = [mmp] * len(weighted_scores_sorted)
        weighted_scores_sorted.to_csv(os.path.join(save_path, mmp + save_file2))

        # Write all scores once
        if not os.path.exists(os.path.join(save_path, 'weighted_all_scores.csv')):
            print("writing all scores")
            all_scores = pd.DataFrame(y_hat_weighted, columns=mmps)
            all_scores['sequences'] = sequences
            all_scores = all_scores.set_index('sequences')
            all_scores.to_csv(os.path.join(save_path, 'weighted_all_scores.csv'))

        if not os.path.exists(os.path.join(save_path, 'all_scores.csv')):
            print("writing all scores")
            all_scores = pd.DataFrame(y_hat_all, columns=mmps)
            all_scores['sequences'] = sequences
            all_scores = all_scores.set_index('sequences')
            all_uncertainty = pd.DataFrame(uncertainty, columns=mmps)
            all_uncertainty['sequences'] = sequences
            all_uncertainty = all_uncertainty.set_index('sequences')
            all_scores.to_csv(os.path.join(save_path, 'all_scores.csv'))
            all_uncertainty.to_csv(os.path.join(save_path, 'all_uncertainty.csv'))
    return scores


def exact_match(sequences, dataloader):
    """
    sequences (str) : list of sequences to compare
    train (str) : list of train data sequences
    ensemble: pick 1 run
    """
    x_train = dataloader.X # use all not just train here

    dist_array = np.zeros((len(sequences), len(x_train)))
    active_array = np.zeros((len(sequences), len(x_train)))
    #s_array = np.zeros((len(sequences)))

    exact_match_freq = []
    active_match_freq = []

    for i in range(len(sequences)):
        full_matches = 0
        active_matches = 0
        #s_array[i] = jellyfish.levenshtein_distance(sequences[i], s1)
        for j in range(len(x_train)):
            active_array[i, j] = jellyfish.levenshtein_distance(sequences[i][2:7], x_train[j][2:7])
            dist_array[i, j] = jellyfish.levenshtein_distance(sequences[i], x_train[j])
        if 0 in dist_array[i, :]:
            index = np.where(dist_array[i] == 0)
            full_matches+=len(index)
        if 0 in active_array[i, :]:
            index_a = np.where(active_array[i] == 0)
            active_matches+=len(index_a)
        exact_match_freq.append(full_matches)
        active_match_freq.append(active_matches)
    return exact_match_freq, active_match_freq #, s_array


def eval_all_mmp(means, x_all, dataloader, save_path, z_score_cutoff=0):
    sequences = ["".join(dataloader.idx2char[s]) for s in x_all]
    df = pd.DataFrame(means, columns=mmps)
    df['freq'] = [0] * len(df)
    for mmp in mmps:
        freq_index = df[df[mmp] >= z_score_cutoff].index
        df['freq'].iloc[freq_index] += 1
    df['sequences'] = sequences
    df = df.set_index('sequences')
    df = df.sort_values('freq', ascending=False)[:20]
    df.to_csv(os.path.join(save_path, 'allmmp_top20_cleaved.csv'))


def extract_sequences(x_train, dataloader):
    seqs_arr = []
    x_train = cleavenet.data.tokenize_sequences(x_train, dataloader)
    for k in range(x_train.shape[1]):
        seqs = ''
        for s in x_train[:, k]:
            tokenized = dataloader.idx2char[s]
            for t in tokenized:
                if t != '-':
                    seqs += t
        seqs_arr.append(seqs)
    return seqs_arr


def count_by_res(seqs_arr):
    dict_arr = []
    for i in seqs_arr:
        aminos = Counter(
             {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
              'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0})
        aminos.update(i)
        dict_arr.append(aminos)
    return dict_arr


def normalize(list):
    norm = sum(list)
    new_list = [item / norm for item in list]
    return new_list


def calc_selectivity_score(mmps, df, penalize=[]):
    df_new = pd.DataFrame()
    for mmp in mmps:
        current_mmp = [mmp]
        other_mmps = [m for m in mmps if m not in current_mmp and m not in penalize]
        ss = np.array(df[current_mmp]).squeeze()-(df[other_mmps].mean(axis=1))
        if len(penalize) > 0:
            ss = ss - np.array(df[penalize].mean(axis=1).squeeze())
        df_new[str(mmp)] = ss
    return df_new


def get_roc_data(y, y_hat, true_z_cutoff=2, width=0.1):
    x_values = []
    y_values = []
    df = pd.DataFrame()
    df['True'] = y #[:, mmp_index]
    df['Pred'] = y_hat #[:, mmp_index]
    thresh_range = np.arange(y.min(), y.max(), width)
    for z_cutoff in thresh_range:
        # Make scores binary according to 2 z-thresh
        df['Cleaved True'] = df['True'].apply(lambda x: 1 if x > true_z_cutoff else 0)
        df['Cleaved Pred'] = df['Pred'].apply(lambda x: 1 if x > z_cutoff else 0)
        # Conditions for TP, FP, FN, TN
        def conditions(s):
            if (s['Cleaved True'] == 1) & (s['Cleaved Pred'] == 1):
                return 'TP'
            elif ((s['Cleaved True'] == 0) & (s['Cleaved Pred'] == 1)):
                return 'FP'
            elif (s['Cleaved True'] == 1) & (s['Cleaved Pred'] == 0):
                return 'FN'
            else:
                return 'TN'
        df['class'] = df.apply(conditions, axis=1)
        #(df.head())
        tp = (df['class'] == 'TP').sum()
        fn = (df['class'] == 'FN').sum()
        tn = (df['class'] == 'TN').sum()
        fp = (df['class'] == 'FP').sum()
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        y_values.append(tpr)
        x_values.append(fpr)
    return x_values, y_values


def calc_auc(x_values, y_values):
    inds = [x for i, x in enumerate(zip(x_values, y_values)) if x[0] != x[1]]
    area = 0
    for i in range(len(inds)-1):
        area += (inds[i][0] - inds[i+1][0]) * (inds[i][1] + inds[i+1][1]) / 2 # calc using trapezoid
    return area