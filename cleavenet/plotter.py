import os

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix

from cleavenet import analysis


def plot_parity(labels, predicted, mmps, save_path, nrows=6, ncols=3):
    """ Plot scatterplot of true vs. predicted z-scores and save .png
        Args:
        labels (np.array): true z-scores
        predicted (np.array): predicted z-scores
        mmp (list) : list of all mmps
        save_path (str): saving path
    """
    for i in range(len(mmps)):
        data = pd.DataFrame({'True': labels[:,i], 'Predicted': predicted[:,i]})
        p = sns.jointplot(data=data, x='True', y='Predicted', kind='reg') #, xlim=(-2, 5), ylim=(-2, 5))
        #corr_pear = data.corr(method='pearson')
        r_squared = stats.pearsonr(labels[:,i], predicted[:,i]).statistic
        label = " $R$=%.2f" % (r_squared)
        #ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14,
        #        verticalalignment='top')
        p.fig.suptitle(mmps[i]+label, weight='bold') #+' R\N{SUPERSCRIPT TWO}=%.2f' % corr_pear.iloc[0, 1])
        p.fig.tight_layout()
        save_dir_test = os.path.join(save_path, mmps[i]+'_true_pred_scatter.svg')
        p.savefig(save_dir_test)


def plot_rmse(rmse, mmps, save_path):
    "Plot average RMSE score for each MMP family after training"
    rmse = np.array(rmse)
    fig = plt.figure(figsize=(12,3))
    # creating the bar plot
    plt.bar(mmps, rmse, color='k', alpha=0.75, width=0.75)
    #plt.xlabel("MMP Family")
    plt.ylabel("RMSE")
    plt.ylim(0,1)
    plt.tight_layout()
    np.savetxt(save_path + 'rmse.csv', rmse, delimiter=", ", fmt='% s')
    save_file = save_path + 'rmse.svg'
    fig.savefig(save_file)


def plot_mae(mae, mmps, save_path):
    "Plot average MAE score for each MMP family after training"
    mae = np.array(mae)
    fig = plt.figure(figsize=(12,3))
    plt.bar(mmps, mae, color='k', alpha=0.75, width=0.75)
    #plt.xlabel("MMP Family")
    plt.ylabel('MAE')
    plt.ylim(0,1)
    plt.tight_layout()
    np.savetxt(save_path+'mae.csv', mae, delimiter=", ", fmt='% s')
    save_file = save_path+'mae.svg'
    fig.savefig(save_file)


def true_pred_ranked_scatter_z(scores, save_path, mmp):
    "Plot ranked predicted z-scores colored by true z-score"
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = {0: 'blue', 1: 'red'}
    scores_sorted = scores.sort_values(by=['Pred Z-scores'], ascending=True)
    ax.scatter(np.arange(len(scores_sorted['Pred Z-scores'].values)),
               scores_sorted['Pred Z-scores'].values,
               c=scores_sorted['Cleaved true'].map(colors))
    plt.axhline(y=0, c='k', alpha=0.50, ls='dashed')
    plt.xlabel('Ranked substrates by predicted z-score')
    plt.ylabel('Pred z-score')
    plt.title(mmp, weight='bold')
    save_file='true_pred_ranked_z.svg'
    save_dir_test = os.path.join(save_path, mmp+save_file)
    fig.savefig(save_dir_test)
    plt.close()


def confidence_ranked_scatter_z(scores, save_path, mmp, threshold=0):
    "Plot ranked z-scores colored by confidence score"
    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = 0
    vmax = 0.5 # scores['uncertainty'].max()
    ymin = scores['Pred Z-scores'].min()-0.25
    ymax = scores['Pred Z-scores'].max()+0.25
    if threshold > 0:
        scores = scores[scores['uncertainty']<=threshold]
        save_file = 'confidence_ranked_z_threshold.svg'
    else:
        save_file = 'confidence_ranked_z.svg'
    scores_sorted = scores.sort_values(by=['Pred Z-scores'], ascending=True)
    p = ax.scatter(np.arange(len(scores_sorted['Pred Z-scores'].values)),
               scores_sorted['Pred Z-scores'].values,
               c=scores_sorted['uncertainty'], vmin=vmin, vmax=vmax)
    plt.axhline(y=0, c='k', alpha=0.50, ls='dashed')
    plt.ylim(ymin, ymax)
    fig.colorbar(p, ax=ax)
    plt.xlabel('Ranked substrates by predicted z-score')
    plt.ylabel('Pred z-score')
    plt.title(mmp, weight='bold')
    save_dir_test = os.path.join(save_path, mmp+save_file)
    fig.savefig(save_dir_test)
    plt.close()


def confusion(y_true, y_pred, save_path, mmp):
    """ Calculate, plot and save confusion matrix

        Args:
        y_true (np.array): true labels
        y_pred (float): predicted labels

        Saves:
        confusion_matrix.png
        confusion_matrix.csv
    """
    # Calculate confusion matrix
    labels = ['Non-cleaved', 'Cleaved']
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Save heatmap plot of confusion matrix
    fig = plt.figure()
    ax = plt.subplot()
    sns.heatmap(cm, cmap=sns.color_palette("Blues", as_cmap=True),annot=True, ax=ax)  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(mmp, weight='bold')
    ax.xaxis.set_ticklabels(labels, size=13)
    ax.yaxis.set_ticklabels(labels, size=13)
    fig.savefig(os.path.join(save_path, mmp+'_confusion_matrix.svg'))
    plt.close()


def confidence_histogram(scores, save_path, mmp):
    ci = analysis.confidence_interval_1_sided(np.std(scores['uncertainty']), len(scores), confidence_level=0.80)
    #print(scores['uncertainty'].mean()+ci)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(data=scores, x='uncertainty', ax=ax, kde=True)
    plt.axvline(x=scores['uncertainty'].mean()+ci, c='k', alpha=0.50, ls='dashed')
    plt.xlabel('Uncertainty')
    plt.ylabel('Count')
    plt.title(mmp)
    save_dir_test = os.path.join(save_path, mmp + '_confidence_hist.svg')
    fig.savefig(save_dir_test)
    plt.close()
    return(scores['uncertainty'].mean()+ci)


def confidence_test_histogram(scores, save_path, mmp):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(data=scores, x='True Z-scores', y='Pred Z-scores', ax=ax, ci=0.8)
    plt.xlabel('Uncertainty')
    plt.ylabel('Count')
    plt.title(mmp)
    save_dir_test = os.path.join(save_path, mmp + '_confidence_hist.svg')
    fig.savefig(save_dir_test)
    plt.close()


def heatmaps(save_path, data_dir = '/generated/', cbar_center=0, weighted=False, cluster=True, top=50):
    plot_mmps = ['MMP1', 'MMP2', 'MMP3', 'MMP8', 'MMP9', 'MMP10', 'MMP11',
            'MMP12', 'MMP13', 'MMP14', 'MMP15', 'MMP16', 'MMP17', 'MMP20',
            'MMP24', 'MMP25', 'MMP7'] #, 'MMP19'] # Exclude MMP19
    if weighted:
        all_scores = pd.read_csv(data_dir+'weighted_all_scores.csv')
    else:
        all_scores = pd.read_csv(data_dir+'all_scores.csv')
    dfs = []
    if weighted:
        file_name = '_weighted_top' + str(top) + '_cleaved.csv'
        if cluster:
            save_name = 'weighted_cluster_heatmap.svg'
            save_df = 'weighted_cluster_df.csv'
            save_col = 'weighted_cluster_col.csv'
            save_row = 'weighted_cluster_row.csv'
        else:
            save_df = 'weighted_df.csv'
            save_name = 'weighted_heatmap.svg'
            save_col = 'weighted_col.csv'
            save_row = 'weighted_row.csv'
    else:
        file_name = '_top' + str(top) + '_cleaved.csv'
        if cluster:
            save_name = 'top_cluster_heatmap.svg'
            save_df = 'top_cluster_df.csv'
            save_col = 'top_cluster_col.csv'
            save_row = 'top_cluster_row.csv'
        else:
            save_df = 'top_df.csv'
            save_name = 'top_heatmap.svg'
            save_col = 'top_col.csv'
            save_row = 'top_row.csv'
    for mmp in plot_mmps:
        file_path = data_dir + mmp + file_name
        df_temp = pd.read_csv(file_path)[:top] # top 20 for each MMP
        dfs.append(df_temp)
    df = pd.concat(dfs)
    df_merge = df.merge(all_scores, how='left')
    df_merge = df_merge.drop_duplicates(subset=['sequences']) # drop duplicate sequences, add additional scores
    fixed_seq_list = [seq[1:] for seq in list(df_merge['sequences'])]
    df_merge['sequences'] = fixed_seq_list
    df_merge = df_merge.set_index('sequences')
    print(df_merge)
    # Default plot
    if cluster:
        ax = sns.clustermap(df_merge[plot_mmps], cmap='RdBu_r', center=cbar_center, figsize=(12,10)) #, vmin=-2, vmax=5)
        cluster_row = ax.dendrogram_row.reordered_ind
        cluster_col = ax.dendrogram_col.reordered_ind
        df_merge.to_csv(save_path+save_df)
        np.savetxt(save_path+save_row, cluster_row, delimiter=", ", fmt='% s')
        np.savetxt(save_path+save_col, cluster_col, delimiter=", ", fmt='% s')
    else:
        fig = plt.figure()
        ax = plt.subplot()
        sns.heatmap(df_merge[plot_mmps], cmap='RdBu_r', center=cbar_center) #, vmin=-2, vmax=5)
        plt.tight_layout()
    save_plot = os.path.join(save_path, save_name)
    plt.savefig(save_plot)


def aa_freq(x_train, x_gen, dataloader, save_path, c='xkcd:pea', save_file='aa_frequency.svg'):
    train_seqs = analysis.extract_sequences(x_train, dataloader)
    if x_gen is not None:
        gen_seqs = analysis.extract_sequences(x_gen, dataloader)
        gen_dict_arr = analysis.count_by_res(gen_seqs)
    dict_arr = analysis.count_by_res(train_seqs)
    # Plot
    nrow = 2
    ncol = 5
    title = ['P5', 'P4', 'P3', 'P2', 'P1', "P1'", "P2'", "P3'", "P4'", "P5'"]
    fig, axs = plt.subplots(nrow, ncol, figsize=(20, 5.5), sharey=False, sharex=False)

    all_kl = []
    all_norm = []
    for i, ax in enumerate(fig.axes):
        if (i >= 2 and i <= 6):
            border_color = 'firebrick'
        else:
            border_color = 'k'
        ax.spines['bottom'].set_color(border_color)
        ax.spines['top'].set_color(border_color)
        ax.spines['left'].set_color(border_color)
        ax.spines['right'].set_color(border_color)
        plt.setp(ax.spines.values(), linewidth=2.5)

        ax.set_title(title[i], weight='bold', fontsize=18)
        ax.set_yticks([])
        keys = list(dict_arr[i].keys())
        values = list(dict_arr[i].values())
        norm = analysis.normalize(values)
        all_norm.append(norm)
        ax.bar(keys, norm, color='xkcd:mid blue', alpha=0.75)
        
        if x_gen is not None:
            if '$' in list(gen_dict_arr[i].keys()):
                gen_dict_arr[i].pop('$')
            g_keys = list(gen_dict_arr[i].keys())
            g_values = list(gen_dict_arr[i].values())
            g_norm = analysis.normalize(g_values)
            ax.bar(g_keys, g_norm, color=c, alpha=0.75)

            kl = scipy.special.kl_div(norm, g_norm)
            kl = [k for k in kl if k !=float('inf')]
            kl = sum(kl)
            all_kl.append(kl)
            kl_label = "$KL$=%.3f" % (kl)
            ax.text(0.65, 0.95, kl_label, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')

            for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(16)

    if x_gen is not None:
        print("KL:", np.mean(all_kl))
    fig.axes[0].set_ylabel('Frequency', fontsize=18)
    fig.axes[5].set_ylabel('Frequency', fontsize=18)
    fig.tight_layout()
    save_plot = os.path.join(save_path, save_file)
    plt.savefig(save_plot)
    if x_gen is None:
        char2idx = {u: i for i, u in enumerate(keys)}
        idx2char = np.array(list(keys))
        return all_norm, char2idx, idx2char


def z_score_distribution(y_train, y_pred, mmps, save_path):
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(18, 20))
    for i, ax in enumerate(axes.ravel()[:18]):
        ax2 = ax.twinx()
        ax.hist(y_train[:, i], bins=20, color='xkcd:mid blue', label='train', alpha=0.75)
        ax2.hist(y_pred[:,i], bins=20, color='xkcd:pea', label='generated', alpha=0.75)
        ax.set_xlabel('Z-score')
        ax.set_ylabel('Count')
        ax.set_title(mmps[i], weight='bold', fontsize=16)
        ax2.axis('off')
        ax.set_yticks([])
    fig.delaxes(axes[4][2])
    fig.delaxes(axes[4][3])
    fig.tight_layout()
    save_plot = os.path.join(save_path, 'z_score_distribution.svg')
    plt.savefig(save_plot)


def plot_auc(y, y_hat, mmp_index, mmps, save_path):
    z_score_list = [0.0, 1.0, 1.5, 2.0, 2.5]
    alphas = [1, 0.8, 0.6, 0.4, 0.2]
    fig, ax = plt.subplots()
    for i, z_score_cutoff in enumerate(z_score_list):
        fpr, tpr = analysis.get_roc_data(y, y_hat, true_z_cutoff=z_score_cutoff)
        auc = analysis.calc_auc(fpr, tpr)
        label = " Z-score:{}, AUC:{:2.2}".format(z_score_cutoff, auc)

        ax.plot(fpr, tpr, label=label, c='#124b9a', alpha=alphas[i])
        ax.set_title(mmps[mmp_index])
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel('True positive rate')
        ax.set_xlabel('False positive rate')
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.legend(loc='lower right')
    save_plot = os.path.join(save_path, mmps[mmp_index]+'_auc.svg')
    plt.savefig(save_plot)


def aa_frequencies(zscore):
    """
    Calculate amino acid frequencies at each position in a set of aligned sequences.

    Parameters:
    - zscore (pd.DataFrame): DataFrame containing a column 'sequence' with aligned amino acid sequences.

    Returns:
    - pd.DataFrame: A DataFrame with amino acids as rows and sequence positions as columns,
                    representing the frequency of each amino acid at each position.
    """
    # Step 1: Initialize parameters
    sequence_length = len(zscore['sequence'].iloc[0])  # Assuming all sequences are the same length
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # Step 2: Create a frequency matrix initialized to zero
    frequency_matrix = pd.DataFrame(0, index=list(amino_acids), columns=range(1, sequence_length + 1))

    # Step 3: Populate the frequency matrix
    for sequence in zscore['sequence']:
        for position, amino_acid in enumerate(sequence, start=1):
            if amino_acid in amino_acids:  # Only process valid amino acids
                frequency_matrix.loc[amino_acid, position] += 1
            else:
                print(f"Warning: Unexpected character '{amino_acid}' in sequence '{sequence}' at position {position}")

    # Step 4: Normalize the frequencies by the total number of sequences
    frequency_matrix = frequency_matrix.div(len(zscore))

    # Step 5: Reorder the rows based on amino acid properties
    reordered_amino_acids = [
        'F', 'W', 'Y',  # Hydrophobic Aromatic
        'P', 'A', 'I', 'L', 'M', 'V', 'G',  # Hydrophobic
        'C', 'N', 'Q', 'S', 'T',  # Hydrophilic
        'D', 'E', 'H',  # Acidic
        'K', 'R'  # Basic
    ]
    frequency_matrix = frequency_matrix.reindex(reordered_amino_acids)

    # Step 6: Rename the columns to reflect sequence positions
    frequency_matrix.columns = ['P5', 'P4', 'P3', 'P2', 'P1', "P1'", "P2'", "P3'", "P4'", "P5'"]

    return frequency_matrix


def aa_frequencies(zscore):
    """
    Calculate amino acid frequencies at each position in a set of aligned sequences.

    Parameters:
    - zscore (pd.DataFrame): DataFrame containing a column 'sequence' with aligned amino acid sequences.

    Returns:
    - pd.DataFrame: A DataFrame with amino acids as rows and sequence positions as columns,
                    representing the frequency of each amino acid at each position.
    """
    # Step 1: Initialize parameters
    sequence_length = len(zscore['sequence'].iloc[0])  # Assuming all sequences are the same length
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # Step 2: Create a frequency matrix initialized to zero
    frequency_matrix = pd.DataFrame(0, index=list(amino_acids), columns=range(1, sequence_length + 1))

    # Step 3: Populate the frequency matrix
    for sequence in zscore['sequence']:
        for position, amino_acid in enumerate(sequence, start=1):
            if amino_acid in amino_acids:  # Only process valid amino acids
                frequency_matrix.loc[amino_acid, position] += 1
            else:
                print(f"Warning: Unexpected character '{amino_acid}' in sequence '{sequence}' at position {position}")

    # Step 4: Normalize the frequencies by the total number of sequences
    frequency_matrix = frequency_matrix.div(len(zscore))

    # Step 5: Reorder the rows based on amino acid properties
    reordered_amino_acids = [
        'F', 'W', 'Y',  # Hydrophobic Aromatic
        'P', 'A', 'I', 'L', 'M', 'V', 'G',  # Hydrophobic
        'C', 'N', 'Q', 'S', 'T',  # Hydrophilic
        'D', 'E', 'H',  # Acidic
        'K', 'R'  # Basic
    ]
    frequency_matrix = frequency_matrix.reindex(reordered_amino_acids)

    # Step 6: Rename the columns to reflect sequence positions
    frequency_matrix.columns = ['P5', 'P4', 'P3', 'P2', 'P1', "P1'", "P2'", "P3'", "P4'", "P5'"]

    return frequency_matrix


def normalize_matrix(frequency_matrix, reference_df):
    """
    Normalize the frequency matrix using a reference DataFrame.

    Parameters:
    - frequency_matrix (pd.DataFrame): DataFrame where rows represent amino acids and columns represent sequence positions, containing frequencies.
    - reference_df (pd.DataFrame): DataFrame with the same shape as frequency_matrix, representing the reference background frequencies.

    Returns:
    - pd.DataFrame: Information content matrix after normalization.
    """

    def calculate_ic(frequency: float, background: float) -> float:
        """Calculate information content given frequency and background frequency."""
        return frequency * np.log2(frequency / background) if frequency > 0 else 0

    assert frequency_matrix.shape == reference_df.shape
    assert (frequency_matrix.index == reference_df.index).all()
    assert (frequency_matrix.columns == reference_df.columns).all()

    ic_matrix_normalized = frequency_matrix.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in frequency_matrix.index:
        for j in frequency_matrix.columns:
            ic_matrix_normalized.at[i, j] = calculate_ic(frequency_matrix.at[i, j], reference_df.at[i, j])

    return ic_matrix_normalized


def plot_icelogo(frequency_matrix, save_path, custom_color_scheme, source=None, name=None):
    """
    Generate and save different types of sequence logos based on frequency and information content.

    Parameters:
    - frequency_matrix (pd.DataFrame): DataFrame where rows represent amino acids and columns represent sequence positions, containing frequencies.
    - bg_df (pd.DataFrame): DataFrame with the same shape as frequency_matrix, representing background frequencies in training.
    - nat_df (pd.DataFrame): DataFrame with the same shape as frequency_matrix, representing background frequencies in natural protein sequences.
    - save_path (str): Directory path where the logo images will be saved.
    - custom_color_scheme (dict): Dictionary containing color for each amino acid.
    - source (str, optional): Name from data source
    - name (str, optional): Name used for the logo files and titles.

    Returns:
    - None: The function saves the sequence logos to the specified path.
    """

    # 1. Generate and save un-normalized sequence logo
    freq_logo = logomaker.Logo(frequency_matrix,
                               shade_below=.5,
                               fade_below=.5,
                               font_name='Arial Rounded MT Bold',
                               color_scheme=custom_color_scheme, figsize=(6,2))
    freq_logo.style_spines(visible=False)
    freq_logo.style_spines(spines=['left', 'bottom'], visible=True)
    freq_logo.style_xticks(rotation=90, fmt='%d', anchor=0)
    freq_logo.ax.set_ylabel(f"Frequency - {name}", labelpad=-1)
    freq_logo.ax.set_xticklabels(['P5', 'P4', 'P3', 'P2', 'P1', "P1'", "P2'", "P3'", "P4'", "P5'"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{source}_logo_{name}.png"), dpi=300)