""" Functions for evaluate the trained DNACipherModel performance
"""

import time

import dnacipher_train.visual.helpers as vhs

import seaborn as sns
import matplotlib.pyplot as plt

import torch

import json

import math
import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests

from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon

import dnacipher_train.train.helpers as helpers
import dnacipher_train.helpers.utils as utils

def eval(run_name, genome_file, sample_file, signal_file, embed_file,
               weights_path, out_dir, model_config_file, ct_knn_json_file,
               # train parameters, need to change depending on resources.
               batch_size, cpu, device, eval_batches):
    """ Runs model evaluation using basic benchmarks of performance.
    """

    #### Output log file
    log_file = open(f'{out_dir}{run_name}_log_file.txt', 'w')

    print(f'DNACipher evaluation run: {run_name}\n', file=log_file, flush=True)

    print(f'With inputs: \ngenome_file={genome_file}\nsample_file={sample_file}\nsignal_file={signal_file}\n'
          f'embed_file={embed_file}\nout_dir={out_dir}\nweights_path={weights_path}\n'
          f'model_config_file={model_config_file}\neval_batches={eval_batches}\n',
          file=log_file, flush=True)

    script_start_time = time.time()

    dataset_split, model = helpers.model_and_dataset_setup(genome_file, sample_file, signal_file, embed_file,
                                                           batch_size, cpu, device, model_config_file, log_file,
                                                           add_exper_weights=False # Not required.
                                                           )
    if not model.softplus_output: # Only doing this if using the old preprint training paradigm, have updated to be less susceptible to issues.
        model.relu_output = True
        model.stratified_loss = True

    print(f'Loaded DNACipher model with random weights: {model}\n', file=log_file, flush=True)

    print("\nLoading the inputted model weights..\n", file=log_file, flush=True)
    weights = torch.load(f'{weights_path}', map_location=torch.device(device))

    # Load the weights to the model
    model.load_state_dict( weights )

    # Loading in the kNN cell types for eval, if provided.
    if type(ct_knn_json_file)==type(None):
        ct_knn_dict = None

    else:
        print("\nWill run eval with cell type kNN benchmark...\n", file=log_file, flush=True)
        with open(ct_knn_json_file, "r") as f:
            ct_knn_dict = json.load(f)

    # Now performing the key evaluations:
    eval_model(run_name, model, dataset_split, log_file, out_dir, eval_batches, batch_size, ct_knn_dict)

    # Fin.
    utils.finalize_log(log_file, script_start_time)

def eval_model(run_name, model, dataset_split, log_file, out_dir, eval_batches, batch_size, ct_knn_dict=None):
    """ Runs evaluation of the inputted model.
    """
    print("\nRunning more extensive performance evaluations\n", file=log_file, flush=True)
    model = model.eval()

    for (region_train, exper_train), (dataset, data_loader) in dataset_split.items():
        seqs = 'TRAIN' if region_train else 'TEST'
        expers = 'TRAIN' if exper_train else 'TEST'
        print(f"#####################################################################", file=log_file, flush=True)
        print(f"  Benchmark results for {seqs} sequences on {expers} celltype, assays: ", file=log_file, flush=True)
        print(f"#####################################################################", file=log_file, flush=True)
        run_bench(run_name, model, data_loader, log_file, out_dir,
                  dataset_split[(region_train, True)][0], # Need this to determine kNN celltypes !!
                  alloc=f'_seq{seqs}-samp{expers}', eval_epochs=1,
                  eval_batches=eval_batches, batch_size=batch_size, ct_knn_dict=ct_knn_dict,
                 )

def visualize_losses(run_name, description, out_dir, loss_callback, log_file):
    """ Visualising the train/test loss throughout training.
    """
    ##### Train/Test MSE
    train_losses = loss_callback.train_losses
    test_losses = loss_callback.val_losses
    lrs = loss_callback.lrs

    print(f"\nTRAIN losses:\n", train_losses, '\n', file=log_file, flush=True)
    print(f"\nTEST losses:\n", test_losses, '\n', file=log_file, flush=True)
    print(f"\nLearning rates:\n", lrs, '\n', file=log_file, flush=True)

    x = list(range(len(train_losses)))

    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(x, train_losses, c='r', label='train_loss')
    ax.scatter(x, train_losses, c='r', label='train_loss')
    ax.plot(x, test_losses, c='g', label='test_loss')
    ax.scatter(x, test_losses, c='g', label='test_loss')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    vhs.dealWithPlot(True, False, True, out_dir,
                     f'{run_name}_{description}_losses.png', 300)
    print(f"Saved ", out_dir + f'{run_name}_{description}_losses.png', file=log_file, flush=True)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x, lrs, c='grey', label='train_loss')
    ax.scatter(x, lrs, c='k', label='train_loss')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate')
    vhs.dealWithPlot(True, False, True, out_dir,
                     f'{run_name}_{description}_lrs.png', 300)
    print(f"Saved ", out_dir + f'{run_name}_{description}_lrs.png', file=log_file, flush=True)

def run_bench(TEST_NAME, model, data_loader, log_file, out_dir, train_data, alloc='',
              eval_epochs=1, eval_batches=None, batch_size=20,
              knn=10, # Number of knn cell types to use!
              ct_knn_dict=None,
              ):
    """ Runs benchmarking on the per-seq model, via correlations.
    """

    if type(eval_batches)==type(None) or eval_batches > len(data_loader):
        eval_batches = len(data_loader)

    dataset = data_loader.dataset.signal_data

    ##### Need to stratify by celltype, assay to see how well performs across...
    celltypes, assays = dataset.celltypes, dataset.assays
    celltype_assay_indices = {} # Mapping from the celltype_index, assay_index to a celltype_assay_index!
    assay_indices = np.zeros((len(dataset.load_celltype_assays)), dtype=np.int64) # Contains the assay index for each column, so can subset to celltypes with measurements for that assay.
    celltype_indices = np.zeros((len(dataset.load_celltype_assays)), dtype=np.int64)
    for i, celltype_assay in enumerate(dataset.load_celltype_assays):
        celltype, assay = celltype_assay.split('---')
        celltype_index = celltypes.index( celltype )

        assay_index = assays.index( assay )
        assay_indices[ i ] = assay_index

        celltype_indices[i] = celltype_index

        celltype_assay_indices[(celltype_index, assay_index)] = i

    if type(ct_knn_dict)!=type(None): # Going to use this information...
        # Creating a map of the desired kNN cell types to use for each celltype...
        ct_index_to_nn_indices = {}
        for ct_, knn_ in ct_knn_dict.items():
            ct_index_to_nn_indices[celltypes.index(ct_)] = [celltypes.index(nn_ct) for nn_ct in knn_]

    # Determining the mappings from the assay_indices to the relevant columns in the train_data so we can determine
    # kNN cell types !!!
    assay_index_set = np.unique(assay_indices)

    assay_indices_train = np.array( [assays.index( ct_assay.split('---')[1] )
                                     for ct_assay in train_data.signal_data.load_celltype_assays] )
    celltype_indices_train = np.array( [celltypes.index( ct_assay.split('---')[0] )
                                     for ct_assay in train_data.signal_data.load_celltype_assays] )

    assay_index_to_ct_indices = {index: np.where(assay_indices==index)[0] for index in assay_index_set}
    assay_index_to_ct_indices_train = {index: np.where(assay_indices_train==index)[0] for index in assay_index_set}

    # Calculating number of points we will use for evaluation....
    eval_points = int(dataset.load_n_expers * batch_size * eval_batches * eval_epochs)

    # The overall MSE
    expected_mse = np.zeros((eval_batches))
    expected_mse_knn = np.zeros((eval_batches))
    observed_mse = np.zeros((eval_batches))

    # The is the actual correlation of the SHAPE across the sequence if you were to use the kNN cell types as the predictor !!
    expected_corr_knn_assay = np.zeros((eval_batches, len(assay_index_set)))
    observed_corr_assay = np.zeros((eval_batches, len(assay_index_set)))

    start_seq = 0
    start_region = 0
    pred = np.zeros((eval_points))
    y_test = np.zeros((eval_points))
    test_celltype_assay_indices = np.zeros((eval_points), dtype=int)
    for eval_epoch in range(eval_epochs):
        for i, (inputs, actual_) in enumerate(data_loader):
            if i >= eval_batches: # Only collect data for a certain number of batches.
                break

            i_global = i + (eval_batches * eval_epoch)  # global batch number

            actual = actual_.numpy().ravel()

            # NOTE the _seq are referring to per epigenetic value, which are ACROSS the celltype,assays
            # start_region, end_region refer to the actual sequences.
            end_seq = start_seq + len(actual)

            y_test[start_seq:end_seq] = actual

            #### Storing the batch celltype, assay information.
            batch_celltype_indices = inputs['celltype_input'].cpu().numpy().ravel()
            batch_assay_indices = inputs['assay_input'].cpu().numpy().ravel()
            batch_celltype_assay_indices = [celltype_assay_indices[(celltype, assay)] for (celltype, assay) \
                                            in zip(batch_celltype_indices, batch_assay_indices)]
            test_celltype_assay_indices[start_seq:end_seq] = batch_celltype_assay_indices

            ### Will just get the average within the batch as the expected. This will be better than the global average...
            # NOTE not currently stratifying by celltype,assay for this test, but will for the down-stream correlations.
            expected = np.array([actual.mean(axis=0)] * len(actual))
            expected_mse[i_global] = mean_squared_error(actual, expected)

            # Getting the inputs to predict, need to filter out extra information outputted from dataset.
            dnac_inputs = [input_.to(model.device) for key, input_ in inputs.items() if '_input' in key]

            track_value_pred = model(*dnac_inputs).to( 'cpu' ).detach().numpy().ravel()
            observed_mse[ i_global ] = mean_squared_error(track_value_pred, actual)

            # Now also getting a knn expected MSE, based on the most similar cell types within the batch...
            # For each cell type measured for a given assay, find the knn cell types, and use that as the guess
            # for the cell types measurements!
            #### Retrieving the training measurements ct/assays
            if 'sampTrain' not in alloc: # Testing, so we need to load these values from the train_data
                _, train_actual = train_data.signal_data[list( inputs['region_indices'][0] )]
            else: # We have already loaded the relevant values! so just use them.
                train_actual = actual_

            actual_numpy = actual_.numpy()

            # Checking reshape operation is correct:
            # torch.all(actual.reshape(actual_.shape) == actual_)
            # Storing the results PER ASSAY so can make it clear which assays are out-performing the mean-baseline !!!
            if type(ct_knn_dict)!=type(None): # Don't do this if we have not defined nearest neighbours..

                track_value_pred_square = track_value_pred.reshape(actual_.shape)

                expected_knn = np.zeros(actual_.shape)
                for assay_i, (assay_index, assay_ct_indices) in enumerate( assay_index_to_ct_indices.items() ):

                    train_assay_ct_indices = assay_index_to_ct_indices_train[ assay_index ]

                    for ct_assay_index in assay_ct_indices: # For this cell type

                        # the actual ct index
                        ct_index = celltype_indices_train[ ct_assay_index ]

                        # Measurements for other cell types in the training data.
                        other_assay_ct_indices = train_assay_ct_indices[train_assay_ct_indices!=ct_assay_index]
                        other_ct_indices = celltype_indices_train[other_assay_ct_indices]

                        ct_knn_indices = [index for index in ct_index_to_nn_indices[ct_index] if index in other_ct_indices][:knn]
                        ct_assay_knn_indices = [ct_assay_index_ for ct_assay_index_, ct_index_ in zip(other_assay_ct_indices,
                                                                                                      other_ct_indices)
                                                        if ct_index_ in ct_knn_indices]
                        # check correct:
                        #np.array(train_data.signal_data.load_celltype_assays)[ct_assay_knn_indices]
                        actual_other_celltypes = train_actual[:, ct_assay_knn_indices]

                        # Finding the kNN cell types....
                        # dot_sim = np.dot(actual_other_celltypes, actual_celltype)[:,0]
                        # order = np.argsort(-dot_sim)
                        # knn_indices = order[:knn]
                        expected_knn[:, ct_assay_index] = actual_other_celltypes.mean(axis=1).numpy()

                    #### Storing per-assay information, for both the kNN expected and the predicted values.
                    # Testing Pearsonr instead, to reflect if it is capturing the shape better rather than
                    # just the average magnitude.
                    expected_corr_knn_assay[i_global, assay_i] = pearsonr(expected_knn[:, assay_ct_indices].ravel(),
                                                                         actual_numpy[:, assay_ct_indices].ravel()
                                                                         )[0]
                    observed_corr_assay[i_global, assay_i] = pearsonr(track_value_pred_square[:, assay_ct_indices].ravel(),
                                                                     actual_numpy[:, assay_ct_indices].ravel()
                                                                     )[0]

                    ##### And the overall mean_square_error for kNN (!!!)
                    expected_knn_flat = expected_knn.ravel()
                    expected_mse_knn[ i_global ] = mean_squared_error(expected_knn_flat, actual)

                # Plot how these compare:
                # plt.scatter(expected_corr_knn_assay[i_global, :], observed_corr_assay[i_global, :])
                # plt.plot(expected_corr_knn_assay[i_global, :], expected_corr_knn_assay[i_global, :], c='k')
                # plt.xlabel(f"Assay expected R from kNN celltypes")
                # plt.ylabel(f"Assay observed R from prediction")
                # plt.show()

            pred[start_seq:end_seq] = track_value_pred

            end_region = start_region + dnac_inputs[2].shape[0]

            start_seq = end_seq
            start_region = end_region

            if i%10 == 0:
                print(f"Eval: {alloc} evaluated {end_region}/{len(dataset)} at eval epoch {eval_epoch+1}/{eval_epochs}",
                  file=log_file, flush=True)

    #### Quantifying correlations...
    corrs = []
    spearmans = []

    ###### Making a matrix to store the Pearson correlations..
    celltype_assay_corrs = pd.DataFrame(np.full((len(celltypes), len(assays)), fill_value=np.nan),
                                        index=celltypes, columns=assays)
    celltype_assay_spears = pd.DataFrame(np.full((len(celltypes), len(assays)), fill_value=np.nan),
                                         index=celltypes, columns=assays)
    celltype_assay_counts = np.full((len(celltypes), len(assays)), fill_value=1/(len(pred)*4)
                                    ) # Fill with pseudocounts
    celltype_assay_true_nonzero = np.zeros((len(celltypes), len(assays))) # Counting number of nonzero observations.
    celltype_assay_pred_nonzero = np.zeros((len(celltypes), len(assays)))  # Count no. of pred nonzero observations.
    present_bool = np.full((len(celltypes), len(assays)), fill_value=False)
    for celltype_assay_index, celltype_assay in enumerate( dataset.load_celltype_assays ):
        celltype, assay = celltype_assay.split('---')
        label = f'{celltype}_{assay}'
        label_bool = test_celltype_assay_indices == celltype_assay_index

        celltype_index_ = celltypes.index(celltype)
        assay_index_ = assays.index(assay)
        n_times_seen = len( np.where( label_bool )[0] )
        present_bool[celltype_index_, assay_index_] = True

        if n_times_seen > 0:
            celltype_assay_counts[celltype_index_, assay_index_] = n_times_seen

        if celltype_assay_counts[celltype_index_, assay_index_]<=3: # Fixed bug where might not have enough example of celltype, assay....
            print(f"Warning, {celltype}/{assay} not represented for benchmarking!")
            corr_ = 0
            spearman_ = 0
            continue

        pred_vals, true_vals =  pred[label_bool], y_test[label_bool]
        # Quantifying non-zero cases used for the benchmarking
        celltype_assay_pred_nonzero[celltype_index_, assay_index_] = len( np.where(pred_vals > 0)[0] )
        celltype_assay_true_nonzero[celltype_index_, assay_index_] = len(np.where(true_vals > 0)[0])

        corr_, spearman_ = quantify_corrs(label, pred_vals, true_vals, log_file)
        corrs.append(corr_)
        spearmans.append(spearman_)

        celltype_assay_corrs.iloc[celltype_index_, assay_index_] = corr_
        celltype_assay_spears.iloc[celltype_index_, assay_index_] = spearman_

    #### Key benchmarking visualises and output.
    quantify_corrs('GLOBAL', pred, y_test, log_file, log=True)
    print(f'\nStratified Pearson correlations with mean {round(np.nanmean(corrs), 4)}:', file=log_file, flush=True)
    celltype_assay_corrs.to_csv(out_dir + f'{TEST_NAME}{alloc}_corrs.txt', sep='\t')
    print(f"Saved ", out_dir + f'{TEST_NAME}{alloc}_corrs.txt\n\n', file=log_file, flush=True)

    print(f'Stratified Spearman correlations with mean {round(np.nanmean(spearmans), 4)}:', file=log_file, flush=True)
    celltype_assay_spears.to_csv(out_dir+f'{TEST_NAME}{alloc}_spears.txt', sep='\t')
    print(f"Saved ", out_dir+f'{TEST_NAME}{alloc}_spears.txt', file=log_file, flush=True)

    #### Baseline test to see if outperforms MSE by just taking the mean of the values as a predictor of itself:
    run_baseline_ttest(expected_mse, observed_mse, TEST_NAME, out_dir, log_file, alloc=alloc)

    # Rest of the base-line requires having defined kNN cell types, therefore skip this if not.
    if type(ct_knn_dict)==type(None):
        return

    run_baseline_ttest(expected_mse_knn, observed_mse, TEST_NAME, out_dir, log_file, alloc=f"_kNN{alloc}")

    assay_names = np.array( assays )[ assay_index_set ]

    ### Plotting the result of these in batches as a supplementary figure (!!!!)
    assay_stats = np.zeros( (len(assay_names), 6) ) #mu_observed, mu_expected, delta, w, z_stat, p_val
    keep_ = np.full((assay_stats.shape[0]), fill_value=True)
    for assayi, assay_name in enumerate( assay_names ):
        assay_corrs_expected = expected_corr_knn_assay[:, assayi]
        assay_corrs_observed = observed_corr_assay[:, assayi]
        # If they were both non_nan, it means a case where the ct_assay being compared is not all 0's
        expected_nan = np.isnan(assay_corrs_expected)
        observed_nan = np.isnan(assay_corrs_observed)
        non_nan = np.logical_and(expected_nan, observed_nan)==False

        assay_corrs_expected[np.logical_and(expected_nan, observed_nan == False)] = 0 # 0 correlation if outputs a constant, but was signal at locus.
        assay_corrs_observed[np.logical_and(observed_nan, expected_nan == False)] = 0

        out = wilcoxon(assay_corrs_expected[non_nan], assay_corrs_observed[non_nan],
                       method='asymptotic')
        w, zstat, pval = out.statistic, out.zstatistic, out.pvalue # note z_stat is difference under null, NOT between the groups.
        # t_stat, pval = ttest_rel(assay_corrs_expected[non_nan], assay_corrs_observed[non_nan],
        #                          alternative='two-sided')
        ##### Collecting other stats.
        expected_mean_corr = np.nanmean(assay_corrs_expected)
        observed_mean_corr = np.nanmean(assay_corrs_observed)
        mean_delta = np.mean(assay_corrs_observed[non_nan] - assay_corrs_expected[non_nan])
        if np.isnan(zstat): # Usually because too small numbers to compare.
            keep_[assayi] = False
            assay_stats[assayi, :] = np.nan
            continue

        assay_stats[assayi, :] = [observed_mean_corr, expected_mean_corr, mean_delta, w, zstat, pval]

    assay_stats_df = pd.DataFrame(assay_stats, columns=['observed_mean_corr', 'kNN_mean_corr', 'mean_delta',
                                                        'w', 'null_z_stat', 'pval'], index=assay_names)
    assay_stats_df.loc[keep_, 'padj'] = multipletests(assay_stats_df.loc[keep_, 'pval'].values, method='fdr_bh')[1]

    stats_file_name = f"{out_dir}{TEST_NAME}{alloc}_assay_kNN_baseline_stats.txt"
    assay_stats_df.to_csv(stats_file_name, sep='\t')
    print(f"Saved ", f'{stats_file_name}', file=log_file, flush=True)

    ###### Plotting the results per assay !!!!!
    nrows, ncols = 4, 4
    total_panels_per_fig = int(nrows * ncols)

    fig = None
    figi = 0
    for assayi, assay_name in enumerate( assay_names ):

        assay_alloc = f"_{assay_name}{alloc}"

        if assayi % total_panels_per_fig == 0:
            if type(fig) != type(None): # Save previous figure.
                file_name_ = f'{TEST_NAME}_{fig_name}_obsVSexp_violin.png'
                vhs.dealWithPlot(True, False, True, out_dir, file_name_, 300,
                                 tightLayout=True)

            # Create new figure.
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
            axes = axes.ravel()
            axi = 0
            fig_name = f"assayFig-{figi}{alloc}"
            figi += 1

        ax = axes[axi]
        run_baseline_ttest(expected_corr_knn_assay[:, assayi], observed_corr_assay[:, assayi],
                                          TEST_NAME, out_dir, log_file,
                                           alloc=assay_alloc, verbose=False, save=False, ax=ax, measure='R')
        padj = assay_stats_df.loc[assay_name, 'padj']
        mean_delta = assay_stats_df.loc[assay_name, 'mean_delta']
        if padj < 0.001:
            padj_stars = f"***"
        elif padj < 0.01:
            padj_stars = f"**"
        elif padj < 0.05:
            padj_stars = f"*"
        else:
            padj_stars = f""

        padj_str = f"mean_delta={mean_delta:.3E}\n{padj_stars}padj={padj:.3E}"

        mid_x = np.mean(ax.get_xlim())*.35
        upper_y = 0.80 * ax.get_ylim()[1]

        ax.text(mid_x, upper_y, padj_str)
        ax.set_ylabel( assay_name )

        axi += 1

        if assayi == len(assay_names)-1: # Last assay plotted, so save it!
            file_name_ = f'{TEST_NAME}_{fig_name}_obsVSexp_violin.png'
            vhs.dealWithPlot(True, False, True, out_dir, file_name_, 300,
                             tightLayout=True)

    file_name_x = file_name_.replace(f"assayFig-{figi}", f"assayFig-*")
    print(f"Saved ", out_dir + f'{file_name_x}', file=log_file, flush=True)

def mean_squared_error(actual, expected, dim=None):
    mse = ((actual - expected) ** 2).mean(dim)
    return mse

def quantify_corrs(label, pred, y_test, log_file, log=False):
    """ Quantifies the correlations between the predicted and actual epi-measures!
    """
    corr_results = pearsonr(pred, y_test)
    spearman_results = spearmanr(pred, y_test)

    if log:
        print(f"\n{label} Pearson results:", corr_results, file=log_file, flush=True)
        print(f"{label} Spearman results:", spearman_results, file=log_file, flush=True)

    return corr_results[0], spearman_results[0]

def run_baseline_ttest(expected_mse, observed_mse,
                       TEST_NAME, out_dir, log_file, alloc='', verbose=True, save=True, ax=None, measure='MSE'):
    """Runs a baseline significance test to determine if model performing significantly better than guessing the average
        epigenetic signals across the sequences.
    """
    # Create the violin plot, Combine the data into a pandas DataFrame for easy plotting
    df = pd.DataFrame({f"expected {measure}": expected_mse, f"observed {measure}": observed_mse})

    if type(ax) == type(None):
        fig, ax = plt.subplots()

    sns.violinplot(data=df, ax=ax)
    for i in np.random.randint(0, len(expected_mse), 100):  # Add lines to connect paired data
        ax.plot([0, 1], [expected_mse[i], observed_mse[i]], 'k-', alpha=0.3)
    ax.set_title('Violin plot with paired data')
    if save:
        vhs.dealWithPlot(True, False, True, out_dir, f'{TEST_NAME}{alloc}_obsVSexp_violin.png', 300)

    # Printing significance via paired t-test (this is two-tailed)
    t_stat, p_val = ttest_rel(observed_mse, expected_mse, alternative='less')

    if verbose:
        print("\nMean expected MSE:", np.mean(expected_mse), "Mean observed MSE:", np.mean(observed_mse),
                                                                                                  file=log_file, flush=True)
        print(f"T-statistic: {t_stat}", file=log_file, flush=True)
        print(f"P-value: {p_val}\n", file=log_file, flush=True)
        if save:
            print(f"Saved ", out_dir + f'{TEST_NAME}{alloc}_obsVSexp_violin.png\n', file=log_file, flush=True)

    return t_stat, p_val
