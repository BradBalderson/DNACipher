""" Functions for performing the Deep Variant Impact Mapping analysis.
"""

import sys

import numpy as np
import pandas as pd

from numba import jit

def stratify_variants(signal_gwas_stats, 
                      var_ref_col='other_allele', var_alt_col='effect_allele', var_loc_col='base_pair_location',
                      p_col='p_value', allele_freq_col='effect_allele_frequency',
                      p_cut=5e-7, lowsig_cut=0.001, n_top=10, allele_freq_cut=.05, min_bg_variants=100,
                      verbose=True):
    """ Performs stratification of variants at GWAS loci to categories: 
        'candidate' variants (common significant variants), 'rare' variants (non-significant rare variants in the same region as the candidate variants), 'background' variants (common non-significant variants), and 'other' variants (rare variants outside of the hit locus).
        
        Parameters
        ----------
        signal_gwas_stats: pd.DataFrame
            GWAS summary statistics for each of the variants at a GWAS locus, within range model predictions.
        var_ref_col: str
            Column in the input dataframe that specifies the variant reference sequence as a string.
        var_alt_col: str
            Column in the input dataframe that specifies the variant alternate sequence as a string.
        var_loc_col: str
            Column in the input dataframe that specifies the variant position as an integer.
        p_col: str
            Name of column in the input dataframe that contains the p-values of the variant-trait associations.
        allele_freq_col: str
            Column in the input dataframe that specifies the variant allele frequency as a float.
        p_cut: float
            P-value cutoff to consider a variant significantly associated with the trait.
        lowsig_cut: float
             Cutoff to consider variants confidently not-significant.
        n_top: int
            If no significant variants, will take this many as the top candidates.
        allele_freq_cut: float 
            If a variant is above this minor allele frequency AND is considered confidently non-significant, then is considered a 'background' variant. If is below this allele frequency, then considered a rare variant.
        min_bg_variants: int
            If have less than this number of background variants, will rank-order potential background variants by scoring
            allele frequency and significance, and take this many variants as significant.
            
        Returns
        --------
        locus_gwas_stats_annotated: pd.DataFrame
            Equivalent to the input dataframe, except with 'var_label' column labelling the variants, and also filtering
            out 'other' variants and indels not currently supported.
    """

    sig_var_bool = signal_gwas_stats[ p_col ] < p_cut
    sig_locs = signal_gwas_stats[ var_loc_col ].values[ sig_var_bool ]

    if len(sig_locs) == 0: # No variants meet the threshold, so will take the top variants as the candidates.
        order = np.argsort(signal_gwas_stats[ p_col ].values)
        sig_var_bool = np.array([vari in order[0:n_top] for vari in range(signal_gwas_stats.shape[0])])
        sig_locs = signal_gwas_stats[ var_loc_col ].values[sig_var_bool]
        
    ### Defining the range of where the signal occurs, based on the range of significant variants.
    sig_range = [np.min(sig_locs), np.max(sig_locs)]

    # Getting rare variants that are within range!!!!
    rare_variant_bool = np.logical_or(signal_gwas_stats[ allele_freq_col ].values < allele_freq_cut,
                                      signal_gwas_stats[ allele_freq_col ].values > (1-allele_freq_cut),
                                 )

    locs = signal_gwas_stats[ var_loc_col ].values
    rare_variant_bool = np.logical_and(rare_variant_bool, locs >= sig_range[0])
    rare_variant_bool = np.logical_and(rare_variant_bool, locs <= sig_range[1])
    
    ### For the background variants, we do NOT want to include rare variants, since they MAY have an effect, BUT
    ### were not significant due to lacking enough observations for sufficient power.
    nonsig_and_not_candidate_rare = np.logical_and(signal_gwas_stats[ p_col ] > lowsig_cut,
                                         rare_variant_bool == False # Don't want to consider under-powered variants in background!
                                )
    # Also making sure they aren't rare variants being selected for the background, so they are confident negatives !!!
    allele_freqs_ = signal_gwas_stats[ allele_freq_col ].values 
    not_rare = np.logical_and(allele_freqs_ > allele_freq_cut,
                              allele_freqs_ < (1-allele_freq_cut))
    
    lowsig_var_bool = np.logical_and(nonsig_and_not_candidate_rare, not_rare)
    
    # Too few background variants.
    if sum(lowsig_var_bool) < min_bg_variants:
        # Lower the allele_freq_cut criteria:
        lowsig_var_bool = nonsig_and_not_candidate_rare
        
        low_sig_var_indices = np.where(lowsig_var_bool)[0]
        
        bg_candidates_allele_freq_order = np.argsort(-allele_freqs_[ lowsig_var_bool ])
        
        ### Defining the candidate background variants we want to set to False cause not in top X
        rare_indices = np.array([lowsig_vari for vari, lowsig_vari in enumerate(low_sig_var_indices) 
                                 if vari not in bg_candidates_allele_freq_order[0:min_bg_variants]])
        
        # Update the background varaints, based on taking the most common non-sig variants.
        # Only want common variants as the background, since confident NOT associated with trait.
        lowsig_var_bool[rare_indices] = False 
        
    ### INDELS currently not supported, will remove indels
    other_allele = [len(allele) > 1 for allele in signal_gwas_stats[ var_ref_col ]]
    effect_allele = [len(allele) > 1 for allele in signal_gwas_stats[ var_alt_col ]]
    remove_vars = np.logical_or(effect_allele, other_allele)

    sig_var_bool[remove_vars] = False
    lowsig_var_bool[remove_vars] = False
    rare_variant_bool[remove_vars] = False

    #### Defining the remaining variants
    other_var_bool = np.logical_and(~lowsig_var_bool, ~sig_var_bool)
    other_var_bool = np.logical_and(other_var_bool, ~rare_variant_bool)
    other_var_bool = np.logical_and(other_var_bool, ~remove_vars)
    
    lowsig_gwas_stats = signal_gwas_stats.loc[lowsig_var_bool, :].copy()
    candidate_gwas_stats = signal_gwas_stats.loc[sig_var_bool, :].copy()
    rare_gwas_stats = signal_gwas_stats.loc[rare_variant_bool, :].copy()
    
    other_gwas_stats = signal_gwas_stats.loc[other_var_bool, :].copy()
    
    candidate_gwas_stats['var_label'] = 'candidate'
    lowsig_gwas_stats['var_label'] = 'background'
    rare_gwas_stats['var_label'] = 'rare'
    other_gwas_stats['var_label'] = 'other'
    
    selected_gwas_stats = pd.concat([candidate_gwas_stats, rare_gwas_stats, lowsig_gwas_stats, other_gwas_stats], ignore_index=True)
    
    if verbose:
        print('No. other variants:', other_gwas_stats.shape[0])
        print(f"Total selected varaints {sum(selected_gwas_stats['var_label'].values!='other')} / {signal_gwas_stats.shape[0]}")
        print('\tNo. candidate variants:', candidate_gwas_stats.shape[0])
        print('\tNo. rare variants:', rare_gwas_stats.shape[0])
        print('\tNo. background variants:', lowsig_gwas_stats.shape[0])

    return selected_gwas_stats

def impact_map(selected_gwas_stats, selected_pred_effects,
               cpu=3, p_cutoff=0.05, min_std=0.01, n_boots=10_000, verbose=True, pseudocount=1, fc_cutoff=0):
    """ Runs the impact mapping of the variants, with the loci specified by the seq_pos column
    """

    # All the loci are in these files, and the variant effect pvals assumes one locus, so need to separate.
    loci = selected_gwas_stats['seq_pos'].values
    loci_to_firstindex = {locus: index for index, locus in enumerate(loci)}
    cpu = min([cpu, len(loci_to_firstindex)])

    # Just confirming assumption the variants are grouped by the locus, which would enable outputting the results
    # in the same order of the inputs.
    indices = list(loci_to_firstindex.values())
    if not np.all([indices[i]>indices[i-1] for i in range(1, len(indices))]):
        raise Exception("Inputted variants are not ordered by locus, and so the outputs from this call will be a different order than the inputs..")

    # Getting a mapping from loci to index so we know the ordering
    loci_set = list( loci_to_firstindex.keys() )
    loci_per_thread = np.array_split(np.array(loci_set), cpu)

    ###### Now running the loci in parallel:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from functools import partial
    partial_func = partial(impact_map_select_loci, p_cutoff, min_std, n_boots, verbose, pseudocount, fc_cutoff,
                           selected_gwas_stats, selected_pred_effects
                           )

    with ProcessPoolExecutor(max_workers=cpu) as executor:

        futures = {executor.submit(partial_func, loci): index for index, loci in enumerate(loci_per_thread)}
        impact_results = {}
        n_threads_finished = 0

        for future in as_completed(futures):
            try:
                result = future.result()  # will re-raise exceptions from worker
                impact_results.update( result )

                n_threads_finished += 1
                if verbose:
                    print(f"Thread {n_threads_finished} / {cpu} finished.", file=sys.stdout)

            except Exception as e:
                threadi = futures[future]
                raise Exception(f"Error in {threadi} {e}.")

    # Compiling all the results from across threads.
    result_dfs_lists = [[], [], [], [], [], []] #boot_pvals_df, boot_counts_df, sig_effects, foldchange_effects, min_common_pval, min_rare_pval
    for locus in loci_set:

        result_dfs = impact_results[locus]
        for resi, result_df in enumerate( result_dfs ):

            result_dfs_lists[resi].append( result_df )

    boot_pvals_df = pd.concat( result_dfs_lists[0] )
    boot_counts_df = pd.concat( result_dfs_lists[1] )
    sig_effects = pd.concat( result_dfs_lists[2] )
    foldchange_effects = pd.concat( result_dfs_lists[3] )

    ### Calling the impact variants within loci
    n_sig_effects = (sig_effects.values).sum(axis=1)
    selected_gwas_stats['n_sig_effects'] = n_sig_effects
    selected_gwas_stats['impact_variant'] = n_sig_effects > 0

    ########## Now adding between loci MHT, so really user should just take the impact variants at significant loci
    ########## for respective variant type.
    ### Now adjusting for multiple hypothesis testing across loci, similar to eQTLs, except don't have LD problem:
    min_common_pvals = result_dfs_lists[4]
    min_rare_pvals = result_dfs_lists[5]

    from statsmodels.stats.multitest import multipletests
    common_min_signal_pvals_adj = multipletests(min_common_pvals, method='fdr_bh')[1]
    rare_min_signal_pvals_adj = multipletests(min_rare_pvals, method='fdr_bh')[1]

    # Adding this locus information to the results.
    locus_common_pvals = dict(zip(loci_set, min_common_pvals))
    locus_rare_pvals = dict(zip(loci_set, min_rare_pvals))
    locus_common_padj = dict(zip(loci_set, common_min_signal_pvals_adj))
    locus_rare_padj = dict(zip(loci_set, rare_min_signal_pvals_adj))

    selected_gwas_stats['locus_common_pval'] = [locus_common_pvals[locus] for locus in loci]
    selected_gwas_stats['locus_rare_pval'] = [locus_rare_pvals[locus] for locus in loci]
    selected_gwas_stats['locus_common_padj'] = [locus_common_padj[locus] for locus in loci]
    selected_gwas_stats['locus_rare_padj'] = [locus_rare_padj[locus] for locus in loci]

    return selected_gwas_stats, sig_effects, foldchange_effects, boot_pvals_df, boot_counts_df

def impact_map_select_loci(p_cutoff, min_std, n_boots, verbose, pseudocount, fc_cutoff,
                          selected_gwas_stats, selected_pred_effects, loci,
                         ):
    """Performs the impact mapping for just one locus.

    Meant to be run in parallel, with the note that any objects parsed will need to be pickled to start a new thread, so
    to make this minimalist will import necessary functions within the function itself.
    """

    import time

    start_ = time.time()

    impact_results = {}
    for locus_i, locus in enumerate( loci ):

        locus_indices = np.where(selected_gwas_stats['seq_pos'].values == locus)[0]

        locus_gwas_stats = selected_gwas_stats.iloc[locus_indices, :]
        locus_pred_effects = selected_pred_effects.iloc[locus_indices, :]

        boot_pvals_df, boot_counts_df = calc_variant_effect_pvals(locus_gwas_stats, locus_pred_effects,
                                                                       p_cutoff=p_cutoff, min_std=min_std,
                                                                       n_boots=n_boots, verbosity=False,
                                                                       pseudocount=pseudocount)

        sig_effects, foldchange_effects = call_sig_effects(locus_gwas_stats, locus_pred_effects, boot_pvals_df,
                                                           p_cutoff=p_cutoff, fc_cutoff=fc_cutoff)

        #### Recording the minimum p-value for the locus, for common versus rare variants:
        common_indices = np.where(locus_gwas_stats['var_label'].values == 'candidate')[0]
        rare_indices = np.where(locus_gwas_stats['var_label'].values == 'rare')[0]
        if len(common_indices) > 0:
            min_common_pval = float(np.min(boot_pvals_df.values[common_indices, :]))
        else:
            min_common_pval = 1

        if len(rare_indices) > 0:
            min_rare_pval = float(np.min(boot_pvals_df.values[rare_indices, :]))
        else:
            min_rare_pval = 1

        impact_results[locus] = [boot_pvals_df, boot_counts_df, sig_effects, foldchange_effects, min_common_pval,
                                                                                                          min_rare_pval]

        if verbose:
            end_ = time.time()
            print(f"Finished impact mapping {locus_i} / {len(loci)} loci in {round((end_-start_)/60, 3)}mins on this thread.\n",
                  file=sys.stdout, flush=True)

    return impact_results

def calc_variant_effect_pvals(selected_gwas_stats, pred_effects, n_boots=10_000, p_cutoff=0.05, pseudocount=1,
                                         min_std=0.01, verbosity=1):
    """ Calculates variant effect p-values compared to non-significant background variants. Works by boot-strapping the
        background variants, determine mean and standard deviation across predicted effects, than performing a one-sample
        z-test for each molecular effect and non-background variant, and recording if the non-background variant was significantly
        different or not. P-values becomes the proportion of times the given predicted effect for a given non-background variant
        was non-significantly different from the background variants.
    
    Parameters
    ----------
    selected_gwas_stats: pd.DataFrame
        GWAS summary statistics for each of the variants at a GWAS locus, with a column 'var_labels' indicating candidate, rare, and background variants. Each row is a variant.
    pred_effects: pd.DataFrame
        Predicted effects for each variant. Each row is a variant, and each column is a predicted molecular effect for that variant.
    n_boots: int
        No. of boot-straps of re-selecting the background variants.
    p_cutoff: float
        P-value below which a non-background variant predicted molecular effect is considered significantly different to the background vars.
    pseudocount: int
        Value added to prevent 0 p-values, defines lower-bound for minimum p-values, should be set to 1.
    min_std: float
        Minimum standard deviation for the background variant effects. Set to avoid 0 std for 0 effects of background variants causing infinite z-scores.
    verbosity:
        Verbosity levels. 0 errors only, 1 prints processing progress, 2 prints debugging information.
    """

    import time
    import math
    from scipy.stats import norm
    
    # Loading the necessary input files.
    if verbosity >= 1:
        print("Loading input files.", file=sys.stdout, flush=True)
        
    var_info_df = selected_gwas_stats
    pred_effects_df = pred_effects
    
    # Need to separate into background versus candidate / rare variants. 
    if verbosity >= 1:
        print("Separating into background and candidate / rare variants.", file=sys.stdout, flush=True)
        
    bg_bool = var_info_df['var_label'].values=='background'
    bg_indices = np.where( bg_bool )[0]
    candidate_rare_indices = np.where( ~bg_bool )[0]
    
    bg_effects = pred_effects_df.values[bg_indices, :]
    candidates_rare_effects = pred_effects_df.values[candidate_rare_indices, :]
    
    bg_abs_effects = np.abs( bg_effects )
    
    candidate_rare_abs_effects = np.abs( candidates_rare_effects )
    
    # Run the p-value testing
    boot_counts = np.zeros( pred_effects_df.shape )
    
    total_ps = candidate_rare_abs_effects.shape[0] * candidate_rare_abs_effects.shape[1]
    
    #### Decided I would separate the correction for the rare and common variants.
    adj_p = p_cutoff / total_ps # Bonferroni nominal p-value.
    critical_value = norm.ppf(1 - adj_p / 2) # Critical value for two-sided z-test.
    if verbosity >= 2: # Print boot-strap details if in debug verbosity.
        print(f"boot_counts_shape, total_ps, adj_p, critical_value: {boot_counts.shape}, {total_ps}, {adj_p}, {critical_value}", 
              file=sys.stdout, flush=True)
        
    random_number_generator = np.random.default_rng(0) # Numpy random generator

    start_ = time.time()
    for booti in range(n_boots):
        
        fast_bootstrap(boot_counts, candidate_rare_indices, candidate_rare_abs_effects, bg_abs_effects, 
                       critical_value, min_std, random_number_generator)
        
        # Old debugging outputs
        #if verbosity >= 2 and (booti % math.ceil(n_boots*0.1))==0: # Print col 0 boot strap details if debugging mode.
        #    col_sum_boot_counts = sum( boot_counts[candidate_rare_indices, 0] )
        #    print(f"boots_shape, boot_mean, boot_std, col_sum_boot_counts: {boots.shape}, {boot_mean}, {boot_std}, {col_sum_boot_counts}", 
        #          file=sys.stdout, flush=True)
            
        if verbosity >= 1 and (booti % math.ceil(n_boots*0.1))==0: # Update progress in 10% increments.
            perc_ = round(booti / n_boots, 1) * 100
            elapsed_mins = round((time.time()-start_) / 60, 1)
            print(f"Finished boot-strapping for {booti} / {n_boots} ({perc_}%) in {elapsed_mins}mins", file=sys.stdout, flush=True)

    if verbosity >= 1:
        print("Finished boot-strapping.", file=sys.stdout, flush=True)
        
    # For the background variants, fill boot_counts simply with n_boots.
    boot_counts[bg_indices, :] = n_boots
    
    # Calculating the p-values based on the number of boot-straps
    boot_pvals = (boot_counts + pseudocount) / (n_boots + pseudocount) # Pseudocount prevents incorrect 0 p-values.
    
    if verbosity >= 1:
        effects_sig = boot_pvals < p_cutoff
        total_sigs = effects_sig.sum()
        var_sigs = effects_sig.sum(axis=1)
        n_vars_sig = sum(var_sigs > 0)
        mean_sigs_per_sig_var = np.mean( var_sigs[var_sigs > 0] )
        
        print(f"\nFound {total_sigs} significant effects.", file=sys.stdout, flush=True)
        print(f"\tTotal vars with significant effects: {n_vars_sig}", file=sys.stdout, flush=True)
        print(f"\tMean significant effects per significant variant: {mean_sigs_per_sig_var}\n", file=sys.stdout, flush=True)
    
    # Converting results to dataframes and saving output.
    boot_counts_df = pd.DataFrame(boot_counts, columns=pred_effects_df.columns.values)
    boot_pvals_df = pd.DataFrame(boot_pvals, columns=pred_effects_df.columns.values)
    
    if verbosity >= 1:
        print(f"DONE!")
        
    return boot_pvals_df, boot_counts_df

@jit(nopython=True)
def fast_bootstrap(boot_counts, candidate_rare_indices, candidate_rare_abs_effects, bg_abs_effects, 
                   critical_value, min_std, random_number_generator,
                   ):
    """ Fast bootstrapping operation with numba.
    """
    n_bg_variants = bg_abs_effects.shape[0]
    for coli in range(bg_abs_effects.shape[1]):

        # Boot-strapping the background variant effect for this particular celltype-assay effect prediction.
        random_indices = random_number_generator.integers(0, n_bg_variants, size = n_bg_variants)
        boots = bg_abs_effects[random_indices, coli]
        boot_mean = np.mean( boots ) # Mean background effect
        stds = np.zeros((2), dtype=np.float64)
        stds[0] = np.std( boots )
        stds[1] = min_std
        boot_std = np.max( stds ) # Std effect, minimum set to avoid calling small effects & control for 0 bg edge-case

        # Z-score for candidate variants belonging to this distribution.
        candidate_rare_zs = (candidate_rare_abs_effects[:, coli] - boot_mean) / boot_std

        # Count the number of times each variant is considered non-signficant via z-test.
        for vari in range(len(candidate_rare_zs)):
            boot_counts[candidate_rare_indices[vari], coli] += (1 if candidate_rare_zs[vari] <= critical_value else 0)

def call_sig_effects(selected_gwas_stats, pred_effects, boot_pvals, p_cutoff=0.05, fc_cutoff=0, absolute_effects=False):
    """ Calls significant variant effects.
    """
    
    background_effects = pred_effects.values[selected_gwas_stats['var_label'].values=='background', :]
    if absolute_effects:
        background_effects = np.abs( background_effects )
    
    background_means = np.mean(background_effects, axis=0)
    
    fcs = np.apply_along_axis(np.subtract, 1, pred_effects.values, background_means)

    sig_bool = np.logical_and(boot_pvals.values < p_cutoff, np.abs(fcs) > fc_cutoff)
    
    sig_bool_df = pd.DataFrame(sig_bool, columns=pred_effects.columns.values)
    fcs_df = pd.DataFrame(fcs, columns=pred_effects.columns.values)
    
    return sig_bool_df, fcs_df
    
            
############### OLD CODE######################
# def call_significant_variant_effects_OLD(selected_gwas_stats, pred_effects, n_boots=10_000, p_cutoff=0.05, pseudocount=1,
#                                      min_std=0.01, verbosity=1):
#     """ Calls variants with significant effect ABOVE non-significant background variants. Works by boot-strapping the
#         background variants, determine mean and standard deviation across predicted effects, than performing a one-sample
#         z-test for each molecular effect and non-background variant, and recording if the non-background variant was significantly
#         different or not. P-values becomes the proportion of times the given predicted effect for a given non-background variant
#         was non-significantly different from the background variants.
    
#     Parameters
#     ----------
#     selected_gwas_stats: pd.DataFrame
#         GWAS summary statistics for each of the variants at a GWAS locus, with a column 'var_labels' indicating candidate, rare, and background variants. Each row is a variant.
#     pred_effects: pd.DataFrame
#         Predicted effects for each variant. Each row is a variant, and each column is a predicted molecular effect for that variant.
#     n_boots: int
#         No. of boot-straps of re-selecting the background variants.
#     p_cutoff: float
#         P-value below which a non-background variant predicted molecular effect is considered significantly different to the background vars.
#     pseudocount: int
#         Value added to prevent 0 p-values, defines lower-bound for minimum p-values, should be set to 1.
#     min_std: float
#         Minimum standard deviation for the background variant effects. Set to avoid 0 std for 0 effects of background variants causing infinite z-scores.
#     verbosity:
#         Verbosity levels. 0 errors only, 1 prints processing progress, 2 prints debugging information.
#     """
    
#     # Loading the necessary input files.
#     if verbosity >= 1:
#         print("Loading input files.", file=sys.stdout, flush=True)
        
#     var_info_df = selected_gwas_stats
#     pred_effects_df = pred_effects
    
#     # Need to separate into background versus candidate / rare variants. 
#     if verbosity >= 1:
#         print("Separating into background and candidate / rare variants.", file=sys.stdout, flush=True)
        
#     bg_bool = var_info_df['var_label'].values=='background'
#     bg_indices = np.where( bg_bool )[0]
#     candidate_rare_indices = np.where( ~bg_bool )[0]
    
#     bg_effects = pred_effects_df.values[bg_indices, :]
#     candidates_rare_effects = pred_effects_df.values[candidate_rare_indices, :]
    
#     bg_abs_effects = np.abs( bg_effects )
    
#     candidate_rare_abs_effects = np.abs( candidates_rare_effects )
    
#     # Run the p-value testing
#     boot_counts = np.zeros( pred_effects_df.shape )
    
#     total_ps = candidate_rare_abs_effects.shape[0] * candidate_rare_abs_effects.shape[1]
    
#     #### Decided I would separate the correction for the rare and common variants.
#     adj_p = p_cutoff / total_ps # Bonferroni nominal p-value.
#     critical_value = norm.ppf(1 - adj_p / 2) # Critical value for two-sided z-test.
#     if verbosity >= 2: # Print boot-strap details if in debug verbosity.
#         print(f"boot_counts_shape, total_ps, adj_p, critical_value: {boot_counts.shape}, {total_ps}, {adj_p}, {critical_value}", 
#               file=sys.stdout, flush=True)
        
#     start_ = time.time()
#     for booti in range(n_boots):

#         for coli in range(bg_effects.shape[1]):

#             # Boot-strapping the background variant effect for this particular celltype-assay effect prediction.
#             boots = np.random.choice(bg_abs_effects[:, coli], replace=True, size=bg_abs_effects.shape[0])
#             boot_mean = np.mean( boots ) # Mean background effect
#             boot_std = np.max([np.std( boots ), min_std]) # Std effect, minimum set to avoid calling small effects & control for 0 bg edge-case

#             # Z-score for candidate variants belonging to this distribution.
#             candidate_rare_zs = (candidate_rare_abs_effects[:, coli] - boot_mean) / boot_std

#             # Count the number of times each variant is considered non-signficant via z-test.
#             boot_counts[candidate_rare_indices, coli] += (candidate_rare_zs <= critical_value).astype(int)
            
#             if verbosity >= 2 and (booti % math.ceil(n_boots*0.1))==0 and coli==0: # Print col 0 boot strap details if debugging mode.
#                 col_sum_boot_counts = sum( boot_counts[candidate_rare_indices, coli] )
#                 print(f"boots_shape, boot_mean, boot_std, col_sum_boot_counts: {boots.shape}, {boot_mean}, {boot_std}, {col_sum_boot_counts}", 
#                       file=sys.stdout, flush=True)
            
#         if verbosity >= 1 and (booti % math.ceil(n_boots*0.1))==0: # Update progress in 10% increments.
#             perc_ = round(booti / n_boots, 1) * 100
#             elapsed_mins = round((time.time()-start_) / 60, 1)
#             print(f"Finished boot-strapping for {booti} / {n_boots} ({perc_}%) in {elapsed_mins}mins", file=sys.stdout, flush=True)

#     if verbosity >= 1:
#         print("Finished boot-strapping.", file=sys.stdout, flush=True)
        
#     # For the background variants, fill boot_counts simply with n_boots.
#     boot_counts[bg_indices, :] = n_boots
    
#     # Calculating the p-values based on the number of boot-straps
#     boot_pvals = (boot_counts + pseudocount) / (n_boots + pseudocount) # Pseudocount prevents incorrect 0 p-values.
    
#     if verbosity >= 1:
#         effects_sig = boot_pvals < p_cutoff
#         total_sigs = effects_sig.sum()
#         var_sigs = effects_sig.sum(axis=1)
#         n_vars_sig = sum(var_sigs > 0)
#         mean_sigs_per_sig_var = np.mean( var_sigs[var_sigs > 0] )
        
#         print(f"\nFound {total_sigs} significant effects.", file=sys.stdout, flush=True)
#         print(f"\tTotal vars with significant effects: {n_vars_sig}", file=sys.stdout, flush=True)
#         print(f"\tMean significant effects per significant variant: {mean_sigs_per_sig_var}\n", file=sys.stdout, flush=True)
    
#     # Converting results to dataframes and saving output.
#     boot_counts_df = pd.DataFrame(boot_counts, columns=pred_effects_df.columns.values)
#     boot_pvals_df = pd.DataFrame(boot_pvals, columns=pred_effects_df.columns.values)
    
#     if verbosity >= 1:
#         print(f"DONE!")
        
#     return boot_pvals_df, boot_counts_df
