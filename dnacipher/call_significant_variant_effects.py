from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated

import typer
import sys

import math

import time

import numpy as np
import pandas as pd
from scipy.stats import norm

app = typer.Typer(pretty_exceptions_short=False)

@app.command()
def call_significant_variant_effects(
    var_info_file: Annotated[str, typer.Argument(help="Variant information files, containing a variant per row with columns including: "+\
                                                 "'chromosome', 'position', 'ref', 'alt', 'var_label', where var_label label variants as: "+\
                                                 "'background', 'candidate', or 'rare'. The later two categories predicted effects will be "+\
                                                "compared with the background variants using a boot-strapping approach.")],
    pred_effects_file: Annotated[str, typer.Argument(help="Variant predicted effects, with the same number of variants as indicated in "+\
                                                    "the var_info_file and in the same row order. The columns indicate the predicted "+\
                                                     "variant effects as determined by DNACipher or other methods.")],
    output_pvals_file: Annotated[
        Optional[str],
        typer.Option(
            "-o",
            "--out_file",
            help=("Write the output file of p-values per variant and molecular effect to this file path. Is written as a tab-separated text "+\
                "files of same shape as the inputted pred_effects_file, but with adjusted p-values."
                ),
            )] = "pred_effects.pvals.txt",
    output_counts_file: Annotated[
        Optional[str],
        typer.Option(
            "-c",
            "--count_file",
            help=("Write the output file of non-significant effect calls across n_boots per variant and molecular effect to this file path. "+\
                  "Is written as a tab-separated text "+\
                "files of same shape as the inputted pred_effects_file, but with counts."
                ),
            )] = "pred_effects.counts.txt",
    n_boots: Annotated[
        Optional[int],
        typer.Option(
            "-b",
            "--boot_straps",
            help=("The number of boot-straps to perform for the stastical test."
                  )
            )
        ] = 10_000,
    p_cutoff: Annotated[
        Optional[float],
        typer.Option(
            "-p",
            "--p_cutoff",
            help=("Adjusted p-value ABOVE which variants are considered non-significantly different from the background variants "+\
                  "the adjustment is based on the Bonferroni correction considering the number of variants and molecular effects "+\
                  "being tested. For simplicity, the rows containing the background variants will be outputted with 1's indicating "+\
                  "non-significance."
                  )
            )
        ] = 0.05,
    pseudocount: Annotated[
        Optional[float],
        typer.Option(
            "-pc",
            "--pseudocount",
            help=("Pseudocount to be added to the number of times the variant molecular effect was non-signficant compared to boot-straped "+\
                  "background variants."
                  )
            )
        ] = 1,
    min_std: Annotated[
        Optional[float],
        typer.Option(
            "-std",
            "--min_std",
            help=("Imposed minimum standard deviation of the background variant predicted effects, controls for very small effects being "+\
                  "significant and also case where all 0s for background causing everything to be called significant."
                  )
            )
        ] = 0.01,
    verbosity: Annotated[
            Optional[int],
            typer.Option(
                "--v",
                "-v",
                "--verbosity",
                "-verbosity",
                help=("Verbosity levels. 0 errors only, 1 prints processing progress, 2 prints debugging information."
                      )
                )
            ] = 1
):
    
    # Loading the necessary input files.
    if verbosity >= 1:
        print("Loading input files.", file=sys.stdout, flush=True)
        
    var_info_df = pd.read_csv(var_info_file, sep='\t', index_col=0)
    pred_effects_df = pd.read_csv(pred_effects_file, sep='\t', index_col=0)
    
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
    
    start_ = time.time()
    for booti in range(n_boots):

        for coli in range(bg_effects.shape[1]):

            # Boot-strapping the background variant effect for this particular celltype-assay effect prediction.
            boots = np.random.choice(bg_abs_effects[:, coli], replace=True, size=bg_abs_effects.shape[0])
            boot_mean = np.mean( boots ) # Mean background effect
            boot_std = np.max([np.std( boots ), min_std]) # Std effect, minimum set to avoid calling small effects & control for 0 bg edge-case

            # Z-score for candidate variants belonging to this distribution.
            candidate_rare_zs = (candidate_rare_abs_effects[:, coli] - boot_mean) / boot_std

            # Count the number of times each variant is considered non-signficant via z-test.
            boot_counts[candidate_rare_indices, coli] += (candidate_rare_zs <= critical_value).astype(int)
            
            if verbosity >= 2 and (booti % math.ceil(n_boots*0.1))==0 and coli == 0: # Print col 0 boot strap details if debugging mode.
                col_sum_boot_counts = sum( boot_counts[candidate_rare_indices, coli] )
                print(f"boots_shape, boot_mean, boot_std, col_sum_boot_counts: {boots.shape}, {boot_mean}, {boot_std}, {col_sum_boot_counts}", 
                      file=sys.stdout, flush=True)
            
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
    
    boot_counts_df.to_csv(output_counts_file, sep='\t')
    boot_pvals_df.to_csv(output_pvals_file, sep='\t')
    
    if verbosity >= 1:
        print(f"Saved boot-strapped counts to: {output_counts_file}", file=sys.stdout, flush=True)
        print(f"Saved adjusted p-values to: {output_pvals_file}", file=sys.stdout, flush=True)

def main():
    root_dir = Path(__file__).parent
    sys.path.append(str(root_dir))
    app()

if __name__ == "__main__":
    main()



