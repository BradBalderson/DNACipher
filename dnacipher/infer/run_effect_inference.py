""" Runs variant effect inference.
"""

import sys
import pandas as pd

import dnacipher_train.command_line_helpers as clh

def infer_multivariant_effects(out_prefix, vcf_path, celltypes, assays, fasta_file_path,
                               sample_file, model_config_file, weights_path, model_name,
                               device, index_base,
                               correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col, batch_size,
                               batch_by, all_combinations, scoring_method, verbose, n_vars=None,
                               res_reduce_factor=None,
                               ):

    #### Output log file
    run_name = out_prefix.split('/')[-1]
    log_file = open(f'{out_prefix}_log_file.txt', 'w')

    print(f'DNACipher inference run: {run_name}\n', file=log_file, flush=True)

    ### Parsing the inputs..
    dnacipher, celltypes, assays = clh.parse_general_input(celltypes, assays, fasta_file_path, log_file,
                                                           # Necessary files for specifying the DNACipher model
                                                           weights_path=weights_path, sample_file_path=sample_file,
                                                           config_path=model_config_file, model_name=model_name,
                                                           device=device, verbose=verbose
                                                           )

    if type(effect_region_start_col) != type(None) and type(effect_region_end_col) != type(None):
        effect_region_cols = (effect_region_start_col, effect_region_end_col)
    else:
        effect_region_cols = None

    var_df = pd.read_csv(vcf_path, sep='\t', header=0)
    if type(n_vars) != type(None):
        var_df = var_df.head(n_vars)

    var_df.iloc[:,0] = var_df.iloc[:,0].astype(str)

    var_pred_effects = dnacipher.infer_multivariant_effects(var_df, celltypes, assays,
                                                            index_base=index_base, correct_ref=correct_ref,
                                                            seq_pos_col=seq_pos_col,
                                                            effect_region_cols=effect_region_cols,
                                                            batch_size=batch_size, batch_by=batch_by,
                                                            all_combinations=all_combinations,
                                                            scoring_method=scoring_method, verbose=verbose,
                                                            log_file=log_file, res_reduce_factor=res_reduce_factor,
                                                            )

    out_path = f"{out_prefix}_var_context_effects.txt.gz"
    var_pred_effects.to_csv(out_path, index=False, sep='\t', compression='gzip')

    if verbose:
        print(f"Wrote {out_path}", file=log_file, flush=True)



