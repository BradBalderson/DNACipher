""" Runs variant effect inference.
"""

import sys
import pandas as pd

import dnacipher.command_line_helpers as clh

# TODO finish working on this, need to get it working for the tutorial.
def infer_effects(out_prefix, chr_, pos, ref, alt, celltypes, assays, fasta_file_path,
                  sample_file, model_config_file, weights_path, model_name,
                  device, index_base, correct_ref,
                  seq_pos, effect_region_start, effect_region_end, batch_size, batch_by, all_combinations, return_all,
                  scoring_method, verbose
    ):

    dnacipher, celltypes, assays = clh.parse_general_input(celltypes, assays, device, fasta_file_path, verbose,
                                                           weights_path=weights_path,
                                                           sample_file_path=sample_file,
                                                           config_path=model_config_file, model_name=model_name,
                                                           )

    if type(effect_region_start) != type(None) and type(effect_region_end) != type(None):
        effect_region = (effect_region_start, effect_region_end)
    else:
        effect_region = None

    if return_all:  # Detailed predictions along the strand !
        context_effects, ref_signals, alt_signals, ref_seq, alt_seq, ref_features, alt_features = \
            dnacipher.infer_effects(chr_, pos, ref, alt, celltypes, assays,
                                    index_base=index_base, correct_ref=correct_ref, seq_pos=seq_pos,
                                    effect_region=effect_region, batch_size=batch_size, batch_by=batch_by,
                                    all_combinations=all_combinations, scoring_method=scoring_method, verbose=verbose,
                                    return_all=True,
                                    )

        ref_signals_normed_and_ordered, alt_signals_normed_and_ordered = dnacipher.normalise_and_ordered_signals(
            ref_signals, alt_signals
        )
        diff_signals = dnacipher.get_diff_signals(ref_signals_normed_and_ordered, alt_signals_normed_and_ordered)

        frames = [context_effects, ref_signals, alt_signals, ref_signals_normed_and_ordered,
                  alt_signals_normed_and_ordered,
                  diff_signals]

        for i, (dataframe, name_) in enumerate(zip(frames, ['context_effects', 'ref_signals', 'alt_signals',
                                                            'ref_signals_normed_and_ordered',
                                                            'alt_signals_normed_and_ordered',
                                                            'diff_signals'])):

            out_path = f"{out_prefix}{name_}.txt"
            dataframe.to_csv(out_path, index=i != 0, sep='\t')  # all of the signal frames need the index

            if verbose:
                print(f"Wrote {out_path}", file=sys.stdout, flush=True)

    else:  # Just the overall locus effects.
        context_effects = dnacipher.infer_effects(chr_, pos, ref, alt, celltypes, assays,
                                                  index_base=index_base, correct_ref=correct_ref, seq_pos=seq_pos,
                                                  effect_region=effect_region, batch_size=batch_size, batch_by=batch_by,
                                                  all_combinations=all_combinations, scoring_method=scoring_method,
                                                  verbose=verbose,
                                                  return_all=False,
                                                  )

        out_path = f"{out_prefix}context_effects.txt"
        context_effects.to_csv(out_path, index=False, sep='\t')

        if verbose:
            print(f"Wrote {out_path}", file=sys.stdout, flush=True)

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



