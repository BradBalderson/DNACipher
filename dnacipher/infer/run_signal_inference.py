""" Runs signal inference for imputation of signals, given inputted celltypes/assays.
"""

import pandas as pd

import dnacipher_train.command_line_helpers as clh

def infer_signals(out_prefix, bed_path, celltypes, assays, fasta_file_path,
                   sample_file, model_config_file, weights_path, model_name, device, batch_size,
                   all_combinations, verbose, n_regions=None,
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

    bed_df = pd.read_csv(bed_path, sep='\t', header=None)
    if type(n_regions) != type(None):
        bed_df = bed_df.head(n_regions)

    bed_df.iloc[:,0] = bed_df.iloc[:,0].astype(str)

    region_signals = dnacipher.infer_signals(bed_df, celltypes, assays, batch_size=batch_size,
                                             all_combinations=all_combinations, verbose=verbose, log_file=log_file
                                              )

    out_path = f"{out_prefix}_signals.txt.gz"
    region_signals.to_csv(out_path, index=True, sep='\t', compression='gzip')

    if verbose:
        print(f"Wrote {out_path}", file=log_file, flush=True)









