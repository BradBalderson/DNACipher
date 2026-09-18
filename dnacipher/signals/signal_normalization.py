""" Script for normalizing signals across assays.
"""

import h5py

import numpy as np

from . import normalise
from . import signal_dataset_classes as signal_dataset_classes
from ..helpers import data_manage

import time

def fit_signal_normalization(run_name, genome_file, sample_file, signal_file, out_dir):
    """ Fits the normalization to the data.
    """

    #### Output log file
    log_file = open(f'{out_dir}{run_name}_fit_log_file.txt', 'w')

    log_file.write( f'Starting normalization {run_name}\n' )

    script_start_time = time.time()

    # Loading only the training regions and experiments.
    signal_dataset = signal_dataset_classes.load_signal_data(genome_file, sample_file, signal_file,
                                                      train_regions_only=True, train_expers_only=True)

    print("\nCreating/Getting the normalisation objects.\n", file=log_file, flush=True)

    rna_assays = np.array([assay for assay in signal_dataset.assays if 'RNA-seq' in assay])

    # This will contain the scaling factors to get the means equivalent across assays,
    rna_norm = normalise.NonzeroShiftedLog2Norm( rna_assays )
    rna_norm.set_input_assay_indices( signal_dataset.load_celltype_assays )

    norm_obj = normalise.Normalise( signal_dataset.assays )
    norm_obj.set_input_assay_indices( signal_dataset.load_celltype_assays )

    norm_stats = {'RNA_nonZeroShiftedLog2': rna_norm,
                  'Normalise': norm_obj
                  }

    process_start = time.time()

    # Fitting the RNA-seq normalisation object first.
    normalise.iterate_seqs('rna-fit', signal_dataset, None, norm_stats, log_file)

    # Now fit the Normalisation object, which will apply the RNA-seq normalisation object first to each batch
    # for each fit.
    normalise.iterate_seqs('norm-fit', signal_dataset, None, norm_stats, log_file)

    process_end = time.time()
    mins = round((process_end - process_start) / 60, 3)
    print(f"Processed all {len(signal_dataset)} seqs in {mins}minutes.\n\n", file=log_file, flush=True)

    pickle_path = out_dir + f'{run_name}_norm_stats.pkl'
    data_manage.saveAsPickle(pickle_path, norm_stats)
    print(f"Saved {pickle_path}", file=log_file, flush=True)

    script_end_time = time.time()

    total_minutes = round((script_end_time - script_start_time) / 60, 3)
    total_hours = round((script_end_time - script_start_time) / 60 / 60, 3)

    print("DONE.", file=log_file, flush=True)
    print("TOTAL minutes: ", total_minutes, file=log_file, flush=True)
    print("TOTAL hours: ", total_hours, file=log_file, flush=True)

    log_file.close()

def apply_signal_normalization(run_name, genome_file, sample_file, signal_file, norm_file, out_dir):
    """ Applies a pre-determined normalization to the data.
    """

    #### Output log file
    log_file = open(f'{out_dir}{run_name}_apply_log_file.txt', 'w')

    log_file.write( f'Starting normalization {run_name}\n' )

    script_start_time = time.time()

    # Loads the whole dataset, regardless of train/test, because applying a normalization learn on the train data.
    signal_dataset = signal_dataset_classes.load_signal_data(genome_file, sample_file, signal_file)

    print("\nCreating/Getting the normalisation objects.\n", file=log_file, flush=True)

    # This will contain the scaling factors to get the means equivalent across assays,
    print(f"Loading the normalization stats from {norm_file}", file=log_file, flush=True)

    norm_stats = data_manage.loadPickle(norm_file)
    if 'RNA_nonZeroShiftedLog2' not in norm_stats or 'Normalise' not in norm_stats:
        raise Exception(f"Loaded pickle file misformatted, does not contain one of keys: RNA_nonZeroShiftedLog2, "
                        f"Normalise, but has keys: {norm_stats.keys()}")

    ##### Need to reset the normalisation objects expected inputs of celltype_assays,
    ##### because the data will have changed !!!
    norm_stats['RNA_nonZeroShiftedLog2'].set_input_assay_indices( signal_dataset.load_celltype_assays )
    norm_stats['Normalise'].set_input_assay_indices( signal_dataset.load_celltype_assays )
    #### This sets the scaling factors for the specific celltype/assays, to the relevant assay scale factor.
    norm_stats['Normalise'].calc_celltype_assay_scalefactors()

    # Determining the name of the output file.
    suffix = '.'.join( signal_file.split('/')[-1].split('.')[1:] ) # Example '_embed.h5'
    precomp_prefix = signal_file.split('/')[-1].replace(suffix, '')
    transform_path = out_dir + f'{precomp_prefix}{run_name}.normalised.{suffix}'
    transform_file = h5py.File(transform_path, 'w')

    process_start = time.time()

    # Applying the normalisation, within each batch need to
    normalise.iterate_seqs('transform', signal_dataset, transform_file, norm_stats, log_file)

    transform_file.close()
    print(f"\nSaved {transform_path}\n", file=log_file, flush=True)

    process_end = time.time()
    mins = round((process_end - process_start) / 60, 3)
    print(f"Processed all {len(signal_dataset)} seqs in {mins}minutes.\n\n", file=log_file, flush=True)

    print(f"Visualising the normalisation effect on the data..", file=log_file, flush=True)

    ##### The original data...
    signal_dataset.plot_assay_nonzeros(nsplits=3)
    data_manage.dealWithPlot(True, False, True, out_dir,
                     f'{run_name}_{precomp_prefix}_epimeasure_distribs.png', 300)
    print(f"Saved {run_name}_{precomp_prefix}_epimeasure_distribs.png", file=log_file, flush=True)

    ##### The normalized data:
    signal_norm_dataset = signal_dataset_classes.load_signal_data(genome_file, sample_file, transform_path)
    signal_norm_dataset.plot_assay_nonzeros(nsplits=3)

    data_manage.dealWithPlot(True, False, True, out_dir,
                     f'{run_name}_{precomp_prefix}_NORMED_epimeasure_distribs.png', 300)
    print(f"Saved {run_name}_{precomp_prefix}_NORMED_epimeasure_distribs.png", file=log_file, flush=True)

    script_end_time = time.time()

    total_minutes = round((script_end_time - script_start_time) / 60, 3)
    total_hours = round((script_end_time - script_start_time) / 60 / 60, 3)

    print("DONE.", file=log_file, flush=True)
    print("TOTAL minutes: ", total_minutes, file=log_file, flush=True)
    print("TOTAL hours: ", total_hours, file=log_file, flush=True)

    log_file.close()




