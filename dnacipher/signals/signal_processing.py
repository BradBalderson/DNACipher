"""
Functions related to processing signals to be predicted.
"""

import time

import os

import h5py

import math
import subprocess
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

from .signal_dataset_classes import BigWigSignals

from ..helpers import data_manage

from .bw_tool_helpers import BwtoolBinaryMatrixFile


def dataloader_signal_compute(TEST_NAME, genome_rois, celltypes, assays, data, region_pad,
                              batch_size, cpu, out_dir, log_file):
    """ Uses Pytorch dataloader implementation to generate the signal file.
    """

    dataset = BigWigSignals(genome_rois.values, celltypes, assays, data, region_pad=region_pad)

    batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)

    loader = DataLoader(dataset, num_workers=cpu, persistent_workers=True, # so don't need to re-open files.
                        sampler=batch_sampler, collate_fn=data_manage.batch_collate_fn)

    #### Writing the epigenetic values to h5 output:
    data_manage.write_dataloader_to_h5(loader, f'{out_dir}{TEST_NAME}_epivalues.h5',
                                       log_file, len(data), 'epi_values')

def deeptools_signal_compute(test_name, genome_rois, data, region_pad, batch_size, cpu, out_dir, log_file):
    """ This implementation is basically a light wrapper around deep tools. Requires deeptools install:

        https://deeptools.readthedocs.io/en/stable/content/installation.html

        mamba install -c conda-forge -c bioconda deeptools
    """
    ### Creating temp folder to write intermediate files to.
    temp_dir = f"{out_dir}_temp/"
    os.system(f"mkdir {temp_dir}")
    print("Making temp folder to store temporary files:", temp_dir, file=log_file, flush=True)

    #### Now creating the bed file of the genome_rois!
    chroms, starts = genome_rois.values[:, 0].copy(), genome_rois.values[:, 1].copy()
    ends = starts+region_pad
    starts = starts-region_pad

    bed_file = f"{temp_dir}query_regions.bed"
    bed_df = pd.DataFrame([chroms, starts, ends]).transpose()
    bed_df.to_csv(bed_file, sep='\t', header=None, index=None)

    # In this context, let batch_size refer to the number of BigWig files to compute at once.
    all_bigwig_files = list(data.values())
    batchi = 0

    out_h5 = f'{out_dir}{test_name}_epivalues.h5'
    epi_values = h5py.File(out_h5, 'w')

    print(f"\nStarting to write epivalues to {out_h5}\n", file=log_file, flush=True)

    # Create a dataset with the final size
    dset = epi_values.create_dataset('epi_values', shape=(genome_rois.shape[0], len(data)), dtype='f', compression='lzf',
                    )
    start = time.time()

    start_index = 0
    end_index = 0
    while end_index < len(data):

        end_index = int(start_index + batch_size)
        bigwig_files = all_bigwig_files[start_index:end_index]
        out_file = f"{temp_dir}signals.{batchi}.npz"
        run_str = f"multiBigwigSummary BED-file -b {' '.join(bigwig_files)} -o {out_file} --BED {bed_file} -p {cpu}"
        os.system( run_str )

        ### Loading the batch values:
        epi_values_ = np.load( out_file )["matrix"]
        epi_values_[np.isnan(epi_values_)] = 0 # Filling in the missing values, which are 0's

        batch_size_ = epi_values_.shape[1]
        end_index = int(start_index + batch_size_)

        dset[:, start_index:end_index] = epi_values_  # This only loads the slice we're writing to

        start_index = end_index

        if batchi == 0:
            end = time.time()
            print(round(end - start, 3), f"seconds to load one batch of size {batch_size_} FIRST time.",
                                                                                              file=log_file, flush=True)

        elif batchi == 1:
            end2 = time.time()
            print(round(end2 - end, 3), f"seconds to load one batch of size {batch_size_} SECOND time.",
                                                                                              file=log_file, flush=True)

        if batchi % 2 == 0:  # Just keep an update..
            print(f"WRITTEN {end_index} out of {len(data)}.", file=log_file, flush=True)

    epi_values.close()
    print(f"FINISHED writing dataloader values to h5 output.\n\n", file=log_file, flush=True )

    #### Now tidying the temporary files:
    os.system(f"rm -r {temp_dir}")
    print("Removed temp directory:", temp_dir, file=log_file, flush=True)

def bwtool_run(temp_dir, region_pad, batchi, bigwig_files):
    """ Wrapper function for running bwtool.
    """

    width = int(region_pad*2)
    bigwigs_joined = ','.join( bigwig_files )

    out_file = f"{temp_dir}{batchi}.bin"

    run_str = f"bwtool -fill=0 matrix -binary-matrix -starts -tiled-averages={width} {width}:{width} {temp_dir}query_regions.bed {bigwigs_joined} {out_file}"

    subprocess.run( run_str.split(' ') )

    # Now should be able to read resulting values now!
    matrix = BwtoolBinaryMatrixFile( out_file ).bwtool_binary_matrix[:, 1::2]

    return matrix

def bwtool_signal_compute(test_name, genome_rois, data, region_pad, batch_size, cpu, out_dir, log_file):
    """"""
    """ This implementation is basically a light wrapper around bwtool. Requires deeptools install:

        https://github.com/CRG-Barcelona/bwtool/issues/49

        Or for a more friendly conda environment friendly approach:

        https://github.com/LieberInstitute/recount.bwtool
    """
    ### Creating temp folder to write intermediate files to.
    temp_dir = f"{out_dir}_temp/"
    subprocess.run(["mkdir", temp_dir])
    print("Making temp folder to store temporary files:", temp_dir, file=log_file, flush=True)

    #### Now creating the bed file of the genome_rois!
    chroms, starts = genome_rois.values[:, 0].copy(), genome_rois.values[:, 1].copy()

    bed_file = f"{temp_dir}query_regions.bed"
    bed_df = pd.DataFrame([chroms, starts, starts]).transpose()
    bed_df.to_csv(bed_file, sep='\t', header=None, index=None)

    # In this context, let batch_size refer to the number of BigWig files to compute at once.
    all_bigwig_files = list(data.values())

    # Python to multi-process across batches of bigwigs!
    # Setting up the .h5 that we will write the results to!
    out_h5 = f'{out_dir}{test_name}_epivalues.h5'
    epi_values = h5py.File(out_h5, 'w')

    # Create a dataset with the final size
    dset = epi_values.create_dataset('epi_values', shape=(genome_rois.shape[0], len(data)), dtype='f',
                                     compression='lzf',
                                     )

    if cpu > 1:

        #### Processing in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from functools import partial
        partial_func = partial(bwtool_run, temp_dir, region_pad)

        # Determining the number of batches
        n_batches = math.ceil( len(all_bigwig_files) / batch_size )

        with ProcessPoolExecutor(max_workers=cpu) as executor:
            futures = {}
            start_index = 0

            for batchi in range(n_batches):

                end_index = int(start_index + batch_size)
                bigwig_files = all_bigwig_files[start_index:end_index]
                end_index = int(start_index + len( bigwig_files ))

                futures[executor.submit(partial_func, batchi, bigwig_files)] = (batchi, start_index, end_index, bigwig_files)

                start_index = end_index

            #finished = {}
            start = time.time()
            for future in as_completed(futures):
                batchi, start_index, end_index, bigwig_files = futures[future]

                try:
                    epi_values_ = future.result()  # will re-raise exceptions from worker

                    dset[:, start_index:end_index] = epi_values_  # This only loads the slice we're writing to

                    if batchi == 0:
                        end = time.time()
                        print(round(end - start, 3),
                              f"seconds to load one batch of size {len(bigwig_files)} FIRST time.",
                              file=log_file, flush=True)

                    elif batchi == 1:
                        end2 = time.time()
                        print(round(end2 - end, 3),
                              f"seconds to load one batch of size {len(bigwig_files)} SECOND time.",
                              file=log_file, flush=True)

                    if batchi % 2 == 0:  # Just keep an update..
                        print(f"WRITTEN {end_index} out of {len(data)}.", file=log_file, flush=True)

                    #finished[batchi] = [start_index, end_index, epi_values_]

                    print(f"Finished computing batch {batchi}\n", file=log_file, flush=True)

                except Exception as e:

                    raise Exception(f"Error: {e}, when processing bigwigs:\n {bigwig_files}")

    if cpu == 1:
        epivalues_ = bwtool_run(temp_dir, region_pad, log_file, 0, all_bigwig_files)

        dset[:, :] = epivalues_

    epi_values.close()
    print(f"FINISHED writing dataloader values to h5 output.\n\n", file=log_file, flush=True )

    #### Now tidying the temporary files:
    subprocess.run(["rm", "-r", temp_dir])
    print("Removed temp directory:", temp_dir, file=log_file, flush=True)

def run_signal_precomputation(test_name, genome_file, sample_folder, sample_file, batch_size, cpu, region_pad, out_dir,
                              method="dataloader"):

    TEST_NAME = test_name

    #### Output log file
    log_file = open(f'{out_dir}{test_name}_log_file.txt', 'w')
    log_file.write( f'Starting precomputation {TEST_NAME} with run parameters:\n' )
    log_file.write( "\n".join([test_name, genome_file, sample_file, f"batch_size:{batch_size}",
                               f"cpu:{cpu}", f"region_pad:{region_pad}", f"method:{method}"]) )

    script_start_time = time.time()

    ####################################################################################################
                                    # Loading the data #
    ####################################################################################################
    #### Loading in ROIs.
    genome_rois = pd.read_csv(genome_file, sep='\t', header=None
                              )

    sample_df = pd.read_csv(sample_file, sep='\t', index_col=0)
    file_type = '.bigWig'
    files_ = list(np.char.add(sample_df['File accession'].values.astype(str), file_type))  # Adding to file name
    files_ = [f"{sample_folder}{bigwig_file}" for bigwig_file in files_]
    celltype_assays = [tuple(celltype_assay.split('---')) for celltype_assay in
                       sample_df['celltype_assay'].values.astype(str)]
    celltypes = set([celltype_assay.split('---')[0] for celltype_assay in sample_df['celltype_assay'].values.astype(str)])
    assays = set([celltype_assay.split('---')[1] for celltype_assay in sample_df['celltype_assay'].values.astype(str)])

    print(f"Will load signals from {len(files_)} files.", file=log_file, flush=True)
    print(f"Representing {len(celltypes)} cell types and {len(assays)} assays", file=log_file, flush=True)

    data = {celltype_assays[i]: f'{files_[i]}' for i in range(len(files_))}

    celltypes = list(np.unique([celltype_assay[0] for celltype_assay in celltype_assays]))
    assays = list(np.unique([celltype_assay[1] for celltype_assay in celltype_assays]))

    ####################################################################################################
                                    # Creating the data loader! #
    ####################################################################################################
    print(f"\nWill precompute epigenetic values for {genome_rois.shape[0]} regions across {len(data)} experiments",
          file=log_file, flush=True)

    methods = ["dataloader", "deeptools", "bwtool"]
    if method == "dataloader":
        dataloader_signal_compute(TEST_NAME, genome_rois, celltypes, assays, data, region_pad, batch_size, cpu,
                                                                                                      out_dir, log_file)

    elif method == "deeptools":
        deeptools_signal_compute(test_name, genome_rois, data, region_pad, batch_size, cpu, out_dir, log_file)

    elif method == "bwtool":
        bwtool_signal_compute(test_name, genome_rois, data, region_pad, batch_size, cpu, out_dir, log_file)

    else:
        raise Exception(f"Got signal precompute method={method} but only support options: {methods}")

    ###################################################################################################
                    # Finishing UP #
    ###################################################################################################
    print("DONE.", file=log_file, flush=True )

    script_end_time = time.time()

    total_minutes = round((script_end_time-script_start_time)/60, 3)
    total_hours = round((script_end_time-script_start_time)/60/60, 3)

    print("TOTAL minutes: ", total_minutes,
          file=log_file, flush=True )
    print("TOTAL hours: ", total_hours,
          file=log_file, flush=True )

    log_file.close()






