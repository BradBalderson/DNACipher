import pickle

import time

import h5py

import os

import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def dealWithPlot(savePlot, showPlot, closePlot, folder, plotName, dpi,
				 tightLayout=True):
	""" Deals with the current matplotlib.pyplot.
	"""

	if tightLayout:
		plt.tight_layout()

	if savePlot:
		plt.savefig(folder+plotName, dpi=dpi,
					format=plotName.split('.')[-1])

	if showPlot:
		plt.show()

	if closePlot:
		plt.close()

def loadPickle(pickleName, loadType='fast'):

	if loadType=='slow':
		"""
		This is a defensive way to write pickle.load, allowing for very large 
		files on all platforms.
		"""
		max_bytes = 2 ** 31 - 1
		try:
			input_size = os.path.getsize(pickleName)
			bytes_in = bytearray(0)
			with open(pickleName, 'rb') as f_in:
				for _ in range(0, input_size, max_bytes):
					bytes_in += f_in.read(max_bytes)
			obj = pickle.loads(bytes_in)
		except:
			return None
		return obj

	else:
		with open(pickleName, 'rb') as input:
			return pickle.load(input)

def saveAsPickle(pickleName, singleCellAnalysisObjects,
                 max_bytes = 2 ** 31 - 1):

	## write
	bytes_out = pickle.dumps(singleCellAnalysisObjects,
							 protocol=pickle.HIGHEST_PROTOCOL)
	with open(pickleName, 'wb') as f_out:
		for idx in range(0, len(bytes_out), max_bytes):
			f_out.write(bytes_out[idx:idx + max_bytes])

def split_h5(run_name, genome_file, h5_file, n_splits, out_dir):
    """ Converts the inputted .h5 file to a split_h5, and outputs the new index file.
    """

    log_file = open(f'{out_dir}{run_name}_log_file.txt', 'w')
    script_start_time = time.time()

    genome_rois = pd.read_csv(f'{ genome_file }', sep='\t', header=None)

    print(f"Starting to generate split file with {n_splits} splits.", file=log_file, flush=True)
    n_seqs = genome_rois.shape[0]
    indices = list(range(n_seqs))
    interval = n_seqs // n_splits

    precomp = h5py.File(h5_file, 'r')
    new_file_name = out_dir + h5_file.split('/')[-1].replace('.h5', f'.{n_splits}N.split.h5')
    new_file = h5py.File(new_file_name, 'w')
    new_file_genome_rois = genome_rois.copy()

    precomp_key = list( precomp.keys() )[0]

    new_indexes = []  # Stores which dataset...
    index_in_dataset = []
    prev_n = 0
    for n_ in range(n_splits):
        if n_ < n_splits-1:
            current_n = prev_n + interval
        else:
            current_n = n_seqs # Allocate extra sequences to last split.

        seqs = precomp[ precomp_key ][indices[prev_n:current_n], :].astype(np.float32)
        prev_n = current_n

        new_dataset = new_file.create_dataset(str(n_), shape=seqs.shape,
                                              dtype='f', compression='lzf',  # compression_opts=9
                                              )
        new_dataset[:, :] = seqs

        new_indexes.extend([n_] * seqs.shape[0])
        index_in_dataset.extend(list(range(seqs.shape[0])))
        del seqs

        print(f"Split {n_} / {n_splits} written.", file=log_file, flush=True)

    new_file_genome_rois['dataset_index'] = new_indexes
    new_file_genome_rois['index_in_dataset'] = index_in_dataset

    precomp.close()
    new_file.close()

    print(f"FINISHED {n_splits} written.\n", file=log_file, flush=True)

    out_file_name = out_dir + genome_file.split('/')[-1].replace('.txt', f'.{n_splits}N.split.txt')
    new_file_genome_rois.to_csv(out_file_name, sep='\t', header=None, index=None)

    print(f"Saved {out_file_name}\n", file=log_file, flush=True)
    print(f"Saved {new_file_name}", file=log_file, flush=True)
    print("FINISHED file writing!\n", file=log_file, flush=True)

    print(f'DONE', file=log_file, flush=True)

    script_end_time = time.time()

    total_minutes = round((script_end_time - script_start_time) / 60, 3)
    total_hours = round((script_end_time - script_start_time) / 60 / 60, 3)

    print("DONE.", file=log_file, flush=True)
    print("TOTAL minutes: ", total_minutes,
          file=log_file, flush=True)
    print("TOTAL hours: ", total_hours,
          file=log_file, flush=True)

    log_file.close()

def batch_collate_fn(outputs):  # Need to get this to match the expected DataLoader output for data_managae.write_dataloader_h5
    new_out = [outputs[0][0], torch.tensor(outputs[0][1])]
    return new_out

def write_dataloader_to_h5(loader, out_h5, log_file, n_columns, dataset_key):
    """ Writes a pytorch dataloader to h5 in the defined dataloader batches. Only does so for the defined outputs.
    """

    ##### Writing TRAIN embeddings...
    epi_values = h5py.File(f'{out_h5}', 'w')

    print(f"\nStarting to write epivalues to {out_h5}\n", file=log_file, flush=True )

    # Create a dataset with the final size
    n_rows = len( loader.dataset )
    dset = epi_values.create_dataset(dataset_key, shape=(n_rows, n_columns), dtype='f', compression='lzf',
                    )

    nseqs = 0
    start_index = 0
    start = time.time()
    for i, out in enumerate( loader ):

        ##### Now writing each of the embeddings...
        inputs, output = out
        epi_values_ = output.to("cpu").numpy()

        batch_size_ = epi_values_.shape[0]
        end_index = int(start_index + batch_size_)
        dset[start_index:end_index, :] = epi_values_  # This only loads the slice we're writing to

        start_index = end_index
        nseqs += batch_size_

        if i == 0:
            end = time.time()
            print(round(end-start, 3), f"seconds to load one batch of size {batch_size_} FIRST time.",
                                                                                             file=log_file, flush=True )

        elif i == 1:
            end2 = time.time()
            print(round(end2-end, 3), f"seconds to load one batch of size {batch_size_} SECOND time.",
                                                                                             file=log_file, flush=True )

        if i % 2 == 0: # Just keep an update..
            end2 = time.time()
            print(f"WRITTEN {nseqs} out of {n_rows} in {round((end2-start)/60, 3)} minutes.", file=log_file, flush=True )

    epi_values.close()
    print(f"FINISHED writing dataloader values to h5 output.\n\n", file=log_file, flush=True )