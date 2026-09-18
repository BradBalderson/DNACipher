""" Functions related to determining the weighting of importance for each experiment, based on the rarity of the
    celltype / assay !!
"""

import numpy as np
import pandas as pd

def optimize_weightings(sample_file):
    """ Determines an optimal way to weight the experiments, to up-weight experiments that represent more rarely
         observed celltypes/assays.
    """

    encode_meta_imputable = pd.read_csv(sample_file, sep='\t', index_col=0)

    celltype_assay_labels = encode_meta_imputable['celltype_assay'].values
    celltype_assays = np.unique(celltype_assay_labels)
    celltypes = np.unique(encode_meta_imputable['celltype_assay'].apply(lambda x: x.split('---')[0]))
    assays = np.unique(encode_meta_imputable['celltype_assay'].apply(lambda x: x.split('---')[1]))

    # Creating a matrix that counts the number of experiments for each cell type and assay:
    celltype_assay_counts = np.zeros((len(celltypes), len(assays)))
    for i, celltype in enumerate(celltypes):
        for j, assay in enumerate(assays):
            celltype_assay_counts[i, j] = len(np.where(celltype_assay_labels == f"{celltype}---{assay}")[0])

    celltype_assay_counts = pd.DataFrame(celltype_assay_counts, index=celltypes, columns=assays)

    # Using this to determine the frequency of each cell type and assay in terms of represented experiments.
    total = celltype_assay_counts.values.sum()
    celltype_freqs = celltype_assay_counts.values.sum(axis=1) / total
    assay_freqs = celltype_assay_counts.values.sum(axis=0) / total

    # Getting an inverse weighting, to try and up-weight rarer cell type / assays
    inverse_celltype_freqs = 1 / celltype_freqs
    inverse_celltype_freqs = inverse_celltype_freqs / sum(inverse_celltype_freqs)

    inverse_assay_freqs = 1 / assay_freqs
    inverse_assay_freqs = inverse_assay_freqs / sum(inverse_assay_freqs)

    # Getting the best assay bias, which minimizes overall bias:
    best_assay_bias, celltype_to_assayindices, assay_to_celltypeindices = get_best_bias(celltypes, assays,
                                                                                        celltype_assays,
                                                                                        inverse_celltype_freqs,
                                                                                        inverse_assay_freqs
                                                                                        )

    # Getting the sampling counts for this:
    sampling_counts = get_sampling_counts(1_000_000, best_assay_bias, celltypes, assays,
                        inverse_celltype_freqs, inverse_assay_freqs,
                        celltype_to_assayindices, assay_to_celltypeindices
                        )
    sampling_counts += 1 # pseudocount
    sample_probs_balanced = sampling_counts / sampling_counts.sum()

    # Now assigning the weights to each of these celltype/assays.
    celltype_assay_samp_probs = np.zeros((len(celltype_assay_labels)), dtype=np.float32)
    for i, celltype_assay in enumerate(celltype_assay_labels):
        celltype, assay = celltype_assay.split('---')
        celltype_index = list( celltypes ).index( celltype )
        assay_index = list( assays ).index( assay )

        #### I won't use the probability here, since results in very small floating-point numbers...
        celltype_assay_samp_probs[i] = sample_probs_balanced[celltype_index, assay_index]

    return celltype_assay_samp_probs

def get_indice_maps(celltypes, assays, celltype_assays):
    """Generates a mapping from available assays for each cell type, and vice versa.
    """

    ### For each assay, determining what cell types are represented...
    from collections import defaultdict

    celltype_to_assayindices = defaultdict(list)
    assay_to_celltypeindices = defaultdict(list)
    for i, assay in enumerate(assays):
        for j, celltype in enumerate(celltypes):

            if f'{celltype}---{assay}' in celltype_assays:
                celltype_to_assayindices[celltype].append(i)
                assay_to_celltypeindices[assay].append(j)

    return celltype_to_assayindices, assay_to_celltypeindices

def get_best_bias(celltypes, assays, celltype_assays, inverse_celltype_freqs, inverse_assay_freqs):
    """ Gets probability of trying to minimize assay_bias that minimizes the bias the best overall.
    """

    celltype_to_assayindices, assay_to_celltypeindices = get_indice_maps(celltypes, assays, celltype_assays)

    # Now running the optimization, to figure out the optimal point of trying to prioritize minimizing bias toward
    # certain assays or certain cell types.
    n_steps = 20
    celltype_biases = np.zeros((n_steps))
    assay_biases = np.zeros((n_steps))

    for i in range(n_steps):

        assay_bias = (1 / n_steps) * i

        sample_counts_balanced_ = get_sampling_counts(10_000, assay_bias, celltypes, assays,
                                                        inverse_celltype_freqs, inverse_assay_freqs,
                                                        celltype_to_assayindices, assay_to_celltypeindices)

        ### balanced sampling
        celltype_counts_balanced_ = sample_counts_balanced_.sum(axis=1)
        assay_counts_balanced_ = sample_counts_balanced_.sum(axis=0)

        celltype_biases[i] = np.std(celltype_counts_balanced_)
        assay_biases[i] = np.std(assay_counts_balanced_)

    #### Getting the best part...
    min_bias_i = np.argmin( celltype_biases + assay_biases )
    best_assay_bias = (1 / n_steps) * min_bias_i

    return best_assay_bias, celltype_to_assayindices, assay_to_celltypeindices

def get_sampling_counts(n_samples, assay_bias, celltypes, assays,
                        inverse_celltype_freqs, inverse_assay_freqs,
                        celltype_to_assayindices, assay_to_celltypeindices
                        ):

    sample_counts_balanced_ = np.zeros((len(celltypes), len(assays)), dtype=np.float32)

    ##### Just to get a sense for representation with our current sampling strategy....
    for _ in range(n_samples):

        ### Using balanced sampling
        priority = np.random.choice(['celltype', 'assay'], p=[1 - assay_bias, assay_bias])

        if priority == 'celltype':
            celltype = np.random.choice(celltypes, p=inverse_celltype_freqs)

            assay_indices = celltype_to_assayindices[celltype]
            assay_probs = inverse_assay_freqs[assay_indices] / sum(inverse_assay_freqs[assay_indices])
            assay = np.random.choice(assays[assay_indices], p=assay_probs
                                     )
        else:
            assay = np.random.choice(assays, p=inverse_assay_freqs)

            celltype_indices = assay_to_celltypeindices[assay]
            celltype_probs = inverse_celltype_freqs[celltype_indices] / sum(inverse_celltype_freqs[celltype_indices])
            celltype = np.random.choice(celltypes[celltype_indices], p=celltype_probs
                                        )

        celltypei = list(celltypes).index(celltype)
        assayi = list(assays).index(assay)

        sample_counts_balanced_[celltypei, assayi] += 1

    return sample_counts_balanced_







