import torch
from torch.utils.data import Dataset

import seaborn as sb
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyBigWig

import resource # Important to be able to open many files.

from dnacipher_train.helpers.dataset_classes import RegionFeaturesSubset

def load_signal_data(genome_file, sample_file, signal_file,
              train_regions_only=False, test_regions_only=False,
              train_expers_only=False, test_expers_only=False, n_regions=None,
              ):
    """ Loads the data that will either be used to fit or apply the normalization.
    """
    genome_rois = pd.read_csv(f'{genome_file}', sep='\t', header=None)

    # Row indices for the dataset
    if train_regions_only:
        load_region_indices = np.where(genome_rois.iloc[:, 2].values=='train')[0]
        if len(load_region_indices) == 0:
            raise Exception(
           f"Train regions requested, BUT 'train' labels not specified in column 3 of input region file: {genome_file}")

    elif test_regions_only:
        load_region_indices = np.where(genome_rois.iloc[:, 2].values=='test')[0]
        if len(load_region_indices) == 0:
            raise Exception(
           f"Test regions requested, BUT 'test' labels not specified in column 3 of input region file: {genome_file}")

    else:
        load_region_indices = None

    # Creating a virtual subset if we only want to train on a smaller fraction of the data, largely for testing.
    if type(n_regions) != type(None):

        if type(load_region_indices) == type(None):
            load_region_indices = np.array(list(range(n_regions)))
        else:
            load_region_indices = load_region_indices[0:n_regions]

    # Determining which experiments to load.
    sample_df = pd.read_csv(sample_file, sep='\t', index_col=0)

    all_celltype_assays = sample_df['celltype_assay'].values.astype(str)
    all_celltype_assays_tuple = [tuple(celltype_assay.split('---')) for celltype_assay in all_celltype_assays]

    celltypes = list(np.unique([celltype_assay[0] for celltype_assay in all_celltype_assays_tuple]))
    assays = list(np.unique([celltype_assay[1] for celltype_assay in all_celltype_assays_tuple]))

    if train_expers_only:
        load_exper_indices = np.where( sample_df['allocation'].values == 'train' )[0]
    elif test_expers_only:
        load_exper_indices = np.where( sample_df['allocation'].values == 'test' )[0]
    else:
        load_exper_indices = np.array( list(range( sample_df.shape[0] )) )

    load_celltype_assays = list( np.array(all_celltype_assays)[ load_exper_indices ] )

    # Setting up the pytorch dataset for the precomputed signal values.
    signal_dataset = SignalsPrecomputed(genome_rois.values, signal_file,
                                                        celltypes, assays, all_celltype_assays,
                                                        load_region_indices=load_region_indices,
                                                        load_celltype_assays=load_celltype_assays)

    return signal_dataset

class SignalsPrecomputed(Dataset):
    """ Represents a dataset of precompute signal values. Can be a split_h5 format or otherwise.
    """

    def __init__(self, regions, epi_precomp_file, celltypes, assays, all_celltype_assays,
                 load_region_indices=None, load_celltype_assays=None, device='cpu',
                 ):
        """ Initiliase a SignalsPrecomputed object.

        Args:
            regions (np.ndarray): Specifies the region midpoints, is also an index for the inputted .split_h5.
                                  Format is. Rows=regions. Columns=chr, start, end, matrix_index, index_in_matrix.
                                  NOTE since midpoints, start and end are equal.



            epi_precomp_file (str): Specifies the path to the precomputed signal values. Each row of data MUST match the
                                    regions specified in the inputted regions file, and the columns must match the
                                    celltype_assays provided on input.

            celltypes (list<str>): List of cell types/biological sample names.
                                   NOTE order is important, since used to specify cell type indices.

            assay (list<str>): List of assay names.
                               NOTE order is important, since used to specify assays indices that are inputted.

            all_celltype_assays (list<str>): Specifies all the precomputed celltype_assay values that are on the columns
                                    of the precomputed signal split.h5 file.

            load_region_indices (list<int>): Specifies which regions to load from this dataset object, enables virtual
                                    subsetting of the .h5 used as input. If None then will be able to iterate over
                                    all of the rows.

            load_celltype_assays (list<str>): Specifies a subset of the celltype_assays which will be loaded when iterating
                                over the dataset, allows for virtual subsetting of the dataset columns. If None then
                                will load all.
        """
        self.device = device
        self.celltypes = celltypes
        self.assays = assays
        self.all_celltype_assays = all_celltype_assays

        # Setting defaults if no subsets were inputted.
        if type(load_celltype_assays)==type(None):
            self.load_celltype_assays = all_celltype_assays
        else:
            self.load_celltype_assays = load_celltype_assays

        if type(load_region_indices)==type(None):
            self.load_region_indices = np.array(list(range(regions.shape[0])))
        else:
            self.load_region_indices = load_region_indices

        # These are the indices which are used for input to the DNACipher model, to represent 'celltype' or 'assay',
        # that is being referred to.
        self.indices = np.array([[celltypes.index(celltype_assay.split('---')[0]) \
                                  for celltype_assay in self.load_celltype_assays],
                                 [assays.index(celltype_assay.split('---')[1]) \
                                  for celltype_assay in self.load_celltype_assays]])

        self.all_n_expers = len(self.all_celltype_assays)  # Number of (celltype, assay) combinations which can be loaded from this dataset.
        self.regions = regions
        self.all_n_regions = regions.shape[0] # Number of regions which can be loaded from this dataset.

        #### Also specifying the subsetted regions!
        self.load_n_expers = len(self.load_celltype_assays)
        self.load_regions = self.regions[self.load_region_indices, :]
        self.load_n_regions = len( self.load_region_indices )

        # Getting the indices which map the celltype_assays that should be 'spit out' from this dataset,
        all_celltype_assays_list = list(self.all_celltype_assays)
        self.load_celltype_assay_indices = [all_celltype_assays_list.index(celltype_assay)
                                                 for celltype_assay in self.load_celltype_assays]
        if np.any(np.array(self.all_celltype_assays)[self.load_celltype_assay_indices] != np.array(self.load_celltype_assays)):
            raise Exception(
                     f"Celltype/assays in the precomputed epigenetic data does not represent provided sample file!")

        #### Dealing with case of precomputed epigenetic values !!!!
        self.epi_precomp_file = epi_precomp_file

        ##### Now attaching the dataset for reading from the .split.h5
        self.epi_precomp_dataset = RegionFeaturesSubset(self.regions, self.epi_precomp_file,
                                                        self.load_region_indices, self.load_celltype_assay_indices)

    def __iter__(self):
        """Only supported for precomputed epigenetic values, and only retrieves the precomputed
            epigenetic values in the batches stored within the .h5 data.
        """
        return self.generator()

    def generator(self):
        """Only supported for precomputed epigenetic values, and only retrieves the precomputed
            epigenetic values in the batches stored within the .h5 data.

            NOTE does not support random ordering of genome_rois!
        """
        for epimeasuresi, spliti in self.epi_precomp_dataset:
            yield epimeasuresi, spliti

    def __len__(self):
        """Number of regions that can be loaded."""
        return self.load_n_regions

    def __getitem__(self, idx):
        """ Retrieves the load_celltype_assay experiment values stored in the split.h5 for the inputted idx,
        with idx being related to the inputted load_region_indices, so the available regions to retrieve are
        only these.
        """

        if type(idx)==tuple:
            raise Exception("Column indexing not supported.")

        #### Dealing with format of the region indices
        if type(idx) == slice:  # Not supported
            raise Exception("Slicing regions not supported.")

        elif type(idx) == int:  # Supported, but convert to list for simplicity.
            idx = [idx]

        elif type(idx) != list:  # Only support list indexing.
            raise Exception(f"idx of type {type(idx)} not supported.")

        ##### Setting up the matrices that record the celltype,assay indices we are loading and their values across
        ##### the requested regions.
        all_cell_type_indices = np.repeat(self.indices[0:1, :], len(idx), axis=0).astype(np.int32)
        all_assay_indices = np.repeat(self.indices[1:2, :], len(idx), axis=0).astype(np.int32)

        all_cell_type_indices = torch.from_numpy( all_cell_type_indices ).to(self.device)
        all_assay_indices = torch.from_numpy( all_assay_indices ).to(self.device)

        all_cell_type_indices = all_cell_type_indices.squeeze()
        all_assay_indices = all_assay_indices.squeeze()

        # Epigenomic measurements, subsetted to requested, managed by the underlying RegionFeaturesSubset
        _, region_values = self.epi_precomp_dataset[ idx ]

        return {'celltype_input': all_cell_type_indices,
                'assay_input': all_assay_indices,
                }, region_values

    def plot_assay_nonzeros(self, nsplits=1, # Number of internal batches to use for plotting.
                            show=False, ax=None,
                            ):
        """ Violin plot of the assay distributions, to assess normalisation.

            NOTE only works with precomputed epimeasure splits.
        """
        assay_labels = np.array([celltypeassay.split('---')[1] for celltypeassay in self.load_celltype_assays])
        assay_values = {assay: [] for assay in self.assays}
        assay_sums = np.zeros(len(self.assays))
        assay_ns = np.zeros(len(self.assays))

        for i, (epimeasuresi, spliti) in enumerate(self):

            if i >= nsplits:
                break

            for i, assay in enumerate(self.assays):
                assay_indices = np.where(assay_labels==assay)[0]

                flat_assay_values = epimeasuresi[:, assay_indices].ravel()
                nonzero_assay_values = flat_assay_values[flat_assay_values>0]
                if len(nonzero_assay_values) > 0:
                    assay_values[assay].extend( nonzero_assay_values )
                    assay_sums[i] += sum(nonzero_assay_values)
                    assay_ns[i] += len(nonzero_assay_values)

        nonzero_bool = assay_ns > 0

        assay_values = list( assay_values.values() )
        assay_means = np.zeros(len(self.assays))
        assay_means[nonzero_bool] = assay_sums[nonzero_bool] / assay_ns[nonzero_bool]

        ### Plotting
        if type(ax)==type(None):
            fig, ax = plt.subplots(figsize=(30, 5))

        #plt.bar(self.assays, assay_means)
        #sb.barplot(assay_values)
        sb.boxenplot(assay_values)
        #sb.violinplot(assay_values)
        ax.xaxis.set_ticklabels(self.assays, fontdict={'rotation': 90})
        plt.xlabel('assays')
        plt.ylabel('assay nonzero-mean')
        if show:
            plt.show()

class BigWigSignals(Dataset):
    """ Loads epigenetic average values across a set of bigWig files for a given set of genomic regions.
    """

    def __init__(self, regions, celltypes, assays, data,
                 region_pad=2997, arcsinh=True, device='cpu', only_start=True, exact=True):
        """ Initiliase a SpecifiedRegionsData object.

        Args:
            regions (np.ndarray): Specifies the region midpoints.
                                  In BED format. Rows=regions. Columns=chr, start, end.
                                  NOTE since midpoints, start and end are equal.

            celltypes (list<str>): List of cell types/biological sample names.
                                   NOTE order is important, since used to specify cell type indices.

            assay (list<str>): List of assay names.
                               NOTE order is important, since used to specify assays indices.

            data (dict<tuple<str, str>: str>): Dictionary with paths to the .bigWig files, specified by tuples.
                                        First value of the tuple is the cell type name (must be in the celltypes list),
                                        and the second value is the assay name (must be in assay list).
                                        The values of the dictionary must be paths to .bigWig files, which specify
                                        the signal values for the (celltype, assay) experiment specified by the key.

            region_pad (int): Specifies additional padding, +/-, to be added to the intervals when read in.
                              This is useful if the regions describe midpoints, and want to experiment with
                              different sizes around a midpoint (i.e. start/end are equal in the inputted regions
                              numpy array). For example, if specify region_pad=3000, every loaded regions will
                              have -3000 to the start, and +3000 to the end.
                              Default 2997, is results in ALMOST 6kb regions, leaving one token at the start for the
                              start token, to be provided as input to the Nucleotide Transformer to extract sequence
                              features.

            arcsinh (bool): Whether to np.arcsinh the signal values.

            transformer_model_name (str): Which Nucleotide Transformer model to use, must be one of
                                     '500M_human_ref', '500M_1000G', '2B5_1000G', '2B5_multi_species'.

            transformer_layer (int): Which layer of the transformer model to use as the sequence features.

            n_output_factors (int): Specify the number of bins for the signal value. Will bin the region into this many
                            bins, and average. Recommend a value of 240, which will result in an average over
                            24.75bp. This reduces noise in the signal alot.

            device (str): Specifies the device the output tensors will be allocated to.

            only_start (bool): Only considers the start position of the specified positions when determining the regions.

            n_tokens (int): Number of tokens represented. Default is 999, which is the number of tokens from nucleotide
                            transformer minus the start position (which is just pad really!)

            n_token_features (int): Specifies the number of token features to use. This is for use when the .h5 corresponds
                                    to the PCA processed embeddings, so it subset to the top N PCs and outputs these.
                                    It is useful to experimental with different token feature sizes, since this accounts
                                    for the largest number of model parameters.
        """
        self.device = device
        self.celltypes = celltypes
        self.assays = assays
        self.data = data
        self.celltype_assays = ['---'.join(celltype_assay) for celltype_assay in list(data.keys())]
        self.regions = regions
        self.only_start = only_start
        self.is_splith5 = self.regions.shape[1] > 3
        self.exact = exact
        self.file_type = list(self.data.values())[0].split('.')[-1]  # Are we reading bed files or bigWigs?
        if self.file_type not in ['bed', 'bigWig']:
            raise Exception(f'Received file type {self.file_type}, but onlly support .bed and .bigWig.')

        self.indices = np.array([[celltypes.index(celltype) for celltype, _ in data.keys()],
                                 [assays.index(assay) for _, assay in data.keys()]])

        self.track_paths = list(data.values())
        self.nexpers = len(self.data)  # Number of (celltype, assay) combinations.
        self.n_regions = regions.shape[0]

        self.region_pad = region_pad
        self.arcsinh = arcsinh

        self.total_values = self.n_regions * self.nexpers

        # Need to max sure file limit set to max to be able to do the opening of all the files in dense reading.
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
        if self.nexpers > hard_limit:  # Will not be possible to do this with that many files!!!!
            raise Exception("Number of files exceeds open file limit, will need to implement batching...")

        # Placeholder
        self.open_files = None

    def __len__(self):
        return self.n_regions

    def get_all_exper_values(self, chr_, start, end):
        """ Retrieves the values at the specified location across all experiments. Only supports bigWig averaging.
        """
        values_ = np.zeros((self.nexpers))
        celltype_indices = np.zeros((self.nexpers))
        assay_indices = np.zeros((self.nexpers))
        for celltype_assay_idx, celltype_assay in enumerate(self.celltype_assays):
            # Opening the bigWig.
            bigwig_open = self.open_files[celltype_assay_idx]  # pyBigWig.open( self.track_paths[celltype_assay_idx] )
            celltype_indices[celltype_assay_idx] = self.indices[0, celltype_assay_idx]
            assay_indices[celltype_assay_idx] = self.indices[1, celltype_assay_idx]

            #### IN some experiments, not all chromosomes are represented (e.g. ChrY), so return 0 for these cases.
            chrom_sizes = bigwig_open.chroms()
            chr_present = chr_ in chrom_sizes
            # Dealing with case where region but be at end of chromosome, not represented in the bigWig.
            if chr_present and chrom_sizes[chr_] < end:
                if chrom_sizes[chr_] < start:
                    chr_present = False  # This region of chromosome not represented!
                else:  # truncate the end of the region, to avoid going over the edge.
                    end = chrom_sizes[chr_]

            ##### Getting the values associated with the region
            if chr_present:  # Get the average across the region....
                values = np.array(bigwig_open.stats(chr_, start, end, exact=self.exact))
                if type(values[0]) == type(None):
                    values = np.array([0], dtype=np.float32)

            else:  # Chromosome not represented, because had no peaks/signal for the celltype,assay
                values = np.array([0], dtype=np.float32)
                # values[0] = 'apple'

            if self.arcsinh:
                values = np.arcsinh(values)

            # Catching incorrectly formatted files, since if -log10(pval) should be positive!!!!
            if np.any(values < 0):
                print(f"{celltype_assay} has negatives! File: {self.data[tuple(celltype_assay.split('---'))]}",
                      flush=True)
            ##### Storing the resulting values....
            values_[celltype_assay_idx] = values[0]

        return values_, celltype_indices, assay_indices

    def close_files(self):
        # Shutting all the open bigwig files !!!
        if self.open_files is not None:
            [bigwig_open.close() for bigwig_open in self.open_files]
            self.open_files = None

    def set_open_files(self):
        #### Will open all these files!!!
        try:
            self.open_files = []
            for exper_idx in range(self.nexpers):
                self.open_files.append(pyBigWig.open(self.track_paths[exper_idx]))

        except Exception:
            raise Exception("Couldn't open file:", self.track_paths[exper_idx])

    def __del__(self):
        # best-effort cleanup in workers
        try:
            self.close_files()
        except Exception:
            pass

    def __getitem__(self, idx):
        """ For the region specified by idx, return a random (celltype, assay) experiment's
        values for this region. If self.arcsinh, returns the np.arcsinh of this region.
        If exper_index is not None, returns the specified experiment's values (with the index
        being those specified by self.data.keys().

        In terms of the values that idx can take, only a list of indices, one index (referring
        to the region), or two indices (first referring to the region, second the experiment)
        are supported.
        """

        if self.open_files is None:
            self.set_open_files()

        #### Dealing with whether one or two dimensions specified
        if type(idx) == tuple:  # Two dimensions specified, second is the experimental index!
            raise Exception("Dual indexing not supported.")

        #### Dealing with format of the region indices
        if type(idx) == slice:  # Not supported
            raise Exception("Slicing regions not supported.")

        elif type(idx) == int:  # Supported, but convert to list for simplicity.
            idx = [idx]

        elif type(idx) != list:  # Only support list indexing.
            raise Exception(f"idx of type {type(idx)} not supported.")

        ##### Retrieving the region specified by the index
        all_cell_type_indices = np.zeros((len(idx), self.nexpers), dtype=np.int32)
        all_assay_indices = np.zeros((len(idx), self.nexpers), dtype=np.int32)

        n_values = self.nexpers  # We need to store the value for each experiment!!!!
        region_values = np.zeros((len(idx), n_values), dtype=np.float32)  # Epigenomic measurements.

        # Opening the bigwig files to read from:
        #self.set_open_files()

        for i, index in enumerate(idx):
            chr_, start = self.regions[index, 0:2]  # in-case more columns specified..
            # Going +/- the inputted start position.
            end = start + self.region_pad
            start -= self.region_pad

            ##########################################################################
            ##### Dense strategy means we will get the values for all experiments...
            values_, celltype_indices, assay_indices = self.get_all_exper_values(chr_, start, end)
            region_values[i, :] = values_
            all_cell_type_indices[i, :] = celltype_indices
            all_assay_indices[i, :] = assay_indices

        #self.close_files()
        region_values = torch.from_numpy(region_values.astype(np.float32))

        # Putting on correct device..
        all_cell_type_indices = torch.from_numpy(all_cell_type_indices).to(self.device)
        all_assay_indices = torch.from_numpy(all_assay_indices).to(self.device)

        # Shaping correctly.
        all_cell_type_indices = all_cell_type_indices.squeeze()
        all_assay_indices = all_assay_indices.squeeze()
        region_values = region_values.squeeze()

        return {'celltype_input': all_cell_type_indices,
                'assay_input': all_assay_indices,
                'region_indices': idx, # Just to make sure they are being iterated by the data loader in correct order!
                }, region_values
