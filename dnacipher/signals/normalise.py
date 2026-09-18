"""
Functions for helping to normalise the epigenetic data.
"""

import time
import numpy as np

def iterate_seqs(operation, dataset, transform_file, norm_stats, log_file, arcsinh=True, post_arcsinh=True):
    """Iterates through the sequences, performing some operation...."""
    start_fit = time.time()

    rna_norm = norm_stats['RNA_nonZeroShiftedLog2']
    norm_obj = norm_stats['Normalise']

    region_counts = 0
    for i, (epimeasuresi, spliti) in enumerate(dataset):  # Batches of epigenetic measures stored in the .split.h5

        region_counts += epimeasuresi.shape[0]

        if arcsinh:
            epimeasuresi = np.arcsinh( epimeasuresi )

        ##### Updating means
        if operation == 'rna-fit':
            rna_norm.partial_fit( epimeasuresi )

        # Now fit the Normalisation object, which will apply the RNA-seq normalisation object first to each batch
        # for each fit.
        elif operation == 'norm-fit':
            epimeasuresi_rna_normed = rna_norm.transform( epimeasuresi )

            norm_obj.partial_fit( epimeasuresi_rna_normed )

        elif operation == 'transform':
            epimeasuresi_rna_normed = rna_norm.transform( epimeasuresi )
            epimeasuresi_normed = norm_obj.transform( epimeasuresi_rna_normed )

            if post_arcsinh: # Helps variance-stabilisation of extreme peaks.
                epimeasuresi_normed = np.arcsinh( epimeasuresi_normed )

            transform_data = transform_file.create_dataset(spliti, shape=epimeasuresi_normed.shape,
                                                             dtype='f', compression='lzf',  # compression_opts=9
                                                             )
            transform_data[:,:] = epimeasuresi_normed

        else:
            raise Exception(f'operation not recognised: {operation}, should be "rna-fit", "norm-fit" or "transform".')

        if i % 10 == 0:
            end_fit = time.time()
            mins = round((end_fit - start_fit) / 60, 3)
            print(f"{operation} for {region_counts} seqs of {len(dataset)} seqs in {mins}minutes.",
                  file=log_file, flush=True)

    end_fit = time.time()
    mins = round((end_fit - start_fit) / 60, 3)
    print(f"{operation} for {region_counts} seqs of {len(dataset)} seqs in {mins}minutes.\n\n",
          file=log_file, flush=True)

class NonzeroShiftedLog2Norm():

    def __init__(self, assays):
        """ Normalisation, which can be progressively fit on data that is loaded in batches.
            It is meant for the RNA-seq assays,
            on fit:
                1) Keeps track of the global nonzero minimum seen.
            on transform:
                1) log2 transforms the nonzero values.
                2) takes off log2 of the global minimum for the nonzero values, to shift the approximate-normal
                    distribution to positive range.
        """
        self.assays = list( assays )
        self.n_assays = len( assays )
        self.assay_indices = None
        self.reset()

    def reset(self):
        """Resets the Z-score stats."""
        self.mins = np.zeros( (self.n_assays) ) # Min values seen so far
        self.mins[:] = 9999999

    def set_input_assay_indices(self, celltype_assays):
        """ Sets the indices in the inputted celltype-assays which correspond to the assays represented by this object.
        """
        self.set_celltype_assays = celltype_assays
        assay_labels = np.array([celltypeassay.split('---')[1] for celltypeassay in celltype_assays])
        self.assay_indices = [np.where(assay_labels == assay)[0] for assay in self.assays]

    def partial_fit(self, regions_by_celltypeassays): # This means assay values across all cell types and regions.
        """ Update estimate of the norm-stats for the nonzero assay values.
        """
        if type(self.assay_indices)==type(None):
            raise Exception("Need to run set_input_assay_indices first!")

        for assay_index, assay in enumerate(self.assays):
            flat_assay_values = regions_by_celltypeassays[:, self.assay_indices[assay_index]].ravel()

            nonzero_indices = np.where(flat_assay_values > 0)[0]

            if len(nonzero_indices)==0:
                continue

            # Updating observation count
            current_min = self.mins[assay_index]
            this_min = np.min( flat_assay_values[nonzero_indices] )
            if this_min < current_min:
                self.mins[assay_index] = this_min

    def __len__(self):
        return len( self.assays )

    def transform(self, regions_by_celltypeassays, # Regions are rows, columns specifies the celltype-assay combo measures
                  ):
        """ Apply the learnt transformation to the appropriate assays.

            NOTE that if extra assays are present, this is handled.
        """
        if len(self.set_celltype_assays)!=regions_by_celltypeassays.shape[1]:
            raise Exception(f"Len of celltype_assays is {len(self.set_celltype_assays)} but regions_by_celltypeassays "
                            f"columns is length {regions_by_celltypeassays.shape[1]}. They need to match.")

        """ Applying a shift, to prevent zero values for the RNA-seq data.
        """
        for assay_index, assay_indexes in enumerate(self.assay_indices):
            assay_sub = regions_by_celltypeassays[:, assay_indexes].ravel()
            nonzero_bool = assay_sub > 0
            assay_sub[nonzero_bool] = np.log2( assay_sub[nonzero_bool] ) - np.log2( self.mins[assay_index] )

            regions_by_celltypeassays[:, assay_indexes] = assay_sub.reshape((regions_by_celltypeassays.shape[0],
                                                                             len(assay_indexes)))

        return regions_by_celltypeassays

class Normalise():

    def __init__(self, assays):
        """ Normalisation object, which can be progressively fit on data that is loaded in batches.
        """
        self.assays = list( assays )
        self.n_assays = len(assays)
        self.reset()

    def reset(self):
        """Resets the Z-score stats."""
        self.n = np.zeros((self.n_assays))  # Number of observations so far

        self.sums = np.zeros((self.n_assays))  # Sum of observations so far
        self.means = np.zeros((self.n_assays))  # Mean value

        self.global_median = 0 # median of the means, used to scale all of the assays by to get equivalent mean.
        self.scale_factors = np.zeros((self.n_assays)) # Scaling factors, which determine what to multiply each
                                                       # assay value by to get equivalent mean between assays.
        self.celltype_assay_scalefactors = None

    def set_input_assay_indices(self, celltype_assays):
        """ Sets the indices in the inputted celltype-assays which correspond to the assays represented by this object.
        """
        self.set_celltype_assays = celltype_assays
        assay_labels = np.array([celltypeassay.split('---')[1] for celltypeassay in celltype_assays])
        self.assay_indices = [np.where(assay_labels == assay)[0] for assay in self.assays]

    def partial_fit(self, regions_by_celltypeassays): # This means assay values across all cell types and regions.
        """ Update estimate of the norm-stats for the nonzero assay values.
        """
        if type(self.assay_indices)==type(None):
            raise Exception("Need to run set_input_assay_indices first!")

        for assay_index, assay in enumerate(self.assays):
            flat_assay_values = regions_by_celltypeassays[:, self.assay_indices[assay_index]].ravel()

            nonzero_indices = np.where(flat_assay_values > 0)[0]
            if len(nonzero_indices)==0: # No non-zeros in this batch, so no point.
                continue

            # Updating observation count
            self.n[assay_index] += len(nonzero_indices)

            # Adding the sums
            self.sums[assay_index] += flat_assay_values[nonzero_indices].sum(axis=0)

            # Updating new means.
            self.means[assay_index] = self.sums[assay_index] / self.n[assay_index]

        # Updating the normalisation information.
        nonzero_bool = self.means > 0
        self.global_median = np.median( self.means[nonzero_bool] )
        self.scale_factors[nonzero_bool] = self.global_median / self.means[nonzero_bool]

    def calc_celltype_assay_scalefactors(self):
        """ This can only be performed after fitting the normstats, and must be performed before transformation.
        """
        total_celltype_assays = sum([len(assay_indexes) for assay_indexes in self.assay_indices])
        celltype_assay_scalefactors = np.zeros( (total_celltype_assays) )

        for i, assay in enumerate( self.assays ):
            assay_indices = self.assay_indices[i]
            celltype_assay_scalefactors[assay_indices] = self.scale_factors[i]

        self.celltype_assay_scalefactors = celltype_assay_scalefactors

    def transform(self, regions_by_celltypeassays, # Regions are rows, columns specifies the celltype-assay combo measures
                  ):
        """ Apply the learnt z-score transform for each feature.
        """
        if type(self.celltype_assay_scalefactors)==type(None): # Need to calculate these..
            raise Exception("Must run calc_celltype_assay_scalefactors(celltype_assays) first, with celltype_assays as "
                            "input that describes the columns of regions_by_celltypeassays.")

        # Multiplying by the scaling factors to get everything with the same mean.
        normalised = np.multiply(regions_by_celltypeassays, self.celltype_assay_scalefactors)
        return normalised


