import h5py

import torch
from torch.utils.data import Dataset

import numpy as np

class RegionFeaturesSubset(Dataset):
    """ A light wrapper over RegionFeatures, which allows for internal subsetting of rows and columns, so the same
        underlying .h5 can be represented, but any indices will be relative to a subsetted version of the matrix
        stored in the .h5.
    """

    def __init__(self, regions, region_feature_file, row_indices=None, column_indices=None):
        """ Initialise a RegionFeaturesSubset object, which is a dataset wrapper for loading from a .split.h5 object.

                Args:
                    regions (np.ndarray): Specifies the region midpoints, is also an index for the inputted .split_h5.
                                          Format is. Rows=regions. Columns=chr, start, end, matrix_index, index_in_matrix.
                                          NOTE since midpoints, start and end are equal.

                    region_feature_file (str): Specifies the path to the precomputed features per region. Each row of data MUST match the
                                            regions specified in the inputted regions file.

                    row_indices (np.array<int>): Specifies row indices to subset the inputted matrix to.

                    column_indices (np.array<int>):  Specifies column indices to subset the inputted matrix to.
        """

        self.full_dataset = RegionFeatures(regions, region_feature_file)

        if type(row_indices) == type(None):
            row_indices = np.array( list( range( regions.shape[0] ) ) )

        if type(column_indices) == type(None):
            # Need to peek inside the file in order to determine the number of columns:
            open_file = h5py.File(region_feature_file, 'r')
            n_cols = open_file[str(0)].shape[1]
            open_file.close()

            column_indices = np.array( list( range( n_cols ) ) )

        self.row_indices = row_indices
        self.column_indices = column_indices

        self.n_rows = len(row_indices)
        self.n_cols = len(column_indices)

    def __len__(self):
        """Number of regions that can be loaded."""
        return self.n_rows

    def shape(self):
        return (self.n_rows, self.n_cols)

    def __iter__(self):
        return self.generator()

    def generator(self):
        """Only supported for precomputed epigenetic values, and only retrieves the precomputed
            epigenetic values in the batches stored within the .h5 data.

            NOTE does not support random ordering of genome_rois!
        """
        # The underlying full dataset will return all of the values, so will modify the generator here
        # by subsetting the values!
        start_index = 0
        for values, spliti in self.full_dataset:

            # Need to figure out the global indices this split of the data represents:
            end_index = start_index + values.shape[0]

            keep_bool = [index in self.row_indices for index in list(range(start_index, end_index))]

            start_index = end_index

            yield values[:, self.column_indices][keep_bool, :], spliti

    def __getitem__(self, idx):
        """ Retrieves the subsetted column values stored in the split.h5 for the inputted idx,
            with idx being related to the subsetted rows.
        """

        #### Parsing the inputting idx
        if type(idx)==tuple:
            raise Exception("Column indexing not supported.")

        #### Dealing with format of the region indices
        if type(idx) == slice:  # Not supported
            raise Exception("Slicing regions not supported.")

        elif type(idx) == int:  # Supported, but convert to list for simplicity.
            idx = [idx]

        elif type(idx) != list:  # Only support list indexing.
            raise Exception(f"idx of type {type(idx)} not supported.")

        # Getting the appropriate values!
        new_idx = list( self.row_indices[idx] )  # Has to be relative to the virtual subset!
        _, values = self.full_dataset[new_idx, self.column_indices]

        return {}, values

class RegionFeatures(Dataset):
    """ Iterates over a split.h5 which contains a single matrix of features per region.
    """

    def __init__(self, regions, region_feature_file,
                 ):
        """ Initialise a RegionFeatures object, which is a dataset wrapper for loading from a .split.h5 object.

        Args:
            regions (np.ndarray): Specifies the region midpoints, is also an index for the inputted .split_h5.
                                  Format is. Rows=regions. Columns=chr, start, end, matrix_index, index_in_matrix.
                                  NOTE since midpoints, start and end are equal.

            region_feature_file (str): Specifies the path to the precomputed features per region. Each row of data MUST match the
                                    regions specified in the inputted regions file.
        """
        self.regions = regions
        self.all_n_regions = regions.shape[0] # Number of regions which can be loaded from this dataset.

        #### Dealing with case of precomputed epigenetic values !!!!
        self.region_feature_file = region_feature_file

        ################## Checking the .h5
        region_features = h5py.File(self.region_feature_file, 'r')
        self.region_feature_datasets = list(region_features.keys())
        self.n_features = region_features[self.region_feature_datasets[0]].shape[1]
        region_features.close()

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
        ####### Preloading by batches already defined within the .split.h5
        open_file = h5py.File(self.region_feature_file, 'r')

        # Index of the stored matrices.
        data_indexes = self.regions[:, 3].astype(str)
        precomp_datasets = np.unique(data_indexes.astype(int)).astype(str)

        # Indexes within each of the stored matrices.
        indexes_in_data = self.regions[:, 4].astype(int)
        data_index_indices = [np.where(data_indexes==dataset_index)[0] for dataset_index in precomp_datasets]

        datai = 0
        while datai < len(precomp_datasets):
            data_index_indice = data_index_indices[datai]

            # This is in-case we have subsetted to less than full split.
            index_in_data = indexes_in_data[data_index_indice]

            # indexing into the stored split matrix, then the appropriate index in that matrix, and then subsetting
            # the columns to the celltype_assays on the columns.
            loaded = open_file[precomp_datasets[datai]][index_in_data, :]
            # we need to yield the dataset it belong to as well, so know where to place the loaded data if
            # doing something to this data then writing out to a different file.
            yield loaded, precomp_datasets[datai]
            datai += 1

        open_file.close()

    def __len__(self):
        """Number of regions that can be loaded."""
        return self.all_n_regions

    def __getitem__(self, idx):
        """ Retrieves the load_celltype_assay experiment values stored in the split.h5 for the inputted idx,
        with idx being related to the inputted load_region_indices, so the available regions to retrieve are
        only these.
        """

        #### Dealing with whether one or two dimensions specified
        if type(idx) == tuple:  # Two dimensions specified, second is the experimental index!
            idx, column_indices = idx

        else:
            column_indices = list(range(self.n_features))

        #### Dealing with format of the region indices
        if type(idx) == slice:  # Not supported
            raise Exception("Slicing regions not supported.")

        elif type(idx) == int:  # Supported, but convert to list for simplicity.
            idx = [idx]

        elif type(idx) != list:  # Only support list indexing.
            raise Exception(f"idx of type {type(idx)} not supported.")

        ##### Setting up the matrices that record the celltype,assay indices we are loading and their values across
        ##### the requested regions.
        region_values = np.zeros((len(idx), len(column_indices)), dtype=np.float32)  # Epigenomic measurements.

        # Opening the .split.h5 to load from.
        region_features = h5py.File(self.region_feature_file, 'r')

        ##### The for loop below is not necessary if we didn't use split indexing and both are precomputed.
        for i, index in enumerate(idx):

            # Always assumes split indexing, so first index is the split matrix, second index is the index in the split matrix.
            dataset_index, index_in_dataset = self.regions[index, [3, 4]]

            region_values[i, :] = region_features[str(dataset_index)][index_in_dataset, column_indices]

        region_features.close()

        region_values = torch.from_numpy(region_values.astype(np.float32)) # Always on cpu because is the on the output.

        # In-case there is an extra dimension that should be dropped off.
        region_values = region_values.squeeze()

        return {}, region_values
