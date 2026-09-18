
from dnacipher_train.signals.signal_dataset_classes import load_signal_data
from dnacipher_train.helpers.dataset_classes import RegionFeaturesSubset

from torch.utils.data import Dataset

class Precomputed(Dataset):
    """ Represents a dataset of precompute signal values and embeddings.
    """

    def __init__(self, genome_file, sample_file, signal_file, embed_file,
                       train_regions_only=False, test_regions_only=False,
                       train_expers_only=False, test_expers_only=False, n_regions=None,
              ):
        """ Dataset representing both precomputed signals across cell types and assays, and also DNA sequence embeddings.
        """

        self.signal_data = load_signal_data(genome_file, sample_file, signal_file,
                                       train_regions_only, test_regions_only,
                                       train_expers_only, test_expers_only, n_regions=n_regions)

        self.embed_data = RegionFeaturesSubset(self.signal_data.regions, embed_file,
                                               self.signal_data.load_region_indices)

    def __len__(self):
        return len(self.signal_data.load_region_indices)

    def __getitem__(self, idx):
        """ Retrieves the region signals values, experiment indices (celltypes, assays), and the sequence embeddings.
        """

        if type(idx) == tuple:
            raise Exception("Column indexing not supported.")

        #### Dealing with format of the region indices
        if type(idx) == slice:  # Not supported
            raise Exception("Slicing regions not supported.")

        elif type(idx) == int:  # Supported, but convert to list for simplicity.
            idx = [idx]

        elif type(idx) != list:  # Only support list indexing.
            raise Exception(f"idx of type {type(idx)} not supported.")

        inputs, signal_values = self.signal_data[idx]
        _, embed_values = self.embed_data[idx]

        inputs.update( {'seqfeatures_input': embed_values,
                        'region_indices': idx} # Useful so can determine which indices to load if comparing between sets.
                    )

        return inputs, signal_values

