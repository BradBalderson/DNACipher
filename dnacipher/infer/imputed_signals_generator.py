""" Related to inference performed by the DNACipher model
"""

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from dnacipher_train.embed.embedding_generator import EmbeddingGenerator

class ImputedSignals(Dataset):
    """ Represents a dataset of signal imputations from the DNACipher model.
    """

    def __init__(self, bed_df, dnacipher_model, embed_model, fasta_file, celltype_indexes, assay_indexes,
                 ):
        """ Dataset representing both precomputed signals across cell types and assays, and also DNA sequence embeddings.

        Parameters
        ----------
        bed_df: pd.DataFrame
            Bed file specifying chr, start, end queries to generate signal predictions for!!
        dnacipher_model: DNACipherModel
            DNACipher model, with trained weights already loaded.
        embed_model: EmbeddingModel
            EmbeddingModel class, that generates the embedding DNA sequence representation.
        fasta_file: str
            Path to the fasta file to read the sequences from.
        celltype_indexes:
            Celltype indexes to pass to the DNACipher model to infer signals.
        assay_indexes:
            Assay indexes to pass to the DNACipher model to infer signals.
        axis:
            Specifies the AXIS to return results. Can either return all query regions within indices for the celltypes/
            assays. OR can return all celltypes/assays for a given set of regions.
        """

        #### Now setting up the embedding generator, which will enable ultra-fast generate of embedding, by not
        #### centring the query sequences on each query, but instead at pre-determined genome tiles, and then extracting
        #### the embedding from those tiles. Thereby closeby sequences do no require loading completely new sequences,
        #### which is a major bottle-neck.
        self.embed_generator = EmbeddingGenerator(embed_model, bed_df.values, fasta_file, None)

        self.dnacipher_model = dnacipher_model # So can run forward passes!
        self.device = self.dnacipher_model.device

        self.celltype_indexes = celltype_indexes # so we know what cell types we want.
        self.assay_indexes = assay_indexes

        # Total number of signals that can be generated, corresponding to the number of query regions and the
        # number of experiments to infer !!
        self.n_signals = int( bed_df.shape[0] * len(celltype_indexes) )

        # Creating a map of the corresponding indices for each signal, so we know what a query index corresponds to
        # in terms of the region, celltype, assay !!!!
        self.signal_indices = np.full((self.n_signals, 4), fill_value=np.nan, dtype=np.int64)
        signali = 0
        for regioni in range(bed_df.shape[0]):
            for experi, (celltypei, assayi) in enumerate( zip(celltype_indexes, assay_indexes) ):

                self.signal_indices[signali, :] = [regioni, experi, celltypei, assayi]

                signali += 1

        if np.any(np.isnan(self.signal_indices)):
            raise Exception("Did not calculate signal indices correctly!")

    def __len__(self):
        return self.n_signals

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

        #### Retrieving the indices to query:
        region_indices = self.signal_indices[idx, 0]
        exper_indices = self.signal_indices[idx, 1]
        ct_indices = self.signal_indices[idx, 2]
        assay_indices = self.signal_indices[idx, 3]

        #### Retrieving the embeddings of the corresponding regions.
        _, region_embeddings = self.embed_generator[list(region_indices)]

        # Formatting input queries
        region_embeds_input = torch.from_numpy(region_embeddings).to(device=self.device)

        celltype_index_input = torch.tensor(ct_indices, device=self.device).unsqueeze(0).reshape(-1, 1)
        assay_index_input = torch.tensor(assay_indices, device=self.device).unsqueeze(0).reshape(-1, 1)

        #### Generating the predicted signals at the corresponding regions:
        with torch.no_grad():
            region_signals = self.dnacipher_model(celltype_index_input, assay_index_input, region_embeds_input)

        region_signals_numpy = region_signals.cpu().detach().numpy()[:, 0]

        #### Bundling up the required outputs!
        indices = {'region_indices': region_indices, 'exper_indices': exper_indices}

        return indices, region_signals_numpy





