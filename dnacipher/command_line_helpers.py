""" Functions to help with some of the CLI commands, in order to keep the main.py clean.
"""

import sys
from pathlib import Path

import torch

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from . import dna_cipher_infer as dnaci
from . import dna_cipher_model as dnacm
from . import dna_cipher_plotting as dnapl
from . import deep_variant_impact_mapping as dvim

def get_best_device():
    """Gets best device available to run DNACipher"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("Using TPU", file=sys.stdout, flush=True)

    except:

        if torch.cuda.is_available():
            device = 'cuda:0'
            print("Will use cuda GPU", file=sys.stdout, flush=True)

        elif torch.backends.mps.is_available():
            # device = 'mps:0' # Currently cannot make long-sequence inference with mps:0 due to this:
            # print("Will use apple metal GPU")
            # https://github.com/pytorch/pytorch/issues/134416
            # Should be fixed in future version
            device = 'cpu'
            print('Using CPU, mps available but currently not working for long-sequence inference.', file=sys.stdout, flush=True)

        else:
            device = 'cpu'
            print("No apparent GPU available, using CPU (will be slow).", file=sys.stdout, flush=True)

    return device

def load_context_data():
    """ Loads the metadata used to train the model, important to see predictable contexts. """

    git_path = Path(__file__).parent # Should be the DNACipher path
    
    model_path = f'{git_path}/weights/'
    sample_file_path = f'{model_path}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv'

    sample_df = pd.read_csv(sample_file_path, sep='\t', index_col=0)

    celltype_assays = [tuple(celltype_assay.split('---')) for celltype_assay in
                       sample_df['celltype_assay'].values.astype(str)]
    celltype_assay_labels = sample_df['allocation'].values # Specifies which were used as train/test
    celltypes = list(np.unique([celltype_assay[0] for celltype_assay in celltype_assays]))
    assays = list(np.unique([celltype_assay[1] for celltype_assay in celltype_assays]))

    return celltype_assays, celltype_assay_labels, celltypes, assays

def load_dnacipher(device, fasta_file_path, verbose):
    """ Loads the DNACipher model. """
    
    if type(device) == type(None):
        device = get_best_device()

    git_path = Path(__file__).parent # Should be the DNACipher path
    
    model_path = f'{git_path}/weights/'
    weights_path = f'{model_path}TRAINING_DNACV5_MID-AVG-GENOME_ORIG-ALLOC_ENFORMER0_FINETUNE_STRATMSE_model_weights.pth'
    sample_file_path = f'{model_path}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv'
    
    # Some extra parameters about the model which cannot be read from the weights:
    config = {'activation_function': 'gelu',
              'relu_genome_layer': True, # Is actually gelu, this just means to use activate function for genome layer.
              'layer_norm': True,
              'n_token_features': 3072,
              'relu_output': True,
              'epi_summarise_method': 'flatten',
             }
    
    dnacipher = dnaci.DNACipher(weight_path=weights_path, sample_file=sample_file_path, config=config,
                            genome_file=fasta_file_path,
                            device=device
                           )
    if verbose:
        print("Successfully loaded DNACipher model", file=sys.stdout, flush=True)
    
    return dnacipher

def parse_general_input(celltypes, assays, device, fasta_file_path, verbose):
    """ Parses general input common to most functional calls """

    dnacipher = load_dnacipher(device, fasta_file_path, verbose)
        
    celltypes = list(open(celltypes, 'r'))[0].strip('\n').split(',,')
    assays = list(open(assays, 'r'))[0].strip('\n').split(',,')

    return dnacipher, celltypes, assays

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
    
    



