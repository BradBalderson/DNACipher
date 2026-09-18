import sys
import torch

from pathlib import Path

import numpy as np
import pandas as pd

import dnacipher_train.embed.md as md

import dnacipher_train.model.dna_cipher_model as dnac
import dnacipher_train.infer.dna_cipher_infer as dnaci

import inspect
import yaml

import matplotlib.pyplot as plt

def get_best_device(training=False):
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

        elif torch.backends.mps.is_available() and not training:
            # device = 'mps:0' # Currently cannot make long-sequence inference with mps:0 due to this:
            # print("Will use apple metal GPU")
            # https://github.com/pytorch/pytorch/issues/134416
            # Should be fixed in future version
            device = 'cpu'
            print('Using cpu, apple silicon mps available but for inference is unclear if the underlying embedding model '
                  'will run on it. Can override by explicitly setting the device.', file=sys.stdout, flush=True)

        elif torch.backends.mps.is_available() and training:
            device = 'mps:0'
            print("Will use apple silicon GPU", file=sys.stdout, flush=True)

        else:
            device = 'cpu'
            print("No apparent GPU available, using CPU (likely slow).", file=sys.stdout, flush=True)

    return device


def load_context_data(sample_file_path=None, verbose=True):
    """ Loads the metadata used to train the model, important to see predictable contexts. """

    if type(sample_file_path)==type(None): # Load the default.
        git_path = Path(__file__).parent  # Should be the DNACipher path
        model_path = f'{git_path}/weights/'
        sample_file_path = f'{model_path}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv'

        if verbose:
            print(f"Loaded default sample_file: {sample_file_path}\n", file=sys.stdout, flush=True)

    sample_df = pd.read_csv(sample_file_path, sep='\t', index_col=0)

    celltype_assays = [tuple(celltype_assay.split('---')) for celltype_assay in
                       sample_df['celltype_assay'].values.astype(str)]
    celltype_assay_labels = sample_df['allocation'].values  # Specifies which were used as train/test
    celltypes = list(np.unique([celltype_assay[0] for celltype_assay in celltype_assays]))
    assays = list(np.unique([celltype_assay[1] for celltype_assay in celltype_assays]))

    return celltype_assays, celltype_assay_labels, celltypes, assays

def parse_model_config(model_config_file, log_file):

    print(f'\nLoading the model_config... has state:\n', file=log_file, flush=True)
    model_config = yaml.load( open(model_config_file, 'r'), yaml.Loader )
    print(f'{model_config}\n', file=log_file, flush=True)

    required_params = ['n_celltype_factors', 'n_assay_factors', 'n_seq_factors', 'n_layers', 'n_nodes']
    missing_params = [param for param in required_params if param not in model_config]
    if len(missing_params) > 0:
        raise Exception(f"DNACipher model config missing required parameters: {missing_params}")

    # Additional named arguments that are allowed:
    allowed_params = list(inspect.signature(dnac.DNACipherModel).parameters.keys())
    extra_params = {param: model_config[param] for param in model_config
                    if param not in required_params and param in allowed_params}
    disallowed_params = {param: model_config[param] for param in model_config
                         if not param in allowed_params}
    if len(disallowed_params) > 0:
        print(f'\nDisallowed params provided in the model_config_file, these will be ignored:{disallowed_params}\n',
                                                                                            file=sys.stderr, flush=True)

    return model_config, extra_params

def load_dnacipher_infer(fasta_file_path, log_file,
                         weights_path=None, sample_file_path=None, config_path=None,
                         model_name=None, embed_config_yaml=None,
                         device=None, verbose=True):
    """ Loads the DNACipher inference model. """

    if type(device) == type(None):
        device = get_best_device()

    git_path = Path(__file__).parent  # Should be the DNACipher path

    if type(weights_path)==type(None):
        weights_path = f'{git_path}/weights/FULL_TRAIN5_model_weights.pth'

    if type(sample_file_path)==type(None):
        sample_file_path = f'{git_path}/weights/encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv'

    if type(config_path)==type(None):
        config_path = f'{git_path}/configs/default_model_config.yaml'

    if verbose:
        print(f"Loading sample_file: {sample_file_path}\n", file=log_file, flush=True)
        print(f"Loading model weights: {weights_path}\n", file=log_file, flush=True)
        print(f"Loading model config: {config_path}\n", file=log_file, flush=True)

    # Some extra parameters about the model which cannot be read from the weights:
    model_config, extra_params = parse_model_config(config_path, log_file)

    # Setting up the embedding model that was used to train this version of DNACipher:
    if type(model_name) == type(None):
        model_name = 'enformer'

    #### Creating the EmbeddingModel.
    if model_name not in md.model_names:
        raise Exception(f"Inputted {model_name} not currently supported, must be one of {md.model_names}")

    #### Reading any extra inputs to the model setup...
    if type(embed_config_yaml) == type(None):
        embed_config = {}
    else:
        embed_config = yaml.load( open(embed_config_yaml, 'r'), yaml.Loader )

    #### Setting up the embedding model !!
    print( f'Loading the embedding {model_name} model...\n', file=log_file, flush=True)
    embed_model = md.model_classes[ md.model_names.index(model_name) ](device, **embed_config)

    print(f'Embedding model is on device: {embed_model.model.device}\n', file=log_file, flush=True)
    print(f"Embedding model details:", str(embed_model), file=log_file, flush=True)

    dnacipher = dnaci.DNACipher(extra_params, weights_path, sample_file_path, fasta_file_path, embed_model, device
                                )
    if verbose:
        print("Successfully loaded DNACipher inference model", file=sys.stdout, flush=True)

    return dnacipher


def parse_general_input(celltypes, assays, fasta_file_path, log_file,
                         weights_path=None, sample_file_path=None, config_path=None,
                         model_name=None, device=None, verbose=True):
    """ Parses general input common to most functional calls """

    dnacipher = load_dnacipher_infer(fasta_file_path, log_file, weights_path=weights_path,
                                     sample_file_path=sample_file_path, config_path=config_path, model_name=model_name,
                                     device=device, verbose=verbose
                                     )

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
