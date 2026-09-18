""" Useful helper functions used across model training and evaluation.
"""

import time

import inspect
import yaml

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from pathlib import Path

from dnacipher_train.train.train_classes import Precomputed
from dnacipher_train.train.loss_weighting import optimize_weightings
from dnacipher_train.model.dna_cipher_model import DNACipherModel

def model_and_dataset_setup(genome_file, sample_file, signal_file, embed_file, batch_size, cpu, device,
                            model_config_file, log_file, learning_rate=0, dropout_rate=0, n_regions=None,
                            add_exper_weights=True):
    """ Sets up a randomly initialized DNACipher model based on user inputs.
    """

    # Loading the default DNACipher model if no config file provided:
    if type(model_config_file)==type(None):
        git_path = Path(__file__).parent.parent # Should be the DNACipher_Train repo path

        model_config_file = f"{git_path}/configs/default_model_config.yaml"

    print(f'\nSetting up the dataset...\n', file=log_file, flush=True)
    dataset_split = get_datasets_and_loaders(genome_file, sample_file, signal_file, embed_file, batch_size, cpu,
                                             n_regions=n_regions)

    # DEBUG - testing if the correct sizes are returned from each split dataset:
    #test_datasets_and_loaders(genome_file, sample_file, dataset_split, batch_size)

    #### Determining the celltype/assay weightings ...
    if add_exper_weights:
        print(f'\nDetermining the weightings for the experiments, based on a rarity of celltype/assay.',
              file=log_file, flush=True)
        start_ = time.time()
        experiment_loss_weights = optimize_weightings( sample_file )
        end_ = time.time()
        print(f'Finished in {round((end_-start_)/60, 3)}mins.\n', file=log_file, flush=True)

        # Need to separate the experiment weights into train/train and test/test, so the train_loss and val_loss
        # can be computed by pytorch lightning appropriately.
        train_exper_indices = dataset_split[(True, True)][0].signal_data.load_celltype_assay_indices
        test_exper_indices = dataset_split[(False, False)][0].signal_data.load_celltype_assay_indices

        train_exper_loss_weights = experiment_loss_weights[ train_exper_indices ]
        test_exper_loss_weights = experiment_loss_weights[ test_exper_indices ]

    else:
        train_exper_loss_weights = None
        test_exper_loss_weights = None

    # Checking these weights are correct with original version
    # TODO NOTE check only makes sense on the full dataset, since these weights were originally computed for the full
    #  dataset.
    # sample_df = pd.read_csv(sample_file, sep='\t', index_col=0)
    # weights_ = sample_df.sampling_weights.values
    # import matplotlib.pyplot as plt
    # plt.scatter(weights_, experiment_loss_weights)
    # plt.show()

    ### Now setting up the DNACipher model to train.
    print(f'Setting up the DNACipher model...\n', file=log_file, flush=True)
    train_dataset = dataset_split[(True, True)][0] # Retrieving for purpose of getting key parameters.
    n_celltypes = len( train_dataset.signal_data.celltypes )
    n_assays = len( train_dataset.signal_data.assays )
    n_seq_features_input = train_dataset.embed_data.n_cols

    print(f'\nLoading the model_config... has state:\n', file=log_file, flush=True)
    model_config = yaml.load( open(model_config_file, 'r'), yaml.Loader )
    print(f'{model_config}\n', file=log_file, flush=True)

    required_params = ['n_celltype_factors', 'n_assay_factors', 'n_seq_factors', 'n_layers', 'n_nodes']
    missing_params = [param for param in required_params if param not in model_config]
    if len(missing_params) > 0:
        raise Exception(f"DNACipher model config missing required parameters: {missing_params}")

    # Additional named arguments that are allowed:
    allowed_params = list(inspect.signature(DNACipherModel).parameters.keys())
    extra_params = {param: model_config[param] for param in model_config
                    if param not in required_params and param in allowed_params}
    disallowed_params = {param: model_config[param] for param in model_config
                         if not param in allowed_params}
    if len(disallowed_params) > 0:
        print(f'\nDisallowed params provided in the model_config_file, these will be ignored:{disallowed_params}\n',
                                                                                              file=log_file, flush=True)

    print(f'\nCreating the model.\n', file=log_file, flush=True)
    model = DNACipherModel(n_celltypes,  model_config['n_celltype_factors'], # Cell type layer information
                           n_assays, model_config['n_assay_factors'], # Assay layer information
                           n_seq_features_input, model_config['n_seq_factors'], # Seq embedding input information
                           model_config['n_layers'], model_config['n_nodes'], # Deep layer information.
                           # These parameters change between the warmup training model and the final model:
                           relu_output=False, stratified_loss=False,
                           # These weights are added now, but will not be used during the warmup training step:
                           celltype_assay_weights=train_exper_loss_weights,
                           celltype_assay_weight_test=test_exper_loss_weights,
                           # Important training params, that are likely dataset specific.
                           learning_rate=learning_rate, dropout_rate=dropout_rate,
                           # Additional model details and training params:
                           **extra_params
                           )
    model.to( device )

    return dataset_split, model

def get_datasets_and_loaders(genome_file, sample_file, signal_file, embed_file,
                             batch_size, cpu, n_regions=None):
    """ Creating the datasets and data loaders split by train/test regions and train/test experiments.
    """
    dataset_split = {}
    for region_train in [True, False]:
        for exper_train in [True, False]:

            # Creating the dataset with the virtual splitting to train/test regions and experiments
            split_data = Precomputed(genome_file, sample_file, signal_file, embed_file,
                                       train_regions_only=region_train, test_regions_only=region_train==False,
                                       train_expers_only=exper_train, test_expers_only=exper_train==False,
                                       n_regions=n_regions
                                     )

            # Would only want to randomly shuffle, if it is the training data.
            shuffle_ = False
            if region_train and exper_train:
                shuffle_ = True

            # And the corresponding dataloader
            split_loader = DataLoader(split_data,
                                      batch_size=batch_size,
                                      num_workers=cpu, shuffle=shuffle_, persistent_workers=False
                                      )

            # encoded as train_region, train_expr
            dataset_split[(region_train, exper_train)] = (split_data, split_loader)

    return dataset_split

def test_datasets_and_loaders(genome_file, sample_file, dataset_split, batch_size):
    """ Making sure the datasets and loaders are given expected outputs.
    """
    genome_rois = pd.read_csv(f'{genome_file}', sep='\t', header=None)
    n_train_regions, n_test_regions = len(np.where(genome_rois.values[:, 2] == 'train')[0]), len(
        np.where(genome_rois.values[:, 2] == 'test')[0])

    sample_df = pd.read_csv(sample_file, sep='\t', index_col=0)
    n_train_expers, n_test_expers = len(np.where(sample_df['allocation'].values == 'train')[0]), len(
        np.where(sample_df['allocation'].values == 'test')[0])
    for split_name, (split_data, split_loader) in dataset_split.items():

        n_regions_ = n_train_regions if split_name[0] else n_test_regions
        n_expers_ = n_train_expers if split_name[1] else n_test_expers

        correct_regions = len(split_data.signal_data.load_region_indices) == \
                          len(split_data.embed_data.row_indices) == n_regions_
        correct_expers = len(split_data.signal_data.load_celltype_assays) == n_expers_

        if not correct_regions or not correct_expers:
            raise Exception("Incorrect number of regions or experiments for data-split detected.")

        for _ in split_loader:
            correct_region_load = _[1].shape[0] == batch_size and np.all([input_.shape[0] == batch_size
                                                                          for input_ in _[0].values()])
            correct_exper_load = _[1].shape[1] == n_expers_
            if not correct_region_load or not correct_exper_load:
                raise Exception("Incorrect number of regions or experiments detected from dataloader.")
            break






