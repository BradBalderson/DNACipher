""" Code for running embedding generation!
"""

from . import md

from .embedding_generator import EmbeddingGenerator

from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

import time

import torch
import pandas as pd

import yaml

from ..helpers import data_manage

def generate_embeddings(run_name, genome_file, fasta_file, query_embed_resolution,
                        model_name, device, batch_size, out_dir,
                        model_config_yaml=None
                        ):
    """ Generates embeddings.
    """

    #### Output log file
    log_file = open(f'{out_dir}{run_name}_log_file.txt', 'w')
    print( f'Starting precomputation of embeds {run_name} with run parameters:\n', file=log_file, flush=True)
    print( "\n".join([run_name, genome_file, model_name, f"batch_size:{batch_size}",
                               f"device:{device}", f"query_embed_resolution:{query_embed_resolution}"]),
                                                                                              file=log_file, flush=True)

    script_start_time = time.time()

    #### Loading the query regions.
    genome_rois = pd.read_csv(f'{genome_file}', sep='\t', header=None)

    #### Creating the EmbeddingModel.
    if model_name not in md.model_names:
        raise Exception(f"Inputted {model_name} not currently supported, must be one of {md.model_names}")

    #### Reading any extra inputs to the model setup...
    if type(model_config_yaml) == type(None):
        model_config = {}
    else:
        model_config = yaml.load( open(model_config_yaml, 'r'), yaml.Loader )

    #### Setting up the embedding model !!
    print( f'Loading the {model_name} model...\n', file=log_file, flush=True)
    embed_model = md.model_classes[ md.model_names.index(model_name) ](device, **model_config)

    print(f'Model is on device: {embed_model.model.device}\n', file=log_file, flush=True)
    print(f"Model details:", str(embed_model), file=log_file, flush=True)

    #### Now setting up the embedding generator
    print(f'Setting up the EmbeddingGenerator...\n', file=log_file, flush=True)
    embed_generator = EmbeddingGenerator(embed_model, genome_rois.values, fasta_file, query_embed_resolution)
    print(f'Finished EmbeddingGenerator setup.\n', file=log_file, flush=True)

    # Using a BatchSampler for the embed_generator is important, so that we parse multiple indices at once, because
    # it works by pulling out parts of embeddings outputted from a forward parse of a single tiled seuqence from the
    # genome. If we use the standard dataloader approach, it will query one index at a time and use the collate_function
    # to bring these data together to form the batch. But in our case the Dataset itself is designed to do this.
    batch_sampler = BatchSampler(SequentialSampler(embed_generator), batch_size=batch_size, drop_last=False)

    loader = DataLoader(embed_generator, sampler=batch_sampler,
                        num_workers=0, # has to be run worker since we are querying a model on GPU.
                        collate_fn=data_manage.batch_collate_fn
                        )

    #### Writing the epigenetic values to h5 output:
    data_manage.write_dataloader_to_h5(loader, f'{out_dir}{run_name}_{model_name}_embeds.h5',
                                       log_file, embed_model.embed_features, 'enformer_embeds'
                                       )

    ###################################################################################################
                                        # Finishing UP #
    ###################################################################################################
    print("DONE.", file=log_file, flush=True )

    script_end_time = time.time()

    total_minutes = round((script_end_time-script_start_time)/60, 3)
    total_hours = round((script_end_time-script_start_time)/60/60, 3)

    print("TOTAL minutes: ", total_minutes, file=log_file, flush=True )
    print("TOTAL hours: ", total_hours, file=log_file, flush=True )

    log_file.close()
