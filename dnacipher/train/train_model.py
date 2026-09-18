""" Main functions for training a new DNACipher model.
"""

import os
import subprocess
import time

import numpy as np

import torch

import pytorch_lightning as pl

from dnacipher_train.train.callbacks import get_callbacks
import dnacipher_train.train.train_eval as train_eval

import dnacipher_train.train.helpers as helpers
import dnacipher_train.helpers.utils as utils

def train_dnacipher_model(run_name, genome_file, sample_file, signal_file, embed_file, out_dir,
                          model_config_file,
                          # train parameters, need to change depending on resources.
                          batch_size, cpu, device, epochs_warmup, epochs_final, eval_batches,
                          learning_rate=0.0014526, dropout_rate=0.05, n_regions=None,
                          ):
    """ Main function for training the DNACipher model.
    """

    #### Output log file
    log_file = open(f'{out_dir}{run_name}_log_file.txt', 'w')

    print(f'DNACipher training run: {run_name}\n', file=log_file, flush=True)

    print(f'With inputs: \ngenome_file={genome_file}\nsample_file={sample_file}\nsignal_file={signal_file}\n'
          f'embed_file={embed_file}\nout_dir={out_dir}\n'#train_config_file={train_config_file}\n'
          f'model_config_file={model_config_file}\n',
          file=log_file, flush=True)

    script_start_time = time.time()

    dataset_split, model = helpers.model_and_dataset_setup(genome_file, sample_file, signal_file, embed_file,
                                                           batch_size, cpu, device, model_config_file, log_file,
                                                           learning_rate=learning_rate, dropout_rate=dropout_rate,
                                                           n_regions=n_regions
                                                           )

    print("\nDNACipher model successfully initialised, with details:\n", model, '\n', file=log_file, flush=True )

    if epochs_warmup > 0:
        #### Setting up the model trainer
        temp_dir, callbacks = get_callbacks(run_name, out_dir)
        trainer = pl.Trainer(max_epochs=epochs_warmup, callbacks=callbacks, devices=1,
                             logger=False,  # logger disabled
                             enable_checkpointing=True,
                             accelerator='cpu' if device=='cpu' else 'gpu',
                             num_sanity_val_steps=0,
                             log_every_n_steps=1,
                             gradient_clip_val=None,
                             )

        print("\nWarmup training commencing..\n", file=log_file, flush=True)
        trainer.fit(model,
                    dataset_split[(True, True)][1], # Fit to train regions/experiments
                    dataset_split[(False, False)][1] # Eval on test regions/experiments
                    )

        #### Logging some basic evals on the overall performance throughout the warmup
        print("\nVisualizing warmup losses throughout epochs..\n", file=log_file, flush=True)
        train_eval.visualize_losses(run_name, "WARMUP", out_dir, callbacks[0], log_file)

        # Reload the weights of the best performance seen during the warmup...
        print("\nReloading the best performing model seen during warmup...\n", file=log_file, flush=True)
        best_warmup_weights = load_best_weights(temp_dir, run_name, device, log_file, clear_temp=True)

        # Load the weights to the model
        model.load_state_dict( best_warmup_weights )

    #### Now the final training, using the stratified loss and relu on the model output:
    if epochs_final > 0:
        if not model.softplus_output:
            model.relu_output = True

        # Always switch to stratified loss.
        model.stratified_loss = True
        temp_dir, callbacks = get_callbacks(run_name, out_dir)
        trainer = pl.Trainer(max_epochs=epochs_final, callbacks=callbacks, devices=1,
                             logger=False,  # logger disabled
                             enable_checkpointing=True,
                             accelerator='cpu' if device=='cpu' else 'gpu',
                             num_sanity_val_steps=0,
                             log_every_n_steps=1,
                             gradient_clip_val=None,
                             )

        print("\nFinal training commencing..\n", file=log_file, flush=True)
        trainer.fit(model,
                    dataset_split[(True, True)][1], # Fit to train regions/experiments
                    dataset_split[(False, False)][1] # Eval on test regions/experiments
                    )

        #### Logging some basic evals on the overall performance throughout the final training
        print("\nVisualizing final losses throughout epochs..\n", file=log_file, flush=True)
        train_eval.visualize_losses(run_name, "FINAL", out_dir, callbacks[0], log_file)

        # Load the best weights seen during the final training.
        print("\nReloading the best performing model seen during final training...\n", file=log_file, flush=True)
        best_final_weights = load_best_weights(temp_dir, run_name, device, log_file, clear_temp=True)

        # Load the weights to the model
        model.load_state_dict( best_final_weights )

    print("\nSaving these model weights: \n", file=log_file, flush=True)
    torch.save(model.state_dict(), out_dir + f'{run_name}_model_weights.pth')

    ################### More extensive evaluations for the final model:
    # Now performing the key evaluations:
    train_eval.eval_model(run_name, model, dataset_split, log_file, out_dir, eval_batches, batch_size)

    utils.finalize_log(log_file, script_start_time)

    return

def load_best_weights(temp_dir, run_name, device, log_file, clear_temp=True):
    """ Loads the best performing checkpointed model from the temp directory.
    """
    saved_models = [file_ for file_ in os.listdir(temp_dir)
                    if file_.startswith(run_name + '--epoch') and file_.endswith('.ckpt')
                    ]
    model_val_loss = [float(saved_model.split('val_loss=')[1].strip('.ckpt')) for saved_model in saved_models]
    best_model_file = saved_models[np.argmin(model_val_loss)]

    print(f'{temp_dir}{best_model_file}', file=log_file, flush=True)

    # Reloading the better model
    weights = torch.load(f'{temp_dir}{best_model_file}', map_location=torch.device( device ))['state_dict']

    if clear_temp:
        #subprocess.run(['rm', '-r',  f'{temp_dir}{run_name}*'])
        os.system(f"rm -r {temp_dir}{run_name}*")

    return weights


