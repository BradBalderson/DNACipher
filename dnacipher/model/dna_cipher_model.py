""" Per-sequence model; does not consider position in the genome of the sequence. Written to work with sequence embeddings
    as input, so can hypothetically train with different sequence-only DL models.

    In original development code, this is called 'dna_cipher_v4'.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class DNACipherModel( pl.LightningModule ):
    """
    """

    def __init__(self, n_celltypes, n_celltype_factors,  # Cell type information
                 n_assays, n_assay_factors,  # Assay type information
                 n_seq_features_input, n_seq_factors,  # Genomic sequence information
                 n_layers, n_nodes, # Deep layer information
                 # Additional neural network parameters.
                 relu_output=False, layer_norm=True, activation_function='gelu',
                 celltype_assay_weights=None, celltype_assay_weight_test=None,
                 # Training parameters
                 learning_rate=0.0014526, lr_scheduler=False,
                 dropout_rate=0.05, out_transform='preprint', stratified_loss=False,
                 # attention related inputs
                 attention=False, n_tokens=16, n_heads=4
                 ):
        """ DNACipher encodes DNA sequence information along with celltype, assay information, to impute
            the effects of variants on experiments that have never been performed.

        Args:

        """
        super().__init__()

        #### Defining some params which change how the network works...
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.stratified_loss = stratified_loss
        if type(celltype_assay_weights) != type(None):
            self.celltype_assay_weights = torch.tensor(celltype_assay_weights,
                                                       dtype=torch.float32)  # used to weight the different celltype,assay during training.
        else:
            self.celltype_assay_weights = None

        if type(celltype_assay_weight_test) != type(None):
            self.celltype_assay_weight_test = torch.tensor(celltype_assay_weight_test, dtype=torch.float32)
        else:
            self.celltype_assay_weights = None

        ##### Defining the embedding input layers
        self.celltype_embedding = nn.Embedding(n_celltypes, n_celltype_factors)
        self.assay_embedding = nn.Embedding(n_assays, n_assay_factors)

        ##### Defining the different layers of genomic input, which will be the features across a single token!
        # For the genome layer, will project the token features down to n_genomic_factors
        self.n_seq_features_input = n_seq_features_input
        self.n_seq_factors = n_seq_factors
        self.genome_layer = nn.Linear(int(self.n_seq_features_input), self.n_seq_factors)
        ##### Defining the concatenation layer
        factor_len = n_celltype_factors + n_assay_factors + self.n_seq_factors
        concat_layer = nn.Linear(factor_len, n_nodes)
        if layer_norm:
            concat_layer = nn.Sequential(concat_layer, nn.LayerNorm(n_nodes, elementwise_affine=False))

        ##### Defining the network
        self.dense_layers = nn.ModuleList([concat_layer])
        for i in range(n_layers - 1):
            layeri = nn.Linear(n_nodes, n_nodes)
            if layer_norm:
                layeri = nn.Sequential(layeri, nn.LayerNorm(n_nodes, elementwise_affine=False))
            self.dense_layers.append(layeri)

        # Optionally adding an attention mechanism:
        self.attention = attention
        if attention:
            # Adding an attention layer before the output linear layer !!!
            attn = FeatureUpdateAttention(n_nodes, n_tokens, n_heads)
            self.dense_layers.append( attn )

        # NOTE this can be overrided below.
        self.output_layer = nn.Linear(n_nodes, 1)

        ##### Defining the dropout function
        self.drop = nn.Dropout(p=dropout_rate)

        ##### Defining the loss function.
        self.out_transform = out_transform
        self.softplus_output = False
        self.relu_output = relu_output
        self.stratified_loss = False # Always default to false, need to set externally if switching loss.
        if self.out_transform=='softplus':
            self.relu_output = False # Can't have this...
            self.softplus_output = True # DO this instead.

        elif self.out_transform=='dual_loss': # BCE + MSE
            raise Exception("TODO implement and try.")
            self.relu_output = False

            # TODO continue working from here!!! Mainly:
            #  * need proportion of zeros to non-zeros, in order to weight the BCEWithLogitsLoss correctly.
            #  * need to handle the output correctly when the model is put in eval mode, namely output 0 when prob is below
            #    some cutoff [need to determine the best cutoff, maybe by calibrating on the training data during eval..]
            #  * but also need to handle case where model is put into eval mode during the validation loss calculation..
            #  *

            self.output_layer = nn.Linear(n_nodes, 2) # logit of non-zero probability, signal magnitude.

        ##### Defining the activation function.
        if activation_function == 'gelu':
            self.activation_function = F.gelu
        elif activation_function == 'relu':
            self.activation_function = F.relu
        else:
            raise Exception(f"Activation function specified not supported: {activation_function}, "
                            f"must be 'gelu' or 'relu'")

    def to(self, device):
        super().to( device )

        # These weighs are not explicitly connected to the NN, so the .to command will not put them on the right device,
        # so doing it here.
        if 'celltype_assay_weights' in vars(self) and type(self.celltype_assay_weights) != type(None):
            self.celltype_assay_weights = self.celltype_assay_weights.to( device )  # Need to make sure on right device..
            self.celltype_assay_weight_test = self.celltype_assay_weight_test.to( device )

    def forward(self, celltype_input, # seq_pos * experiments (cell type indices specified)
                assay_input, # seq_pos * experiments (assay indices specified)
                genome_input, # seq_pos * seq_features
                embed_layer=None # If you want to retrieve the latent model embedding at an internal layer
                ):
        """ This assumes are providing input that refers to a FULL sequence/s.
            So need to reshape the input, so that it refers to PER token, and then reshapes back to per sequence info.
        """

        if len(genome_input.shape) == 1:  # Accounting for 1 input, so adding an extra dimension to have 1 row!
            genome_input = genome_input.unsqueeze(0)

        nseqs = genome_input.shape[0]  # Need to store this to reshape BACK to per sequence information!
        ncelltype_input = celltype_input.shape[1] # meaning number of experiments
        total_input = nseqs * ncelltype_input

        ##### Accounting for DENSE format input, where we have multiple celltype,assays being imputed for a given region
        #### Is a shallow copy, so can run matrix multipication without the extra memory cost !!!
        # Strategy here repeats each of the sequence region values (as shallow script) ncelltype_input times in a row
        # before then repeating the next set of sequence features, and so on.
        # Essentially now getting matrices with shape experiments * seq_positions as input, and will re-shape later
        genome_input_ = genome_input.unsqueeze(1).expand(-1, ncelltype_input, -1).reshape(-1, genome_input.size(1))
        if genome_input_.shape[0] != total_input:
            raise Exception("Shape problem!")

        celltype_input_ = celltype_input.ravel()  # Flattens this to point observations.
        assay_input_ = assay_input.ravel()

        return self.forward_flattened(celltype_input_, assay_input_, genome_input_, nseqs, embed_layer)

    def forward_flattened(self,
                celltype_input, # experiments (cell type indices specified)
                assay_input, # experiments (assay indices specified)
                genome_input, # experiments * seq_features
                nseqs, # original number of sequence embeddings input, necessary to reshape the output
                embed_layer):

        #### Embeddings for the cell type and assay information
        celltype = self.celltype_embedding( celltype_input )
        assay = self.assay_embedding( assay_input )
        # TODO should investigate whether should have applied dropout / activation to these inputs..

        # Project the token features down.
        genome_input = genome_input.to( celltype.dtype )
        genome = self.genome_layer( genome_input )
        genome = self.drop(genome) # TODO NOTE dropped and activated this, but not the above..
        genome = self.activation_function( genome )

        # Now creating the set of features which will go through the deeper layers to predict epigenetic signal
        if len(celltype_input.shape) == 0:  # Accounting for a single value provided as input !
            feature_tensors = [celltype.unsqueeze(0), assay.unsqueeze(0), genome]
        else:
            feature_tensors = [celltype, assay, genome]

        # Pushing the feature tensors through deeper layers
        x = torch.cat(feature_tensors, dim=1)
        for i, layer in enumerate(self.dense_layers):
            x = self.drop(layer(x))
            x = self.activation_function(x)

            #### Want to return an embedding layer!
            if type(embed_layer) != type(None) and i == embed_layer:
                break

        # Predicting epigenetic signal
        if type(embed_layer) == type(None):
            y = self.output_layer( x ).squeeze()

            y = y.view(nseqs, y.shape[0] // nseqs)

            if self.relu_output:
                y = F.relu(y)

            elif self.softplus_output:
                y = F.softplus(y)

            if torch.any(y.isnan()).item():
                print("NAN in output")

        ##### Returning the embedding features !!!
        else:
            y = x.squeeze()

        return y

    def get_genomic_embedding(self, genome_input):
        """ Assumes have full token features information from an input sequence/s
        """
        if len(genome_input.shape) == 1:  # Accounting for 1 input, so adding an extra dimension to have 1 row!
            genome_input = genome_input.unsqueeze(0)

        genome = self.genome_layer( genome_input )

        return genome

    def dual_loss(self, seq_values_pred, track_values, weights=None):
        """Considers both the probability of non-zero signal and also the signal value separately.
        """
        pass
        # lambda_zero = 0.01  # to try and account for case where the binary head is correct...
        #
        # loss_reg = (
        #         lambda_nonzero * mse(y_hat[y > 0], y[y > 0]) +
        #         lambda_zero * mse(y_hat[y == 0], 0)
        # )

    def loss_function(self, inputs_, seq_values_pred, track_values):
        """ Loss function under different conditions, since will have a warmup / final training, and also weighted/unweighted.
        """

        general_loss_func = nn.MSELoss()
        if not self.stratified_loss:
            loss = general_loss_func(seq_values_pred, track_values)

        else:
            if type(self.celltype_assay_weights) == type(None) or \
                    type(self.celltype_assay_weight_test) == type(None):
                raise Exception("Stratified loss specified for DNACipher model, but no weights specified in model to "
                                "weight the experiments contribution to the loss.")

            nseqs = inputs_[0].shape[0]
            nexpers = inputs_[0].shape[1]
            if nexpers == self.celltype_assay_weights.shape[0]:
                weights = self.celltype_assay_weights
            else: # Test experiments..
                weights = self.celltype_assay_weight_test

            seq_values_pred_by_region = seq_values_pred.view(nseqs, nexpers)
            seq_values_by_region = track_values.view(nseqs, nexpers)

            errors_stratified = torch.pow((seq_values_pred_by_region - seq_values_by_region), 2)
            errors = torch.multiply(errors_stratified, weights * 100).sum(axis=0)
            weighted_errors = errors

            loss = weighted_errors.mean() # errors are now balanced according to the weighting scheme!

        return loss

    def training_step(self, batch, batch_idx, train_log=True):
        input_kwargs, track_values = batch
        inputs_ = [input_kwargs[key] for key in input_kwargs if key.endswith('_input')]
        seq_values_pred = self(input_kwargs['celltype_input'],
                                 input_kwargs['assay_input'],
                                 input_kwargs['seqfeatures_input'],
                                 )

        loss = self.loss_function(inputs_, seq_values_pred, track_values)

        if train_log:
            self.log('train_loss', loss)

        param_grads = next(self.output_layer.parameters()).grad
        if type(param_grads)!=type(None):
            max_grad = param_grads.max()
            min_grad = param_grads.min()
            print('Original grads:', max_grad, min_grad)
            print(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train_log=False)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if not self.lr_scheduler:
            return optimizer

        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.9,
                patience=2,
                threshold=0.0001,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # This must match the name used in self.log()
                    "frequency": 1  # How often to check (1 = every epoch)
                },
            }

class FeatureUpdateAttention(torch.nn.Module):
    def __init__(self, n_nodes=512, num_tokens=16, num_heads=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = n_nodes // num_tokens

        # Multi-head attention
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = torch.nn.LayerNorm(self.token_dim)

    def forward(self, x):
        # x: [batch, 512]
        batch_size = x.shape[0]

        # 1. Reshape to "Sequence" form: [batch, 16, 32]
        x = x.view(batch_size, self.num_tokens, self.token_dim)

        # 2. Self-Attention (Updating values based on context)
        # Each of the 16 tokens looks at the other 15
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + attn_out)  # Residual connection

        # 3. Flatten back: [batch, 512]
        return x.reshape(batch_size, -1)




