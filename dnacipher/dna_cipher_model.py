""" Per-sequence model; does not consider position in the genome of the sequence. Written to work with sequence embeddings
    as input, so can hypothetically train with different sequence-only DL models.

    In original development code, this is called 'dna_cipher_v4'.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        avg_train_loss = trainer.callback_metrics.get('train_loss')
        if avg_train_loss is not None:
            self.train_losses.append(avg_train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_val_loss = trainer.callback_metrics.get('val_loss')
        if avg_val_loss is not None:
            avg_val_loss = avg_val_loss.item()
            self.val_losses.append(avg_val_loss)


class DNACipherModel( pl.LightningModule ):
    """
    """

    def __init__(self, n_celltypes, n_celltype_factors,  # Cell type information
                 n_assays, n_assay_factors,  # Assay type information
                 n_genomic_inputs,  # REDUNDANT IN THIS VERSION
                 n_genomic_factors,  # Genomic sequence information
                 n_output_factors,  # REDUNDANT IN THIS VERSION
                 n_layers, n_nodes, dropout_rate=0.1,
                 loss_function=None,
                 freeze_celltypes=False, freeze_assays=False,
                 freeze_genome=False, freeze_network=False,
                 dropout_genome_layer=True, relu_genome_layer=True,
                 learning_rate=1e-5, layer_norm=False,
                 n_token_features=1280,
                 n_token_factors=1,  # REDUNDANT IN THIS VERSION
                 activation_function=None,
                 n_tokens=999,  # Number of tokens, important for using the train function.
                 relu_output=False,  # whether to apply Relu to the output. Suspect will improve performance.
                 triplet_loss=False, # Whether to use triplet-loss contrastive learning!
                 epi_summarise_method='top_mean', #Method to use for summarising the epi-measure per sequence.
                 epi_top_k=100, #Only relevant if epi_summarise_method='top_mean', specifies number of top signals.
                 epi_signal_cutoff=6,  #Only relevant if epi_summarise_method='significant_fraction', specifies cutoff
                                      # above which is considered a significant. If untransformed signal values, default
                                      # refers to 1e-6 p-value cutoff
                 stratified_loss=False, celltype_assay_weights=None, celltype_assay_weight_test=None,
                 huber_delta=None,
                 ):
        """ DNACipher encodes DNA sequence information along with celltype, assay information, to impute
            the effects of variants on experiments that have never been performed.

            In this version, instead of purely treating the features associated with each token as flat input,
            will create a separate layer for each token, and summarise the features down to 1 feature within each of
            these, and THEN flatten. HOPE is to reduce the model size, to make the training better!

        Args:
            n_celltypes:
            n_celltype_factors:
            n_assays:
            n_assay_factors:
            n_genomic_inputs:
            n_genomic_factors:
            n_output_factors:
            n_layers:
            n_nodes:
            dropout_rate:
            loss_function:
            freeze_celltypes:
            freeze_assays:
            freeze_genome:
            freeze_network:
            dropout_genome_layer:
            relu_genome_layer:
            learning_rate:
            layer_norm:
            n_token_features: The number of features associated with each token provided by n_genomic_inputs.
                               is used to 'cut-up' the flattened array back to the original tokens. Such that
                               the inputs for the layer of the first token will be latent_space[:n_token_features].
            n_token_factors: The number of features to project the latent factors down to for each token. i.e. will take
                            n_token_features for each token and project down to n_token_factors (default 1280 -> 1).
                            So for a 6Kb input, will project down to 1,000 features for each position.
        """
        #super(DNACipherModel, self).__init__()
        super().__init__()

        #### Defining some params which change how the network works...
        self.dropout_genome_layer = dropout_genome_layer
        self.relu_genome_layer = relu_genome_layer
        self.learning_rate = learning_rate
        self.triplet_loss = triplet_loss
        self.huber_delta = huber_delta
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

        epi_summarise_methods = ['top_mean', 'significant_fraction', 'mean', 'flatten']
        if epi_summarise_method not in epi_summarise_methods:
            raise Exception(f"Got epi_summarise_method input {epi_summarise_method}, "
                            f"but only support {epi_summarise_methods}")

        self.epi_summarise_method = epi_summarise_method
        self.epi_top_k = epi_top_k
        self.epi_signal_cutoff = epi_signal_cutoff

        ##### Defining the embedding input layers
        self.celltype_embedding = nn.Embedding(n_celltypes, n_celltype_factors)
        self.celltype_embedding.requires_grad = freeze_celltypes == False

        self.assay_embedding = nn.Embedding(n_assays, n_assay_factors)
        self.assay_embedding.requires_grad = freeze_assays == False

        ##### Defining the different layers of genomic input, which will be the features across a single token!
        # For the genome layer, will project the token features down to n_genomic_factors
        self.n_tokens = n_tokens
        self.n_token_features = n_token_features
        self.genome_layer = nn.Linear(int(self.n_token_features), n_genomic_factors)
        self.genome_layer.requires_grad = freeze_genome == False

        ##### Defining the concatenation layer
        factor_len = n_celltype_factors + n_assay_factors + n_genomic_factors
        concat_layer = nn.Linear(factor_len, n_nodes)
        concat_layer.requires_grad = freeze_network == False
        if layer_norm:
            concat_layer = nn.Sequential(concat_layer, nn.LayerNorm(n_nodes, elementwise_affine=False))

        ##### Defining the network
        self.dense_layers = nn.ModuleList([concat_layer])
        for i in range(n_layers - 1):
            layeri = nn.Linear(n_nodes, n_nodes)
            layeri.requires_grad = freeze_network == False
            if layer_norm:
                layeri = nn.Sequential(layeri, nn.LayerNorm(n_nodes, elementwise_affine=False))
            self.dense_layers.append(layeri)

        # Predict average epigenetic signal at the single token.
        self.output_layer = nn.Linear(n_nodes, 1)
        self.output_layer.requires_grad = freeze_network == False

        ##### Defining the dropout function
        self.drop = nn.Dropout(p=dropout_rate)

        ##### Defining the loss function.
        if type(loss_function) == type(None):
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = loss_function

        self.relu_output = relu_output

        ##### Defining the activation function.
        if type(activation_function) == type(None) or activation_function == 'gelu':
            self.activation_function = F.gelu
        elif activation_function == 'relu':
            self.activation_function = F.relu

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

        # Project the token features down.
        genome_input = genome_input.to( celltype.dtype )
        genome = self.genome_layer( genome_input )
        if self.dropout_genome_layer:  ### These are optional, since didn't implement in original version.
            genome = self.drop(genome)
        if self.relu_genome_layer:
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

        nseqs = genome_input.shape[0]  # Need to store this to reshape BACK to per sequence information!

        #### NEED to reformat the input to refer to each sequence, by averaging the features across tokens.
        if genome_input.shape[1] != self.n_token_features:
            seqfeatures_by_token = genome_input.view(genome_input.shape[0], self.n_tokens, self.n_token_features)
            seqfeatures = seqfeatures_by_token.mean(axis=1)  # average features across tokens
        else:
            seqfeatures = genome_input

        genome = self.genome_layer( seqfeatures )
        return genome

    #@staticmethod
    def summarise_epi_values(self, track_values ):
        """ Summarise the epivalues across a region down to a single measure.
        """
        # The values have already been summarised, as evidenced by one value per-seq, so just return.
        if len(track_values.shape)==1:
            return track_values
        elif track_values.shape[1]==1:
            return track_values[:,0]

        # Need to summarise down to a single epigenetic measure per sequence.
        seq_values = None
        if self.epi_summarise_method == 'top_mean': # Average of top 100 signals across seqs.
            topk_values, _ = torch.topk(track_values, 100, dim=1)
            seq_values = torch.mean(topk_values, dim=1)

        elif self.epi_summarise_method == 'significant_fraction': # Proportion of signal above significance threshold...
            seq_values = (track_values > self.epi_signal_cutoff).sum(axis=1) / track_values.shape[1]

        elif self.epi_summarise_method == 'mean':
            seq_values = track_values.mean(axis=1)

        ##### This is relevant for DENSE training, since we will flatten the predictions for each celltype, assay
        ##### regardless of region....
        elif self.epi_summarise_method == 'flatten':
            seq_values = track_values.ravel()

        return seq_values

    @staticmethod
    def continuous_contrastive_loss(y_pred, y_true, margin=1.0):
        """ Contrastive loss
        """
        # Calculate pairwise differences
        diff_pred = torch.pdist(y_pred, p=1).pow(2)
        diff_true = torch.pdist(y_true, p=1).pow(2)

        # Calculate fold-change loss: (log(y_pred / y_true))^2
        fold_change_loss = torch.log(diff_pred / (diff_true + 1e-8) + 1e-8).pow(2)

        # Apply margin to ensure positive fold-changes are penalized more
        contrastive_loss = torch.mean(torch.max(fold_change_loss - margin, torch.zeros_like(fold_change_loss)))

        return contrastive_loss

    @staticmethod
    def get_triplets(batch_size):
        """ Generates triplet lists for fast indexing...
        """
        anchor_index = 0

        triplets = []
        for i in range(1, batch_size):
            for j in range(i+1, batch_size):
                triplets.append( [anchor_index, i, j] )

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

    @staticmethod
    def triplet_contrastive_loss(y_pred, y_true, margin=1):
        triplets = DNACipherModel.get_triplets(y_pred.shape[0])

        triplets = triplets.to(y_pred.device.type)

        ai_distances = (y_pred[triplets[:, 0]] - y_pred[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        aj_distances = (y_pred[triplets[:, 0]] - y_pred[triplets[:, 2]]).pow(2).sum(1)
        keep_indices = torch.tensor(list(range(len(ai_distances))))
        #keep_indices = torch.where(ai_target_distances < aj_target_distances)[0]
        #keep_indices = torch.where((aj_distances > ai_distances) & (aj_distances < ai_distances + margin))[0]
        log_ratio = torch.log( ai_distances[keep_indices] / aj_distances[keep_indices] )

        # Trying case where we subset to the numerator is small than the denominator...
        ai_target_distances = (y_true[triplets[:, 0]] - y_true[triplets[:, 1]]).pow(2).sum(1)
        aj_target_distances = (y_true[triplets[:, 0]] - y_true[triplets[:, 2]]).pow(2).sum(1)

        target_log_ratio = torch.log(ai_target_distances[keep_indices] / aj_target_distances[keep_indices])

        # losses = F.relu(ap_distances - an_distances + self.margin)
        losses = (log_ratio - target_log_ratio).pow(2)
        losses = losses[losses.isinf()==False] # Removing 0 numerator examples..
        loss = losses.quantile(.5) # Median more robust to outliers
        if len(losses)==0 or torch.any(y_true[:,0].isnan()).item() or torch.any(y_pred[:,0].isnan()).item()\
                or loss.isnan().item():
            raise Exception(f"ERROR \nlosses: {losses}\n y_true {y_true}\n y_pred {y_pred} \n loss {loss}")

        return loss

    def training_step(self, batch, batch_idx, train_log=True):
        input_kwargs, track_values = batch
        inputs_ = [input_kwargs[key] for key in input_kwargs if key.endswith('_input')]
        seq_values_pred = self(input_kwargs['celltype_input'],
                                 input_kwargs['assay_input'],
                                 input_kwargs['seqfeatures_input'],
                                 )
        #### Here we have to summarise the values to prevent changing the DataLoader.
        # Getting average of top 100 values across the sequence...
        seq_values = self.summarise_epi_values( track_values ).unsqueeze(1)

        if not self.triplet_loss:
            if not self.stratified_loss:
                loss = self.loss_function(seq_values_pred, seq_values)

            elif type(self.celltype_assay_weights)!=type(None): # Will weight the MSEs across regions...
                nseqs = inputs_[0].shape[0]
                nexpers = inputs_[0].shape[1]
                if nexpers == self.celltype_assay_weights.shape[0]:
                    weights = self.celltype_assay_weights
                else:
                    weights = self.celltype_assay_weight_test

                seq_values_pred_by_region = seq_values_pred.view(nseqs, nexpers)
                seq_values_by_region = seq_values.view(nseqs, nexpers)

                # Standardising the errors..
                # Attempt one, getting some performance benefit, but when I look at the values still think the loss is
                # dominated by extreme values, particular for experiments with a larger dynamic range.

                # WEIGHTED-V2 is the name given to the jobs using this below strategy on Wiener!!!
                if type(self.huber_delta)==type(None):
                    errors_stratified = torch.pow((seq_values_pred_by_region - seq_values_by_region), 2)
                    errors = torch.multiply(errors_stratified, weights*100).sum(axis=0)
                    weighted_errors = errors

                # Will try a different strategy, using the 'Huber loss', that will dull the effect of extreme values!
                # errors_stratified_se = torch.pow((seq_values_pred_by_region - seq_values_by_region), 2)
                # y = (errors_stratified_se).sum(axis=0).detach().numpy()
                # y1 = torch.multiply(errors_stratified_se, weights*100).sum(axis=0).detach().numpy()
                if type(self.huber_delta)!=type(None):
                    errors_stratified = F.huber_loss(seq_values_pred_by_region, seq_values_by_region,
                                                     delta=self.huber_delta, reduction='none')
                    #y2 = (errors_stratified).sum(axis=0).detach().numpy()
                    errors = torch.multiply(errors_stratified, weights*1000).sum(axis=0)
                    #y3 = errors.detach().numpy()
                    weighted_errors = errors

                # x = weights.numpy()
                # plt.scatter(x,y)
                # plt.show()
                # plt.scatter(x,y1)
                # plt.show()
                # plt.scatter(x, y2)
                # plt.show()
                # plt.scatter(x, y3)
                # plt.show()

                # Now weighting these errors:

                loss = weighted_errors.mean() # errors are now balanced according to the weighting scheme!

            else:
                ### Implementing a different loss, which stratifies the MSE by celltype, assay, and takes the mean of this!
                ct_labels = inputs_[0].ravel()
                assay_labels = inputs_[1].ravel()
                cts = inputs_[0][0, :].unique()
                assays = inputs_[1][0, :].unique()

                stratified_mses = []
                n_max = max([len(cts), len(assays)])
                for i in range(n_max):
                    if i < len(cts):
                        ct_bool = ct_labels == cts[i]
                        ct_loss = self.loss_function(seq_values_pred[ct_bool], seq_values[ct_bool])
                        stratified_mses.append(ct_loss)

                    if i < len(assays):
                        assay_bool = assay_labels == assays[i]
                        assay_loss = self.loss_function(seq_values_pred[assay_bool], seq_values[assay_bool])
                        stratified_mses.append(assay_loss)

                pooled_mses = torch.stack( stratified_mses )
                loss = torch.mean(pooled_mses) # Average of the MSE when stratified by celltypes and assays

        else:
            loss = self.triplet_contrastive_loss(seq_values_pred, seq_values)

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                )






