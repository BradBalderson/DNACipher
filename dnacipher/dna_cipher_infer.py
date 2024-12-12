"""
Inference interface for DNACipher, implements querying the model for anticipated common tasks, example inferring
the effect of a genetic variant.
"""

import math
import numpy as np
import pandas as pd

import torch

import inspect # Use this to figure out how many positional arguments are required by the model forward pass!

from itertools import product

# For loading sequences
import pyfaidx
import kipoiseq
from kipoiseq import Interval

from scipy.spatial import cKDTree

#Sequence feature extraction via Nucleotide Transformer
import re

from .dna_cipher_model import DNACipherModel

class DNACipher():

    def __init__(self, weight_path, sample_file, genome_file, config=None,
                 transformer_model_name='500M_human_ref', transformer_layer=24, device='cpu',
                 seqlen=5994, embed_method='mean', kernel_size=50,
                 dnacv='dnacv4', # Version of DNACipher to use
                 ):
        """ Constructs a DNACipher object, which allows for inference of variant function....
        """

        # Base run params
        self.unsummed_methods = ['mean', 'max-diff', 'max-diff-token'] # Work on average embeddings.
        self.summed_methods = ['sum-abs', 'sum-abs-signed']
        self.allowed_methods = self.unsummed_methods + self.summed_methods
        if embed_method not in self.allowed_methods:
            raise Exception(f"Inputted embed_method {embed_method} is not one of {self.allowed_methods}")
        self.embed_method = embed_method
        self.kernel_size = kernel_size #For smoothing the embedding features!
        self.device = device
        self.transformer_model_name = transformer_model_name
        self.transformer_layer = transformer_layer

        # Important for loading sequence information.
        self.seq_extracter = FastaStringExtractor( genome_file )
        # if seqlen > 5994:
        #     raise Exception("seqlen cannot be over maximum of 5994.")

        # Length of sequence to be ingested by model, must not exceed the model max sequence length
        self.seqlen = seqlen

        ################################################################################################################
                                # Storing information about the samples that were trained on #
        ################################################################################################################
        sample_df = pd.read_csv(sample_file, sep='\t', index_col=0)

        self.celltype_assays = [tuple(celltype_assay.split('---')) for celltype_assay in
                           sample_df['celltype_assay'].values.astype(str)]
        self.celltype_assay_labels = sample_df['allocation'].values # Specifies which were used as train/test
        self.celltypes = list(np.unique([celltype_assay[0] for celltype_assay in self.celltype_assays]))
        self.assays = list(np.unique([celltype_assay[1] for celltype_assay in self.celltype_assays]))

        self.imputable_celltype_assays = []
        for celltype in self.celltypes:
            for assay in self.assays:
                self.imputable_celltype_assays.append( f"{celltype}---{assay}" )
        self.imputable_celltype_assays = np.array( self.imputable_celltype_assays )
        self.n_imputable_celltype_assays = len(self.celltypes)*len(self.assays)

        ################################################################################################################
                                             # Creating the DNACipher model #
        ################################################################################################################
        # Loading the model weights, which will then use to infer the model architecture.
        self.weight_path = weight_path
        model_weights = torch.load(self.weight_path, map_location=torch.device(self.device))

        ##### Determining model parameters...
        n_celltypes = model_weights['celltype_embedding.weight'].shape[0]
        n_celltype_factors = model_weights['celltype_embedding.weight'].shape[1]

        n_assays = model_weights['assay_embedding.weight'].shape[0]
        n_assay_factors = model_weights['assay_embedding.weight'].shape[1]

        n_genomic_inputs = model_weights['genome_layer.weight'].shape[1]
        n_genomic_factors = model_weights['genome_layer.weight'].shape[0]

        dense_layer_keys = [info for info in list(model_weights.keys()) if info.startswith('dense_layers')]
        n_layers = max(
            [int(info.split('.')[1]) for info in dense_layer_keys]) + 1
        self.n_layers = n_layers

        n_nodes = model_weights[dense_layer_keys[0]].shape[0]
        self.n_nodes = n_nodes

        n_output_factors = 1

        #### Loading the config... if not provide resort to default.
        if type(config) == type(None):
            # TODO should add a .config as output when training a DNACipher model, so can simply read this information
            #  from the config file when setting up a given model...
            #  ALSO without the config, is not clear what indexes represent the celltype, assays!
            ##### values below are why we need a config outputted as well as the model weights....
            config = {'dropout_genome_layer': False,  # Will be in evaluation mode the whole time anyhow.
                      'relu_genome_layer': True,  # This is something a config would be useful to get
                      'learning_rate': 0,  # N/A in eval mode.
                      'layer_norm': False,
                      'relu_output': True,
                      'activation_function': 'gelu',
                      'triplet_loss': False,
                      'epi_summarise_method': 'mean',
                      'epi_signal_cutoff': 0,
                      'dropout_rate': 0,
                      'n_token_features': 1280,
                      'n_token_factors': 1,  # REDUNDANT
                      }

        ##### Determining the number of input arguments required by the inputted model!
        signature = list(inspect.signature(DNACipherModel.__init__).parameters.keys())

        # Need to filter out stored options not used for creating the model! That's DNACipher V5...
        config = {key_: value_ for key_, value_ in config.items() if key_ in signature}

        ##### Should be able to infer the model parameters from this...
        self.model_ = DNACipherModel(n_celltypes, n_celltype_factors,  # Cell type information
                                       n_assays, n_assay_factors,  # Assay type information
                                       n_genomic_inputs, n_genomic_factors,  # Genomic sequence information
                                       n_output_factors,  # Epigenetic information to output
                                       n_layers, n_nodes, **config).eval().to(self.device)
        # Load the weights to the model
        self.model_.load_state_dict(model_weights)

        ################################################################################################################
                        # Also attaching the Enformer model to generate the sequence embeddings #
        ################################################################################################################
        self.seqlen_max = 196608 # The input sequence length to Enformer!
        if self.seqlen > self.seqlen_max:
            raise Exception(f"seqlen {seqlen} exceeds maximum model seqlen {self.seqlen_max}")

        from enformer_pytorch import from_pretrained

        self.transformer_model = from_pretrained('EleutherAI/enformer-official-rough').to(self.device)

    def get_celltype_embeds(self):
        """ Gets the celltype embeddings.
        """
        celltype_embeddings = np.zeros((len(self.celltypes), self.model_.celltype_embedding.embedding_dim))
        for celltype_index, celltype in enumerate(self.celltypes):
            celltype_embeddings[celltype_index, :] = self.model_.celltype_embedding(
                torch.tensor(celltype_index).to(self.model_.device)
            ).detach().cpu().numpy()

        return pd.DataFrame(celltype_embeddings, index=self.celltypes)

    def get_assay_embeds(self):
        """ Gets the celltype embeddings.
        """
        assay_embeddings = np.zeros((len(self.assays), self.model_.assay_embedding.embedding_dim))
        for assay_index, assay in enumerate(self.assays):
            assay_embeddings[assay_index, :] = self.model_.assay_embedding(torch.tensor(assay_index).to(
                                                                self.model_.device)).detach().cpu().numpy()

        return pd.DataFrame(assay_embeddings, index=self.assays)

    def get_seq(self, chr_, start, end):
        """ Gets a sequence associated with a particular region.
        """
        target_interval = kipoiseq.Interval(chr_, start, end)
        seq = self.seq_extracter.extract(target_interval)

        return seq

    def get_seq_tokens(self, seq, seq_index=None):
        """Gets sequence token for relevant model."""
        if self.transformer_model_name != 'enformer': # Is NT model
            return self.get_seq_tokens_NT(seq, seq_index=seq_index)
        else:
            return self.get_seq_tokens_enformer(seq)

    def get_seq_tokens_enformer(self, seq):
        """ Tokenises the sequences for expected Enformer input.
        """
        dna_lkp = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return torch.tensor( [dna_lkp[char_.upper()] for char_ in seq] )

    def get_seq_tokens_NT(self, seq, seq_index=None):
        """ Tokenises a given sequence appropriately for input to nucleotide transformer.
        """
        # This is to make sure the tokenizer works properly, since treats every 'N'
        # as a token, and as such results in too many tokens if excessive 'N' not removed.
        tokens = self.transformer_tokenizer.tokenize(seq)[0][1:]  # Remove start token
        diff = len(tokens) - 999  # Number of excess tokens
        seq_N = re.sub(r'(?<=N)N(?=N)', '', seq, diff)  # Delete N's surround by N

        token_output = self.transformer_tokenizer.tokenize(seq_N)
        token_str = token_output[0]
        token_id = token_output[1]
        if len(token_str) > 1000:  # Too many tokens
            # Let's try deleting any N's with a neighbouring N
            diff = len(token_id) - 1000  # Number of excess tokens
            seq_N = re.sub(r'N(?=N)', '', seq_N, diff)

            token_output = self.transformer_tokenizer.tokenize(seq_N)
            token_str = token_output[0]
            token_id = token_output[1]

            if len(token_str) > 1000:  # STILL too many tokens
                # Now we can't remove the N without NOT representing the gap,
                # we have to shorten the sequence on either side by some tokens..
                diff = len(token_id) - 1000  # Number of excess tokens
                front_remove = math.ceil(diff / 2) + 1  # +1 to remove the start token
                back_remove = diff // 2
                token_str_truncated = token_str[front_remove:len(token_str) - back_remove]
                seq_N = ''.join(token_str_truncated)

                token_output = self.transformer_tokenizer.tokenize(seq_N)
                token_str = token_output[0]
                token_id = token_output[1]

                if len(token_str) > 1000:  # STILL too many tokens
                    # In this case, I'm not sure what went wrong, so throw error!
                    error_msg = "Sequence not tokenized correctly"
                    if type(seq_index) != type(None):
                        error_msg += f" for seq specified by region index {seq_index}, sequence:\n{seq}\n"
                    # print(error_msg) # FOR DEBUGGING
                    # return token_id, token_str
                    raise Exception(error_msg)

        return token_id

    def get_token_features(self, token_ids):
        """ Retrieves the features from Nucleotide Transformer based on the sequence tokens.
        """
        tokens = self.jnp.asarray(token_ids, dtype=self.jnp.int32)

        # Running token through the network
        outs = self.transformer_model.apply(self.transformer_parameters, self.random_key,
                                            tokens)

        # Extracting features per token, excluding the start token
        token_features = self.jax.device_get(outs[f'embeddings_{self.transformer_layer}'][:, 1:, :])

        #### Is implemented elsewhere now.
        #### We average the features across the tokens
        #####token_features_reshaped = token_features.mean(axis=1)
        #####return token_features_reshaped
        return token_features

    def get_seq_features(self, seq):
        """ Performs tokenization and extracts sequence features at a specified layer
            in the specified transformer model.

            For consecutation N's, Nucletide transformer treats these as a
            single character, and not a 6-mer. This is dealt with my replacing
            every 6 N's in a row with a single N.
        """
        token_id = self.get_seq_tokens(seq)
        if self.transformer_model_name != 'enformer': # NT model.
            return self.get_token_features([token_id])

        else: # Dealing with Enformer, so need to extract features from it!
            _, embeddings = self.transformer_model(token_id.to(self.device), return_embeddings=True)
            return embeddings # 896 X 3072, i.e. 128bp bins, with 3072 seq features, only one transformation to the signal values.

    def get_seqs(self, chr_, pos, ref, alt, index_base=0):
        """Gets and checks the reference sequence"""
        side_seq = self.seqlen // 2
        if len(ref) == 1:  # Just a single position being replaced, so our midpoint is that position.
            midpoint = pos - index_base
        else:  # Longer ref, so our midpoint is halfway up the insertion..
            midpoint = (pos + (len(ref) // 2)) - index_base

        start = midpoint - side_seq
        mut_start = (pos - index_base) - start  # Mutation starts at this index position.
        end = midpoint + side_seq

        # NOTE the fasta string extractor below ALREADY handles 'N' padding if the query region is longer than
        # the chromosome!!
        #### Extracting the sequence information and adding the mutation....
        ref_seq = self.get_seq(chr_, start, end)

        # Will pad either side of the sequence with N content if is not the max-length of the model sequence
        # Using this as a way to evaluate if Enformer is actually better than the NT embeddings at equivalent sequence
        # lengths, OR is primarily better due to the longer sequence context that Enformer can act on!
        if len(ref_seq) != self.seqlen_max:
            n_pad_left = (self.seqlen_max - len(ref_seq)) // 2
            n_pad_right = math.ceil((self.seqlen_max - len(ref_seq)) / 2)
            ref_seq = ('N'*n_pad_left) + ref_seq + ('N'*n_pad_right)

            # Also need to update the mut_start, since we now have left padding that will increase the position.
            mut_start += n_pad_left

        if len(ref) == 1:  # Simple SNP.
            if ref_seq[mut_start] != ref:
                raise Exception(f"Ref was mean to have bp {ref} but got {ref_seq[mut_start]}.")
            alt_seq = ref_seq[0:mut_start] + alt + ref_seq[mut_start + 1:]
        else:  # A little more complicated, since need to deal with INDEL variants #
            mutation_indices = list(range(mut_start, mut_start + len(ref)))
            ref_seq_split = np.array(list(ref_seq))
            ref_ = ''.join(ref_seq_split[mutation_indices])
            if ref_ != ref:
                raise Exception(f"Ref was mean to have bp {ref} but got {ref_}.")
            alt_seq = ref_seq[0:mut_start] + alt + ref_seq[mut_start + len(ref):]

        # I don't consider case where alternate is shorter, since gets average sequence features anyhow..
        # Did check in the debugger, and NT automatically trims of spacer tokens, and I trim start, so is mean for alt_seq
        if len(alt_seq) > self.seqlen_max:  # It's longer, possibly due to an INDEL. Will truncate on either side to keep ref in middle
            diff = len(alt_seq) - self.seqlen_max
            start_truncate = math.floor(diff / 2)
            end_truncate = math.ceil(diff / 2)
            alt_seq = alt_seq[start_truncate:-end_truncate] # seqs will be slightly out of alignment BUT average features across anyhow.

        return ref_seq, alt_seq

    def check_seqs(self, vcf_df, index_base=0, #variant position are based on 0- or 1- base indexing.
                   verbose=True, log_file=sys.stdout):
        """ Checks if the reference sequence is correct for all listed variants.
        """
        if verbose:
            print(f"CHECKING {vcf_df.shape[0]} variants.", file=log_file, flush=True)

        good_indices = []
        incorrect_cnt = 0
        for i in range(vcf_df.shape[0]):

            chr_, pos, ref, alt = vcf_df.values[i, 0:4]
            try:
                ref_seq, alt_seq = self.get_seqs(chr_, pos, ref, alt, index_base=index_base)
                good_indices.append( i )
            except Exception as exception:
                except_msg = str(exception)
                if not except_msg.startswith('Ref'):
                    print(f"Got unexpected exception for variant {chr_, pos, ref, alt}: {except_msg}",
                          file=sys.stderr, flush=True)
                incorrect_cnt += 1

            if verbose and i % 100 == 0:
                print(f"CHECKED {i}/{vcf_df.shape[0]} variants. {incorrect_cnt}/{i} are incorrect.",
                      file=log_file, flush=True)

        if verbose:
            print(f"CHECKED {vcf_df.shape[0]}/{vcf_df.shape[0]} variants. "
                  f"{incorrect_cnt}/{vcf_df.shape[0]} are incorrect.", file=log_file, flush=True)

        print(f"All variants checked. Returning correct variants.", file=log_file, flush=True)
        return vcf_df.iloc[good_indices, :]

    def get_variant_embeds(self, chr_, pos, ref, alt, index_base=0, full_embeds=False, transpose=True):
        """Gets the embeddings for the reference and alternative variant.
        """
        ref_seq, alt_seq = self.get_seqs(chr_, pos, ref, alt, index_base=index_base)

        #### Extracting the features...
        # Checked how this worked for truncated alt from INDEL, and works fine.
        # BUT have not checked if the
        ref_features = self.get_seq_features(ref_seq)
        alt_features = self.get_seq_features(alt_seq)

        if full_embeds: # For predicting across the sequence region
            return ref_features, alt_features, ref_seq, alt_seq

        # Just average each, same as used for training.
        if self.embed_method == 'mean':
            ref_features = ref_features.mean( axis=1 )
            alt_features = alt_features.mean( axis=1 )

        else:
            #### Dealing with case where there is a deletion. To handle this, will find the closest matching token
            #### in the reference, and subset the reference to this.
            if ref_features.shape[1] != alt_features.shape[1]:
                # Will subset the reference based on matching tokens, to get the difference
                tree = cKDTree(ref_features[0, :, :])
                _, indices = tree.query(alt_features[0, :, :])
                ref_features = ref_features[:, indices, :]

            if self.embed_method=='max-diff':
                # Will construct the ref and alt based embeddings by taking token values that are maximally differenct
                # between sequences.
                diff = ref_features - alt_features
                abs_diff = np.abs( diff )
                max_diff_tokens = np.argmax(abs_diff, axis=1)[0,:]
                ref_features = np.array([[ref_features[0, max_diff_tokens[i], i] for i in range(ref_features.shape[-1])]])
                alt_features = np.array([[alt_features[0, max_diff_tokens[i], i] for i in range(alt_features.shape[-1])]])

                #### For development purposes, want to see how much of a difference it makes in terms of the embedding.
                # Below shows that using the maximum difference tokens for each feature results in a very significant
                # difference in the latent space change. Whereas the change in the averaged embedding for SNP is almost
                # negligible (which would explain some preliminary results where DNACipher predicting almost zero effects
                # for all variants).
                # dev_ = False
                # if dev_:
                #     max_diffs = np.array([diff[0, max_diff_tokens[i], i] for i in range(max_diff_tokens.shape[-1])])
                #     np.histogram(max_diffs)
                #     # (array([  9,  50, 186, 399,  80, 375, 142,  29,   9,   1]),
                #     #  array([-0.952913  , -0.73517025, -0.51742744, -0.2996847 , -0.08194195,
                #     #          0.13580081,  0.35354358,  0.5712863 ,  0.78902906,  1.0067718 ,
                #     #          1.2245146 ], dtype=float32))
                #
                #     x, y = ref_features.mean(axis=1), alt_features.mean(axis=1)
                #     diffs_avg = x-y
                #     np.histogram(diffs_avg)
                #     # (array([2, 12, 74, 176, 319, 354, 225, 92, 18, 8]),
                #     #  array([-3.6056042e-03, -2.9033958e-03, -2.2011877e-03, -1.4989793e-03,
                #     #         -7.9677103e-04, -9.4562769e-05, 6.0764549e-04, 1.3098537e-03,
                #     #         2.0120621e-03, 2.7142703e-03, 3.4164786e-03], dtype=float32))

            elif self.embed_method=='max-diff-token':
                # Will construct the ref and alt embeddings by taking the token with the largest difference between ref
                # and alt.
                diff = ref_features - alt_features
                abs_avg_diff = np.abs(diff[0,:,:]).mean(axis=1)
                token_max_diff = np.argmax( abs_avg_diff )

                ref_features = ref_features[:, token_max_diff, :]
                alt_features = alt_features[:, token_max_diff, :]

                ##### For testing purposes... this looks promising, can see some changes now.
                # np.histogram(ref_features-alt_features)
                # (array([6, 23, 105, 237, 321, 327, 182, 68, 10, 1]),
                #  array([-0.8210764, -0.65007627, -0.47907612, -0.308076, -0.13707586,
                #         0.03392428, 0.20492442, 0.37592456, 0.5469247, 0.71792483,
                #         0.88892496], dtype=float32))

            elif self.embed_method in self.summed_methods:
                # Will put a kernal over the features to reduce noise! #
                kernel = np.ones(self.kernel_size) / self.kernel_size

                # Enformer has different embedding shape, 896 X 3072 features, making this step unnecesary.
                if self.transformer_model_name!='enformer': # NT model
                    ref_features = ref_features[0,:,:]
                    alt_features = alt_features[0,:,:]

                else: # However do need to convert enformer features to numpy array.
                    ref_features = ref_features.cpu().detach().numpy()
                    alt_features = alt_features.cpu().detach().numpy()

                ref_features_smoothed = np.array(
                           [np.convolve(ref_features[:, col], kernel, 'valid') for col in range(ref_features.shape[1])])
                alt_features_smoothed = np.array(
                           [np.convolve(alt_features[:, col], kernel, 'valid') for col in range(alt_features.shape[1])])

                if transpose:
                    ref_features = ref_features_smoothed.transpose()
                    alt_features = alt_features_smoothed.transpose()
                else:
                    ref_features = ref_features_smoothed
                    alt_features = alt_features_smoothed

        return ref_features, alt_features, ref_seq, alt_seq # features are in shape feature X seq_token

    def infer_specific_effect(self, chr_, pos, ref, alt, celltype, assay, index_base=0,
                              return_all=False # Whether to return intermediate results..
                              ):
        """ Infer the variant effect size for the given celltype, assay.
        """
        if self.embed_method in self.unsummed_methods:
            return self.infer_specific_effect_original(chr_, pos, ref, alt, celltype, assay, index_base=index_base,
                                                       return_all=return_all  # Whether to return intermediate results..
                                                        )

        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base)

        ### Predicting the change...
        celltype_index = torch.tensor(self.celltypes.index(celltype)).expand(ref_features.shape[0], 1)
        assay_index = torch.tensor(self.assays.index(assay)).expand(ref_features.shape[0], 1)

        ref_input = torch.tensor(ref_features, device=self.device)
        alt_input = torch.tensor(alt_features, device=self.device)

        #### Predicting change
        with torch.no_grad():
            # TODO need to implement contrast embed layer for DNACV5, currently only implemented for DNACV4
            if self.dnacv=='dnacv4':
                ref_output = self.model_(celltype_index, assay_index, ref_input, self.contrast_embed_layer)
                alt_output = self.model_(celltype_index, assay_index, alt_input, self.contrast_embed_layer)
            elif self.dnacv=='dnacv5':
                # Need to get the positional encodings..
                pos_indices = self.model_.get_positional_indices(chr_, pos)
                ref_output = self.model_(celltype_index, assay_index, ref_input, *pos_indices)
                alt_output = self.model_(celltype_index, assay_index, alt_input, *pos_indices)

        diff = torch.abs(ref_output-alt_output).sum().cpu()
        if self.embed_method == 'sum-abs-signed': # TODO need to check this actually works!!!
            sign = (ref_output - alt_output).sum().cpu()
            nonzero = torch.abs(sign) > 0
            sign[nonzero] = sign[nonzero] / np.abs(sign[nonzero])
            diff = diff * sign

        if not return_all:
            return diff[0][0].item()
        else:
            return diff[0][0].item(), ref_seq, alt_seq, ref_features, alt_features, ref_output, alt_output

    def infer_specific_effect_original(self, chr_, pos, ref, alt, celltype, assay, index_base=0,
                              return_all=False # Whether to return intermediate results..
                              ):
        """ Infer the variant effect size for the given celltype, assay.
        """
        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base)

        ### Predicting the change...
        celltype_index = torch.tensor(self.celltypes.index(celltype), device=self.device)
        assay_index = torch.tensor(self.assays.index(assay), device=self.device)

        ref_input = torch.tensor(ref_features, device=self.device)
        alt_input = torch.tensor(alt_features, device=self.device)

        with torch.no_grad():
            ref_output = self.model_(celltype_index, assay_index, ref_input)
            alt_output = self.model_(celltype_index, assay_index, alt_input)

        diff = ref_output-alt_output

        if not return_all:
            return diff[0][0].item()
        else:
            return diff[0][0].item(), ref_seq, alt_seq, ref_features, alt_features, ref_output, alt_output

    def infer_specific_effect_genome(self, chr_, pos, ref, alt, celltype, assay, index_base=0,
                                     return_all=False, # Whether to return intermediate results..
                                    ):
        """ Infers the variant effect across the full 6KB sequence context.
        """
        if self.dnacv == 'dnacv5':
            print("WARNING: using DNACV5 but have not implemented this method for that model yet...",
                  file=sys.stdout, flush=True)

        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base,
                                                                               full_embeds=True)

        ### Predicting the change...
        celltype_index = torch.tensor(self.celltypes.index(celltype), device=self.device).unsqueeze(0).expand(
                                                                                              ref_features.shape[1], -1)
        assay_index = torch.tensor(self.assays.index(assay), device=self.device).unsqueeze(0).expand(
                                                                                              ref_features.shape[1], -1)

        ref_input = torch.tensor(ref_features, device=self.device).squeeze(0)
        alt_input = torch.tensor(alt_features, device=self.device).squeeze(0)

        with torch.no_grad():
            ref_output = self.model_(celltype_index, assay_index, ref_input)
            alt_output = self.model_(celltype_index, assay_index, alt_input)

        diff = ref_output-alt_output

        if not return_all:
            return diff.numpy()[:,0]
        else:
            return diff.numpy()[:,0], ref_seq, alt_seq, ref_features, alt_features, ref_output, alt_output

    def infer_effects(self, chr_, pos, ref, alt, index_base=0, batch_size=200,
                      batch_axis = 0 #Refers to batch by sequence. Think might be more memory efficient to batch by celltype,assay..
                     ):
        """ Infers effects across all celltype, assays.
        """
        if self.embed_method in self.unsummed_methods:
            return self.infer_effects_original(chr_, pos, ref, alt, index_base=index_base)

        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base,
                                                                               transpose=False)
        ref_features = torch.tensor(ref_features, device=self.device, dtype=torch.float32)
        alt_features = torch.tensor(alt_features, device=self.device, dtype=torch.float32)
        ref_features = ref_features.transpose(0, 1)  # .view(ref_features.shape[1], ref_features.shape[0])
        alt_features = alt_features.transpose(0, 1)  # .view(alt_features.shape[1], alt_features.shape[0])

        torch.cuda.empty_cache()

        with torch.no_grad():
            ##### Need to construct for multi-input...
            celltype_indices = list(range(len(self.celltypes)))
            assay_indices = list(range(len(self.assays)))
            celltype_index_combs = []
            assay_index_combs = []
            for ct_index in celltype_indices:
                for assay_index in assay_indices:
                    celltype_index_combs.append( ct_index )
                    assay_index_combs.append( assay_index )

            # Strategy here repeats each of the sequence region values (as shallow script) ncelltype_input times in a row
            # before then repeating the next set of sequence features, and so on.
            # NOTE the .reshape results in deep copy....
            n_combs = len(celltype_index_combs)  # The number of combinations we need to expand up to!
            ref_input = ref_features
            alt_input = alt_features

            celltype_index_input = torch.tensor(celltype_index_combs, device=self.device).unsqueeze(0).expand(
                                                                                            ref_input.size(0), -1
                                                                                        )
            assay_index_input = torch.tensor(assay_index_combs, device=self.device).unsqueeze(0).expand(
                                                                                            ref_input.size(0), -1
                                                                                        )

            #### Checking to make sure the celltype,assays being repeated correctly.
            #print(celltype_index_input[0:n_combs+10], assay_index_input[0:n_combs+10])
            #### Will do in batches to get it to scale !
            if batch_axis not in [0, 1]:
                raise Exception(f"Invalid batch axis: {batch_axis}.")

            total = ref_input.shape[0] if batch_axis==0 else n_combs
            batch_size = min([batch_size, total])
            n_batches = math.ceil( total / batch_size )

            # Getting positional embeddings for each if using dnacv5
            if self.dnacv == 'dnacv5':
                positional_indices = self.model_.get_positional_indices(chr_, pos, nreps=batch_size)

            starti = 0
            ref_batch_preds = []
            alt_batch_preds = []
            for batchi in range(n_batches):
                endi = min([starti+batch_size, total])
                batchi_size = endi-starti

                # Need to update the positional indices if the batchi_size changes...
                if self.dnacv == 'dnacv5' and batchi_size != batch_size:
                    positional_indices = self.model_.get_positional_indices(chr_, pos, nreps=batchi_size)

                if batch_axis == 0: # Batching either by sequence or celltype/assay
                    celltype_index_batchi = celltype_index_input[starti:endi]
                    assay_index_batchi = assay_index_input[starti:endi]
                    ref_input_batchi = ref_input[starti:endi]
                    alt_input_batchi = alt_input[starti:endi]
                    pred_rows, pred_cols = batchi_size, n_combs

                else:
                    celltype_index_batchi = celltype_index_input[:, starti:endi]
                    assay_index_batchi = assay_index_input[:, starti:endi]
                    ref_input_batchi = ref_input
                    alt_input_batchi = alt_input
                    pred_rows, pred_cols = ref_input.shape[0], batchi_size

                torch.cuda.empty_cache()
                #### dnacv5
                if self.dnacv == 'dnacv5':
                    ref_output = self.model_(celltype_index_batchi, assay_index_batchi, ref_input_batchi, *positional_indices)
                    alt_output = self.model_(celltype_index_batchi, assay_index_batchi, alt_input_batchi, *positional_indices)
                else:
                    ref_output = self.model_(celltype_index_batchi, assay_index_batchi, ref_input_batchi)
                    alt_output = self.model_(celltype_index_batchi, assay_index_batchi, alt_input_batchi)

                torch.cuda.empty_cache()

                # Signal across the sequence #
                ref_preds = ref_output.unsqueeze(1).view(pred_rows, pred_cols)
                alt_preds = alt_output.unsqueeze(1).view(pred_rows, pred_cols)
                ref_batch_preds.append( ref_preds )
                alt_batch_preds.append( alt_preds )

                starti = endi

            ref_preds = torch.concat(ref_batch_preds, dim=batch_axis)
            alt_preds = torch.concat(alt_batch_preds, dim=batch_axis)

            diff = torch.abs((ref_preds - alt_preds)).sum(axis=0).cpu().numpy()
            if self.embed_method == 'sum-abs-signed':
                sign = (ref_preds - alt_preds).sum(axis=0).cpu().numpy()
                nonzero = np.abs(sign) > 0
                sign[nonzero] = sign[nonzero] / np.abs(sign[nonzero])
                diff = diff * sign

            #### Stratifying differences to celltype, assays.
            stratified_result = np.zeros((len(self.celltypes), len(self.assays)), dtype=np.float32)
            stratified_result[celltype_index_combs, assay_index_combs] = diff
            stratified_result = pd.DataFrame(stratified_result, index=self.celltypes, columns=self.assays)

        return stratified_result

    def infer_effects_original(self, chr_, pos, ref, alt, index_base=0):
        """ Infers effects across all celltype, assays.
        """
        print("NOTE this version of inferring variant effects has not been updated for new DNACipher models and "
              "may not work!")
        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base)

        ##### Need to construct for multi-input...
        ref_features_ = np.zeros((self.n_imputable_celltype_assays, ref_features.shape[1]), dtype=np.float32)
        alt_features_ = np.zeros((self.n_imputable_celltype_assays, alt_features.shape[1]), dtype=np.float32)
        celltype_indices = np.zeros((self.n_imputable_celltype_assays), dtype=int)
        assay_indices = np.zeros((self.n_imputable_celltype_assays), dtype=int)

        currenti = 0
        for i, celltype in enumerate(self.celltypes):
            for j, assay in enumerate(self.assays):

                ref_features_[currenti, :] = ref_features[0, :]
                alt_features_[currenti, :] = alt_features[0, :]

                celltype_indices[currenti] = i
                assay_indices[currenti] = j

                currenti += 1

        ref_features_ = torch.from_numpy(ref_features_).to(self.device)
        alt_features_ = torch.from_numpy(alt_features_).to(self.device)
        celltype_indices = torch.from_numpy(celltype_indices).to(self.device)
        assay_indices = torch.from_numpy(assay_indices).to(self.device)

        #### Performing inference...
        with torch.no_grad():
            ref_output = self.model_(celltype_indices, assay_indices, ref_features_)
            alt_output = self.model_(celltype_indices, assay_indices, alt_features_)

        diff = (ref_output-alt_output).detach().cpu().numpy()[:, 0]

        #### Stratifying differences to celltype, assays.
        stratified_result = np.zeros((len(self.celltypes), len(self.assays)), dtype=np.float32)
        currenti = 0
        for i, celltype in enumerate(self.celltypes):
            for j, assay in enumerate(self.assays):

                stratified_result[i, j] = diff[currenti]
                currenti += 1

        stratified_result = pd.DataFrame(stratified_result, index=self.celltypes, columns=self.assays)

        return stratified_result

    def infer_multivariant_effects(self, vcf_df, index_base=0, #variant position are based on 0- or 1- base indexing.
                                   verbose=True, log_file=sys.stdout, batch_size=200, batch_axis=0):
        """Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per
            variant, and predicted effect sizes across the columns for all celltype/assay combinations.
        """
        multivariant_effects = np.zeros((vcf_df.shape[0], self.n_imputable_celltype_assays), dtype=np.float32)
        var_names = []
        for i in range(vcf_df.shape[0]):

            chr_, pos, ref, alt = vcf_df.values[i, 0:4]
            variant_effects = self.infer_effects(chr_, pos, ref, alt, index_base,
                                                 batch_size=batch_size, batch_axis=batch_axis).values.ravel()

            multivariant_effects[i,:] = variant_effects
            var_names.append( f"{chr_}_{pos}_{ref}_{alt}" )

            if verbose and i % math.ceil(min([100, 0.1*vcf_df.shape[0]])) == 0:
                print(f"PROCESSED {i}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        if verbose:
            print(f"PROCESSED {vcf_df.shape[0]}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        multivariant_effects = pd.DataFrame(multivariant_effects,
                                            index=var_names, columns=self.imputable_celltype_assays)
        return multivariant_effects

    ####################################################################################################################
                # Functions from here are for inferring variant differences using the deep embeddings only #
    ####################################################################################################################
    def infer_effects_embeds(self, chr_, pos, ref, alt, embed_layer, index_base=0, batch_size=200,
                      batch_axis = 0 #Refers to batch by sequence. Think might be more memory efficient to batch by celltype,assay..
                     ):
        """ Infers effects across all celltype, assays.
        """

        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base,
                                                                               transpose=False)
        ref_features = torch.tensor(ref_features, device=self.device, dtype=torch.float32)
        alt_features = torch.tensor(alt_features, device=self.device, dtype=torch.float32)
        ref_features = ref_features.transpose(0, 1)  # .view(ref_features.shape[1], ref_features.shape[0])
        alt_features = alt_features.transpose(0, 1)  # .view(alt_features.shape[1], alt_features.shape[0])

        torch.cuda.empty_cache()

        with torch.no_grad():
            ##### Need to construct for multi-input...
            celltype_indices = list(range(len(self.celltypes)))
            assay_indices = list(range(len(self.assays)))
            celltype_index_combs = []
            assay_index_combs = []
            for ct_index in celltype_indices:
                for assay_index in assay_indices:
                    celltype_index_combs.append( ct_index )
                    assay_index_combs.append( assay_index )

            # Strategy here repeats each of the sequence region values (as shallow script) ncelltype_input times in a row
            # before then repeating the next set of sequence features, and so on.
            # NOTE the .reshape results in deep copy....
            n_combs = len(celltype_index_combs)  # The number of combinations we need to expand up to!
            ref_input = ref_features
            alt_input = alt_features

            celltype_index_input = torch.tensor(celltype_index_combs, device=self.device).unsqueeze(0).expand(
                                                                                            ref_input.size(0), -1
                                                                                        )
            assay_index_input = torch.tensor(assay_index_combs, device=self.device).unsqueeze(0).expand(
                                                                                            ref_input.size(0), -1
                                                                                        )

            #### Checking to make sure the celltype,assays being repeated correctly.
            #print(celltype_index_input[0:n_combs+10], assay_index_input[0:n_combs+10])
            #### Will do in batches to get it to scale !
            if batch_axis not in [0, 1]:
                raise Exception(f"Invalid batch axis: {batch_axis}.")

            ##### Batches are in terms of tokens, i.e. batch size of 10 will run 10 tokens through the model at a time.
            total = ref_input.shape[0] if batch_axis==0 else n_combs
            batch_size = min([batch_size, total])
            n_batches = math.ceil( total / batch_size )

            # Getting positional embeddings for each if using dnacv5
            if self.dnacv == 'dnacv5':
                positional_indices = self.model_.get_positional_indices(chr_, pos, nreps=batch_size)

            starti = 0
            dot_sum_by_batch = []
            for batchi in range(n_batches):
                endi = min([starti+batch_size, total])
                batchi_size = endi-starti

                if batch_axis == 0: # Batching either by sequence or celltype/assay
                    celltype_index_batchi = celltype_index_input[starti:endi]
                    assay_index_batchi = assay_index_input[starti:endi]
                    ref_input_batchi = ref_input[starti:endi]
                    alt_input_batchi = alt_input[starti:endi]
                    pred_rows, pred_cols = batchi_size, n_combs

                else:
                    celltype_index_batchi = celltype_index_input[:, starti:endi]
                    assay_index_batchi = assay_index_input[:, starti:endi]
                    ref_input_batchi = ref_input
                    alt_input_batchi = alt_input
                    pred_rows, pred_cols = ref_input.shape[0], batchi_size

                torch.cuda.empty_cache()
                #### dnacv5
                if self.dnacv == 'dnacv5':
                    # Need to update the positional indices if the batchi_size changes...
                    if batchi_size != batch_size:
                        positional_indices = self.model_.get_positional_indices(chr_, pos, nreps=batchi_size)

                    ref_output = self.model_.forward_embedding(celltype_index_batchi, assay_index_batchi,
                                                               ref_input_batchi, embed_layer, *positional_indices)
                    alt_output = self.model_.forward_embedding(celltype_index_batchi, assay_index_batchi,
                                                               alt_input_batchi, embed_layer, *positional_indices)
                else:
                    ref_output = self.model_(celltype_index_batchi, assay_index_batchi, ref_input_batchi, embed_layer)
                    alt_output = self.model_(celltype_index_batchi, assay_index_batchi, alt_input_batchi, embed_layer)

                torch.cuda.empty_cache()

                # Reshaping to get the celltype/assays specific sequence features #
                ref_preds = ref_output.unsqueeze(1).view(pred_rows, pred_cols, self.n_nodes)
                alt_preds = alt_output.unsqueeze(1).view(pred_rows, pred_cols, self.n_nodes)

                # Getting the dot-product across the features, as measurement of difference!
                #x = (ref_preds * alt_preds).sum(axis=2)
                #print(x.std(dim=0), x.std(dim=1))

                ##### Using the dot-product to determine the difference between the two, similar to NT!
                dot_sums = (ref_preds * alt_preds).sum(axis=2)

                dot_sum_by_batch.append( dot_sums )

                starti = endi

            # Summing the dot-products across the DNACipher embeddings, to get per celltype-assay dot-products across
            # DNACipher latent embeddings AND tokens.
            global_dots = torch.concat(dot_sum_by_batch)
            global_dot_sums = global_dots.sum(dim=0)

            #### Stratifying differences to celltype, assays.
            stratified_result = np.zeros((len(self.celltypes), len(self.assays)), dtype=np.float32)
            stratified_result[celltype_index_combs, assay_index_combs] = global_dot_sums.cpu().numpy()
            stratified_result = pd.DataFrame(stratified_result, index=self.celltypes, columns=self.assays)

        return stratified_result

    def infer_multivariant_embedding_effects(self, vcf_df, embed_layer, index_base=0, #variant position are based on 0- or 1- base indexing.
                                   verbose=True, log_file=sys.stdout, batch_size=200, batch_axis=0):
        """Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per
            variant, and predicted effect sizes across the columns for all celltype/assay combinations.
        """
        if embed_layer >= self.n_layers:
            raise Exception(f'Contrast embed layer specified as {embed_layer}, but only {self.n_layers} embeddings.')

        multivariant_effects = np.zeros((vcf_df.shape[0], self.n_imputable_celltype_assays),
                                        dtype=np.float32)
        var_names = []
        for i in range(vcf_df.shape[0]):

            chr_, pos, ref, alt = vcf_df.values[i, 0:4]
            variant_effects = self.infer_effects_embeds(chr_, pos, ref, alt, embed_layer, index_base,
                                                 batch_size=batch_size, batch_axis=batch_axis).values.ravel()

            multivariant_effects[i,:] = variant_effects
            var_names.append( f"{chr_}_{pos}_{ref}_{alt}" )

            if verbose and i % math.ceil(min([100, 0.1*vcf_df.shape[0]])) == 0:
                print(f"PROCESSED {i}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        if verbose:
            print(f"PROCESSED {vcf_df.shape[0]}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        multivariant_effects = pd.DataFrame(multivariant_effects, index=var_names,
                                            columns = [f'{ct_assay}---embed-layer-{embed_layer}'
                                                       for ct_assay in self.imputable_celltype_assays])
        return multivariant_effects

    ####################################################################################################################
      # Functions from here are for inferring variant differences using the Nucleotide Transformer embeddings only #
    ####################################################################################################################
    def infer_NT_embedding_effects(self, chr_, pos, ref, alt, index_base=0,
                      batch_axis = 0 #Refers to batch by sequence. Think might be more memory efficient to batch by celltype,assay..
                     ):
        """ Infers effects across Nucleotide Transformer sequence features, has not celltype,assay info.
        """
        # Pulling out the variant embeddings.
        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base,
                                                                               transpose=False, full_embeds=True)

        # Using the best performing approach for predicting eQTLs from the Nucleotide Transformer, with the
        # caveat I will not use average embeddings, but instead take the dot product across the tokens for each feature,
        # to generate a distance in terms of each 1280 features between the two embeddings.
        # Then will use this information to benchmark NT embedding feature performance for causal variant
        # prioritisation.
        # Just a place holder for development.
        variant_embedding_effects = np.zeros(1280, dtype=float)

        return variant_embedding_effects

    def infer_multivariant_NT_embedding_effects(self, vcf_df, index_base=0,
                                   verbose=True, log_file=sys.stdout, batch_size=200, batch_axis=0):
        """Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per
            variant, and predicted effect sizes across the columns for all celltype/assay combinations.
        """
        multivariant_effects = np.zeros((vcf_df.shape[0], self.model_.n_token_features), dtype=np.float32)
        var_names = []
        for i in range(vcf_df.shape[0]):

            chr_, pos, ref, alt = vcf_df.values[i, 0:4]
            variant_embedding_effects = self.infer_NT_embedding_effects(chr_, pos, ref, alt, index_base,
                                                 batch_size=batch_size, batch_axis=batch_axis).values.ravel()

            multivariant_effects[i, :] = variant_embedding_effects
            var_names.append(f"{chr_}_{pos}_{ref}_{alt}")

            if verbose and i % math.ceil(min([100, 0.1 * vcf_df.shape[0]])) == 0:
                print(f"PROCESSED {i}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        if verbose:
            print(f"PROCESSED {vcf_df.shape[0]}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        # Naming the columns (NT model, layer, feature details) and rows (variants)
        multivariant_effects = pd.DataFrame(multivariant_effects,
                                            index=var_names,
                         columns=[f'NT_feature-{k}_layer-{self.transformer_layer}_model-{self.transformer_model_name}'
                                  for k in self.model_.n_token_features])
        return multivariant_effects

class FastaStringExtractor:
    """ Helper class for extracting sequences.
    """

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

"""Junk code

            if expand:  #### EXPAND strategy

                ref_input = ref_features.unsqueeze(1).expand(-1, n_combs, -1).reshape(-1, ref_features.size(1))
                alt_input = alt_features.unsqueeze(1).expand(-1, n_combs, -1).reshape(-1, ref_features.size(1))

                nseq_combs = ref_input.shape[0]
                celltype_index_input = torch.tensor(celltype_index_combs, device=self.device).unsqueeze(0).expand(
                                                                                                ref_features.size(0), -1
                                                                                                   ).reshape(nseq_combs)
                assay_index_input = torch.tensor(assay_index_combs, device=self.device).unsqueeze(0).expand(
                                                                                                ref_features.size(0), -1
                                                                                                   ).reshape(nseq_combs)


"""


