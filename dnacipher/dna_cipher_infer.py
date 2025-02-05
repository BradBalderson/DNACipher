"""
Inference interface for DNACipher, implements querying the model for anticipated common tasks, example inferring
the effect of a genetic variant.
"""

import math
import numpy as np
import pandas as pd
import time

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
import sys

from .dna_cipher_model import DNACipherModel

class DNACipher():

    def __init__(self, weight_path, sample_file, genome_file, config=None,
                 transformer_model_name='enformer', transformer_layer=24, device='cpu',
                 seqlen=196608, embed_method='sum-abs-signed', kernel_size=50,
                 dnacv='dnacv4', # Version of DNACipher to use
                 ):
        """ Constructs a DNACipher object, which allows for inference of variant function....
        """

        # Base run params
        self.unsummed_methods = ['mean', 'max-diff', 'max-diff-token'] # Work on average embeddings, so just take diff between ref and alt.
        self.summed_methods = ['sum-abs', 'sum-abs-signed'] # work on the full embedding, so different methods for contrasting ref and alt predictions.
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
        model_weights = torch.load(self.weight_path, map_location='cpu', #torch.device(self.device),
                                   weights_only=True)

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
                                       n_layers, n_nodes, **config).eval() #.to( self.device )
        # Load the weights to the model
        self.model_.load_state_dict(model_weights)
        self.model_ = self.model_.to( self.device )

        ################################################################################################################
                        # Also attaching the Enformer model to generate the sequence embeddings #
        ################################################################################################################
        self.seqlen_max = 196608 # The input sequence length to Enformer!
        if self.seqlen > self.seqlen_max:
            raise Exception(f"seqlen {seqlen} exceeds maximum model seqlen {self.seqlen_max}")

        from enformer_pytorch import from_pretrained

        self.transformer_model = from_pretrained('EleutherAI/enformer-official-rough').to( self.device )

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

    def get_seqs(self, chr_, pos, ref, alt, index_base=0, seq_pos=None):
        """Gets and checks the reference sequence

        seq_pos: Specifies the position of the sequence, so that can make predictions that are not variant-centred.
                 Must be within the window size of the model.
        """
        if type(seq_pos) == type(None):
            seq_pos = pos

        if abs(seq_pos - pos) >= self.seqlen_max:
            raise Exception(f"Inputted sequence centre location (seq_pos={seq_pos}) is > max sequence length from the "
                            f"inputted variant location (pos={pos}).")
        elif abs(seq_pos - pos) >= (self.seqlen_max*0.95):
            raise Warning(f"Inputted sequence centre location is >0.90 the maximum long-range effect prediction from "
                          f"the variant, may be less accurate.")

        side_seq = self.seqlen // 2
        midpoint = seq_pos - index_base
        # if len(ref) == 1:  # Just a single position being replaced, so our midpoint is that position.
        #     midpoint = seq_pos - index_base
        # else:  # Longer ref, so our midpoint is halfway up the insertion..
        #     midpoint = (seq_pos + (len(ref) // 2)) - index_base

        start = midpoint - side_seq
        mut_start = (pos - index_base) - start  # Mutation starts at this index position.
        end = midpoint + side_seq

        # NOTE the fasta string extractor below ALREADY handles 'N' padding if the query region is longer than
        # the chromosome!!
        #### Extracting the sequence information and adding the mutation....
        ref_seq = self.get_seq(chr_, start, end)
        ### Check for WRN locus example, where centre between the gene TSS site and the variant location.
        # variant_seq_pos = (len(ref_seq)//2)+(pos-seq_pos)
        # 141343
        # ref_seq[variant_seq_pos-3:variant_seq_pos], ref_seq[variant_seq_pos], ref_seq[variant_seq_pos+1:variant_seq_pos+8]
        # ('TAA', 'T', 'ATTAGAG')

        # Will pad either side of the sequence with N content if is not the max-length of the model sequence
        if len(ref_seq) != self.seqlen_max:
            n_pad_left = (self.seqlen_max - len(ref_seq)) // 2
            n_pad_right = math.ceil((self.seqlen_max - len(ref_seq)) / 2)
            ref_seq = ('N' * n_pad_left) + ref_seq + ('N' * n_pad_right)

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
            alt_seq = alt_seq[start_truncate:-end_truncate]  # seqs will be slightly out of alignment BUT average features across anyhow.

        #### Another check:
        # alt_seq[variant_seq_pos-3:variant_seq_pos], alt_seq[variant_seq_pos], alt_seq[variant_seq_pos+1:variant_seq_pos+8]
        # ('TAA', 'C', 'ATTAGAG') # Correct!

        return ref_seq, alt_seq

    def get_seqs_var_middle(self, chr_, pos, ref, alt, index_base=0):
        """Gets and checks the reference sequence.

        Old version of get_seqs that does not support alternative sequence-centring from the variant position.
        """
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

    def get_variant_embeds(self, chr_, pos, ref, alt, index_base=0, full_embeds=False, transpose=True,
                           seq_pos=None,
                           ):
        """Gets the embeddings for the reference and alternative variant.
        """
        ref_seq, alt_seq = self.get_seqs(chr_, pos, ref, alt, index_base=index_base, seq_pos=seq_pos)

        #### Extracting the features...
        # Checked how this worked for truncated alt from INDEL, and works fine.
        # BUT have not checked if the
        torch.cuda.empty_cache()
        ref_features = self.get_seq_features(ref_seq)
        torch.cuda.empty_cache()
        alt_features = self.get_seq_features(alt_seq)
        torch.cuda.empty_cache()

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


                # Enformer has different embedding shape, 896 X 3072 features, making this step unnecesary.
                if self.transformer_model_name!='enformer': # NT model
                    ref_features = ref_features[0,:,:]
                    alt_features = alt_features[0,:,:]

                else: # However do need to convert enformer features to numpy array.
                    ref_features = ref_features.cpu().detach().numpy()
                    alt_features = alt_features.cpu().detach().numpy()

                # if kernel_size > 0:
                # Will put a kernal over the features to reduce noise! #
                kernel = np.ones(self.kernel_size) / self.kernel_size

                # This actually results in a transposition...
                ref_features_smoothed = np.array(
                           [np.convolve(ref_features[:, col], kernel, 'valid') for col in range(ref_features.shape[1])])
                alt_features_smoothed = np.array(
                           [np.convolve(alt_features[:, col], kernel, 'valid') for col in range(alt_features.shape[1])])

                # else: # Confusing, but it is because I was originally smoothing each time, now I smooth only optionally.
                #     ref_features_smoothed = ref_features.transpose()
                #     alt_features_smoothed = alt_features.transpose()

                if transpose:
                    ref_features = ref_features_smoothed.transpose()
                    alt_features = alt_features_smoothed.transpose()
                else:
                    ref_features = ref_features_smoothed
                    alt_features = alt_features_smoothed

        return ref_features, alt_features, ref_seq, alt_seq # features are in shape feature X seq_token

    def infer_specific_effect(self, chr_, pos, ref, alt, celltype, assay, index_base=0,
                              return_all=False, # Whether to return intermediate results..
                              seq_pos=None, # the position to centre the ref and alt sequence on!
                              full_embeds=True, all_combinations=True,
                              ):
        """ Infer the variant effect size for the given celltypes and assays.

        Parameters
        ----------
        chr_: str
            Chromosome location of the genetic variant.
        pos: int
            Location on the chromosome of the variant.
        ref: str
            Reference sequence at the chromosome location.
        alt: str
            Alternative sequence at the chromosome location.
        celltype: str or list<str>
            A single cell type from within dnacipher.celltypes, or a list of such cell types.
        assay: str or list<str>
            A single assay from within dnacipher.assays, or a list of such assays.
        index_base: int
            0 or 1, specifies the index-base of the inputted variation position (pos).
        return_all: bool
            True to return the signals across the ref and alt sequeneces, the ref and alt sequences, and the latent
            genome representation of the ref and alt sequences.
        seq_pos: int
            Specifies the position to centre the query sequence on, must be within dnacipher.seqlen_max in order to predict
            effect of the genetic variant. If None then will centre the query sequence on the inputted variant.
        full_embeds: bool
            Whether to use the full set of embeddings generated from the sequence. If False then depending on the
            initialisation of the DNACipher model will use different methods to summarise the sequence embedding prior
            to inference.
        all_combinations: bool
            True to generate predicetiosn for all combinations of inputted cell types and assays.
            If False, then celltype and assays input must be lists of the same length,
            and only those specific combinations will be generated.

        Returns
        --------
        diff: float or numpy.array<float>
            Predicted overall signal difference across the sequence for the predicted signal of the ref and alt sequences.
        experiments_outputted: list< tuple<str, str> >
            In-line with diff, each tuple in the list specifies the (celltype, assay) of the predicted effect.
        ref_signals: np.ndarray<float>
            Rows specify the position in the sequence, which are 128bp resolution bins, corresponding to the middle 114,688bp of the input sequence.
            Columns refer to the experiment, and are in-line with 'experiments_outputted'.
        alt_signals: np.ndarray<float>
            Equivalent to 'ref_signals', except for the alternative sequence.
        ref_seq: str
            Reference sequence used for the reference signal inference.
        alt_seq: str
            Alternative sequence use for the alternative signal inference.
        ref_features: np.ndarray<float>
            Rows are position in the sequence (equivalent to rows in ref_signals), columns are sequence features used for inference.
        alt_features: np.ndarray<float>
            Equivalent to ref_features for the alternative sequence.
        """

        ### Predicting the change...
        if type(celltype) == str:
            celltype = [celltype]

        if type(assay) == str:
            assay = [assay]

        missing_celltypes = [ct for ct in celltype if ct not in self.celltypes]
        missing_assays = [assay_ for assay_ in assay if assay_ not in self.assays]
        if len(missing_celltypes) > 0:
            raise Exception(f"Inputted cell types not represented in the model: {missing_celltypes}")
        if len(missing_assays) > 0:
            raise Exception(f"Inputted assays not represented in the model: {missing_assays}")

        if not all_combinations and len(celltype) != len(assay):
            raise Exception("Specified not predicting all combinations of inputted cell types and assays, yet "
                            "did not input the same number of cell types and assays to specify specific experiments."
                            f" Number of cell types inputted, number of assays inputted: {len(celltype), len(assay)}.")

        celltype_indexes = np.array( [self.celltypes.index( celltype_ ) for celltype_ in celltype] )
        assay_indexes = np.array( [self.assays.index( assay_ ) for assay_ in assay] )

        # Generating all pairs of the cell types and assays
        if all_combinations:
            experiments_to_pred = np.array(list( product(celltype_indexes, assay_indexes) ), dtype=int).transpose()
            celltype_indexes = experiments_to_pred[0,:]
            assay_indexes = experiments_to_pred[1,:]

        # Getting the sequence features
        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base,
                                                                               seq_pos=seq_pos, full_embeds=full_embeds)

        celltype_input = torch.tensor(celltype_indexes, device=self.device).unsqueeze(0).expand(ref_features.size(0), -1
                                                                                                )
        assay_input = torch.tensor(assay_indexes, device=self.device).unsqueeze(0).expand(ref_features.size(0), -1)

        ref_input = torch.tensor(ref_features, device=self.device)
        alt_input = torch.tensor(alt_features, device=self.device)

        #### Predicting change
        with torch.no_grad():
            ref_output = self.model_(celltype_input, assay_input, ref_input)
            ref_output = ref_output.unsqueeze(1).view(ref_features.size(0), celltype_input.shape[1]) #position X assay

            alt_output = self.model_(celltype_input, assay_input, alt_input)
            alt_output = alt_output.unsqueeze(1).view(ref_features.size(0), celltype_input.shape[1])  # position X assay

        diff = torch.abs(alt_output - ref_output).sum(axis=0).cpu()
        if self.embed_method == 'sum-abs-signed': # TODO need to check this actually works!!!
            sign = (alt_output - ref_output).sum(axis=0).cpu()
            nonzero = torch.abs(sign) > 0
            sign[nonzero] = sign[nonzero] / np.abs(sign[nonzero])
            diff = diff * sign

        if len(diff.shape) == 0: # Only a single experiment predicted
            diff = diff.item()
        else:
            diff = diff.numpy()

        #### Getting the cell type assay names that we output!
        experiments_output = [(str( self.celltypes[ct_index] ), str(self.assays[assay_index]))
                              for ct_index, assay_index in zip(celltype_indexes, assay_indexes)]

        if not return_all:
            return diff, experiments_output
        else:
            return diff, experiments_output, ref_output.cpu().detach().numpy(), alt_output.cpu().detach().numpy(), ref_seq, alt_seq, ref_features.cpu().detach().numpy(), alt_features.cpu().detach().numpy()

    def infer_effects(self, chr_, pos, ref, alt, index_base=0, batch_size=900,
                      batch_axis = 1, #Refers to batch by sequence. batch_axis=1 will batch by celltype/assay, batch_axis=0 will do so by sequence embeddings.
                      seq_pos = None, # the position to centre the ref and alt sequence on!
                      full_embeds = True, # Whether to use the full embeds or not for the effect inference,
                                           # if false, will perform some smoothing of the embeddings, making them slightly
                                           # smaller but lowering the resolution of the embeddings..
                      verbose=False,
                     ):
        """ Infers effects across all celltype, assays. Using batching strategy to circumvent high memory requirements.

        Parameters
        ----------
        chr_: str
            Chromosome location of the genetic variant.
        pos: int
            Location on the chromosome of the variant.
        ref: str
            Reference sequence at the chromosome location.
        alt: str
            Alternative sequence at the chromosome location.
        index_base: int
            0 or 1, specifies the index-base of the inputted variation position (pos).
        batch_size: int
            The number of experiments or sequence embeddings to parse at a time through the model to predict signals. Lower if run into memory errors.
        batch_axis: int
            0 or 1, 1 will batch by experiments, batch_axis=0 will batch by sequence position. Default is best.
        seq_pos: int
            Specifies the position to centre the query sequence on, must be within dnacipher.seqlen_max in order to predict
            effect of the genetic variant. If None then will centre the query sequence on the inputted variant.
        full_embeds: bool
            Whether to use the full set of embeddings generated from the sequence. If False then depending on the
            initialisation of the DNACipher model will use different methods to summarise the sequence embedding prior
            to inference.
        verbose: bool
            True for detailed printing of progress.

        Returns
        --------
        pred_effects: pd.DataFrame
            Rows are cell types, columns are assays, values are summarised predicted effects of the variant in the particular celltype/assay combination.
        """

        ref_features, alt_features, ref_seq, alt_seq = self.get_variant_embeds(chr_, pos, ref, alt, index_base,
                                                                               transpose=False,
                                                                               seq_pos=seq_pos,
                                                                               full_embeds=full_embeds
                                                                               )
        ref_features = torch.tensor(ref_features, device=self.device, dtype=torch.float32)
        alt_features = torch.tensor(alt_features, device=self.device, dtype=torch.float32)
        # ref_features = ref_features.transpose(0, 1)  # .view(ref_features.shape[1], ref_features.shape[0])
        # alt_features = alt_features.transpose(0, 1)  # .view(alt_features.shape[1], alt_features.shape[0])

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

            starti = 0
            ref_batch_preds = []
            alt_batch_preds = []
            start_ = time.time()
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
                ref_output = self.model_(celltype_index_batchi, assay_index_batchi, ref_input_batchi)
                alt_output = self.model_(celltype_index_batchi, assay_index_batchi, alt_input_batchi)

                torch.cuda.empty_cache()

                # Signal across the sequence #
                ref_preds = ref_output.unsqueeze(1).view(pred_rows, pred_cols)
                alt_preds = alt_output.unsqueeze(1).view(pred_rows, pred_cols)
                ref_batch_preds.append( ref_preds )
                alt_batch_preds.append( alt_preds )

                starti = endi

                if verbose:
                    print(f"Finished inference for batch {batchi+1} / {n_batches} in {round((time.time()-start_)/60, 3)}mins")

            ref_preds = torch.concat(ref_batch_preds, dim=batch_axis)
            alt_preds = torch.concat(alt_batch_preds, dim=batch_axis)

            diff = torch.abs((alt_preds - ref_preds)).sum(axis=0).cpu().numpy()
            if self.embed_method == 'sum-abs-signed':
                sign = (alt_preds - ref_preds).sum(axis=0).cpu().numpy()
                nonzero = np.abs(sign) > 0
                sign[nonzero] = sign[nonzero] / np.abs(sign[nonzero])
                diff = diff * sign

            #### Stratifying differences to celltype, assays.
            stratified_result = np.zeros((len(self.celltypes), len(self.assays)), dtype=np.float32)
            stratified_result[celltype_index_combs, assay_index_combs] = diff
            stratified_result = pd.DataFrame(stratified_result, index=self.celltypes, columns=self.assays)

        return stratified_result

    def infer_multivariant_effects(self, vcf_df, index_base=0, #variant position are based on 0- or 1- base indexing.
                                   verbose=True, log_file=sys.stdout, batch_size=900, batch_axis=1,
                                   full_embeds=True, seq_pos_col=None):
        """Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per
            variant, and predicted effect sizes across the columns for all celltype/assay combinations.

        Parameters
        ----------
        vcf: pd.DataFrame
            Rows represent particular genetic variants, columns are CHR, POS, REF, ALT
        index_base: int
            0 or 1, specifies the index-base of the inputted variant position (pos).
        batch_size: int
            The number of experiments or sequence embeddings to parse at a time through the model to predict signals. Lower if run into memory errors.
        batch_axis: int
            0 or 1, 1 will batch by experiments, batch_axis=0 will batch by sequence position. Default is best.
        seq_pos_col: int
            Column in vcf that specifies the position to centre the query sequence on, must be within dnacipher.seqlen_max in order to predict
            effect of the genetic variant. If None then will centre the query sequence on the inputted variant.
        full_embeds: bool
            Whether to use the full set of embeddings generated from the sequence. If False then depending on the
            initialisation of the DNACipher model will use different methods to summarise the sequence embedding prior
            to inference.
        verbose: bool
            True for detailed printing of progress.

        Returns
        --------
        var_pred_effects: pd.DataFrame
            Rows are variants, and columns are predicted experiments, in format '<celltype>---<assay>'. Values are summarised predicted effects of the variant in the particular celltype/assay combination.
        """
        seq_positions = None
        if type(seq_pos_col)!=type(None):
            seq_positions = vcf_df[seq_pos_col].values

        multivariant_effects = np.zeros((vcf_df.shape[0], self.n_imputable_celltype_assays), dtype=np.float32)
        var_names = []
        for i in range(vcf_df.shape[0]):

            chr_, pos, ref, alt = vcf_df.values[i, 0:4]
            if type(seq_pos_col)!=type(None):
                seq_pos = seq_positions[i]

            variant_effects = self.infer_effects(chr_, pos, ref, alt, index_base,
                                                 batch_size=batch_size, batch_axis=batch_axis,
                                                 full_embeds=full_embeds, seq_pos=seq_pos, verbose=verbose).values.ravel()

            multivariant_effects[i,:] = variant_effects
            var_names.append( f"{chr_}_{pos}_{ref}_{alt}" )

            if verbose and i % math.ceil(min([100, 0.1*vcf_df.shape[0]])) == 0:
                print(f"PROCESSED {i}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        if verbose:
            print(f"PROCESSED {vcf_df.shape[0]}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        multivariant_effects = pd.DataFrame(multivariant_effects,
                                            index=var_names, columns=self.imputable_celltype_assays)
        return multivariant_effects

    @staticmethod
    def normalise_and_order_effects(pred_effects,
                                    small_max_=0.6 # Below this value, will not scale the predictions, since they very small
                                    ):
        """ Performs a normalisation for the predict effects, that scaled the predictions by the maximum observed signal
            predicted across the locus.

        Parameters
        ----------
        pred_effects: pd.DataFrame
            Rows are cell types, columns are assays, values are summarised predicted effects of the variant in the particular celltype/assay combination.
        small_max_: float
            The normalisation proposed can make very small differences appear large, this cutoff essentially does not perform the normalisation, and so the small effects remain small.

        Returns
        --------
        pred_effects_ordered: pd.DataFrame
            Same as pred_effects input, except the values are now normalised so they are more comparable across assays. The rows and columns are re-ordered to reflect the most effect assays and cell types.
        """

        maxs_ = pred_effects.values.max(axis=0)
        maxs_[maxs_ < small_max_] = 1  # Some will be very small effects, so will keep these the same
        pred_effects_normed = pred_effects / maxs_

        # For the pred effects, recommend squaring the RNA-seq
        rnaseq_assays_bool = ['RNA-seq' in exper for exper in pred_effects.columns]
        pred_effects_normed.loc[:, rnaseq_assays_bool] = pred_effects_normed.loc[:, rnaseq_assays_bool] ** 2

        #### Getting order of effects to plot.
        assays_mean_effects = np.abs(pred_effects.values).mean(axis=0)
        assays_order = np.argsort(-assays_mean_effects)

        celltypes_mean_effects = np.abs(pred_effects.values[:, assays_order[0:5]]).mean(axis=1)
        celltypes_order = np.argsort(-celltypes_mean_effects)

        pred_effects_ordered = pred_effects_normed.iloc[:,assays_order].iloc[celltypes_order,:]

        return pred_effects_ordered

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


