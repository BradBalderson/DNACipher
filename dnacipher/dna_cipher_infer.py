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

import pybedtools

import sys

from .dna_cipher_model import DNACipherModel

class DNACipher():

    def __init__(self, weight_path, sample_file, genome_file, config, device='cpu',
                 transformer_model_name='enformer',
                 ):
        """ Constructs a DNACipher object, which allows for inference of variant function....
        """

        # Base run params
        self.device = device

        model_configs = {'enformer':
                             {'seqlen': 196_608, # total receptive field
                              'seq_pred_range': 114_688, # middle of sequence with predictions
                              'seq_pred_binsize': 128, # resolution of predictions
                              }
                         }
        if transformer_model_name not in model_configs:
            raise Exception(f"Unsupported model: {transformer_model_name}")
        else:
            model_config = model_configs[ transformer_model_name ]

        self.transformer_model_name = transformer_model_name

        # Important for loading sequence information.
        self.seq_extracter = FastaStringExtractor( genome_file )

        self.seqlen = model_config['seqlen'] # Receptive field of the model
        self.seq_pred_range = model_config['seq_pred_range'] # The middle of the sequence for which predictions are made
        self.seq_edge_len = (self.seqlen - self.seq_pred_range) // 2 # +/- this edge does not have predictions.
        self.seq_pred_binsize = model_config['seq_pred_binsize']

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

        #### Loading the config... if not provided resort to default.
        if type(config) == type(None):
            raise Exception("config cannot be None")

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
        from enformer_pytorch import from_pretrained

        self.transformer_model = from_pretrained('EleutherAI/enformer-official-rough').to( self.device )

    def get_celltype_embeds(self):
        """ # TODO test
        Gets the celltype embeddings.
        """
        celltype_embeddings = np.zeros((len(self.celltypes), self.model_.celltype_embedding.embedding_dim))
        for celltype_index, celltype in enumerate(self.celltypes):
            celltype_embeddings[celltype_index, :] = self.model_.celltype_embedding(
                torch.tensor(celltype_index).to(self.model_.device)
            ).detach().cpu().numpy()

        return pd.DataFrame(celltype_embeddings, index=self.celltypes)

    def get_assay_embeds(self):
        """ # TODO test
        Gets the celltype embeddings.
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

    def get_seq_tokens(self, seq):
        """Gets sequence token for relevant model."""
        if self.transformer_model_name == 'enformer':
            return self.get_seq_tokens_enformer(seq)

    def get_seq_tokens_enformer(self, seq):
        """ Tokenises the sequences for expected Enformer input.
        """
        dna_lkp = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return torch.tensor( [dna_lkp[char_.upper()] for char_ in seq] )

    def get_seq_features(self, seq):
        """ Performs tokenization and extracts sequence features at a specified layer
            in the specified transformer model.

            For consecutation N's, Nucletide transformer treats these as a
            single character, and not a 6-mer. This is dealt with my replacing
            every 6 N's in a row with a single N.
        """
        token_id = self.get_seq_tokens(seq)
        if self.transformer_model_name == 'enformer': # NT model.
            _, embeddings = self.transformer_model(token_id.to(self.device), return_embeddings=True)
            return embeddings  # 896 X 3072, i.e. 128bp bins, with 3072 seq features, only one transformation to the signal values.

    def get_seqs(self, chr_, pos, ref, alt, index_base=0, seq_pos=None, correct_ref=False):
        """Gets and checks the reference sequence

        seq_pos: Specifies the position of the sequence, so that can make predictions that are not variant-centred.
                 Must be within the window size of the model.
        """
        if type(seq_pos) == type(None):
            seq_pos = pos

        if abs(seq_pos - pos) >= self.seqlen:
            raise Exception(f"Inputted sequence centre location (seq_pos={seq_pos}) is > max sequence length from the "
                            f"inputted variant location (pos={pos}).")
        elif abs(seq_pos - pos) >= (self.seqlen*0.95):
            raise Warning(f"Inputted sequence centre location is >0.90 the maximum long-range effect prediction from "
                          f"the variant, may be less accurate.")

        side_seq = self.seqlen // 2
        midpoint = seq_pos - index_base

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

        if len(ref) == 1:  # Simple SNP.
            if ref_seq[mut_start] != ref and not correct_ref:
                raise Exception(f"{chr_}:{pos}:{ref}:{alt} was meant to have bp {ref} " + \
                                f"but got {ref_seq[mut_start]}. Check correct genome / coordinates. \n" +\
                                f"If genome/coordinates are correct, the REF may be wrong due to lifting over \n" +\
                                f"variant coordinates between genomes but NOT lifting over the actual variant sequence.\n" +\
                                f"If this is the case, re-run with correct_ref=True, which will introduce the ref/alt base-pairs \n"+\
                                f"regardless of what is found in the inputted reference genome.")
            
            ### Correcting the reference sequence if is incorrect bp as per user request via correct_ref input.
            elif ref_seq[mut_start] != ref and correct_ref:
                ref_seq = ref_seq[0:mut_start] + ref + ref_seq[mut_start + 1:]

            alt_seq = ref_seq[0:mut_start] + alt + ref_seq[mut_start + 1:]

        else:  # A little more complicated, since need to deal with INDEL variants #
            mutation_indices = list(range(mut_start, mut_start + len(ref)))
            ref_seq_split = np.array(list(ref_seq))
            ref_ = ''.join(ref_seq_split[mutation_indices])
            if ref_ != ref and not correct_ref:
                raise Exception(f"{chr_}:{pos}:{ref}:{alt} was meant to have bp {ref} " + \
                                f"but got {ref_seq[mut_start]}. Check correct genome / coordinates. \n" +\
                                f"If genome/coordinates are correct, the REF may be wrong due to lifting over \n" +\
                                f"variant coordinates between genomes but NOT lifting over the actual variant sequence.\n" +\
                                f"If this is the case, re-run with correct_ref=True, which will introduce the ref/alt base-pairs \n"+\
                                f"regardless of what is found in the inputted reference genome.")
            
            ### Correcting the reference sequence if is incorrect bp as per user request via correct_ref input.
            elif ref_ != ref and correct_ref:
                ref_seq = ref_seq[0:mut_start] + ref + ref_seq[mut_start + len(ref):]
            
            alt_seq = ref_seq[0:mut_start] + alt + ref_seq[mut_start + len(ref):]

        # I don't consider case where alternate is shorter, since gets average sequence features anyhow..
        # Did check in the debugger, and NT automatically trims of spacer tokens, and I trim start, so is mean for alt_seq
        if len(alt_seq) > self.seqlen:  # It's longer, possibly due to an INDEL. Will truncate on either side to keep ref in middle
            diff = len(alt_seq) - self.seqlen
            start_truncate = math.floor(diff / 2)
            end_truncate = math.ceil(diff / 2)
            alt_seq = alt_seq[start_truncate:-end_truncate]  # seqs will be slightly out of alignment BUT average features across anyhow.

        #### Another check:
        # alt_seq[variant_seq_pos-3:variant_seq_pos], alt_seq[variant_seq_pos], alt_seq[variant_seq_pos+1:variant_seq_pos+8]
        # ('TAA', 'C', 'ATTAGAG') # Correct!

        # seq_range, converting back to the index_base the user is using
        seq_range = (chr_, start+index_base, end+index_base)

        return ref_seq, alt_seq, seq_range

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
                ref_seq, alt_seq, _ = self.get_seqs(chr_, pos, ref, alt, index_base=index_base)
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

    def get_variant_embeds(self, chr_, pos, ref, alt, index_base=0, seq_pos=None, correct_ref=False):
        """ Gets the embeddings for the reference and alternative variant.
        """
        ref_seq, alt_seq, seq_range = self.get_seqs(chr_, pos, ref, alt, 
                                                    index_base=index_base, 
                                                    seq_pos=seq_pos,
                                                    correct_ref=correct_ref)

        #### Extracting the features...
        # Checked how this worked for truncated alt from INDEL, and works fine.
        # BUT have not checked if the
        torch.cuda.empty_cache()
        ref_features = self.get_seq_features( ref_seq )
        torch.cuda.empty_cache()
        alt_features = self.get_seq_features( alt_seq )
        torch.cuda.empty_cache()

        # Determining the positions of the sequence bins
        pred_start = seq_range[1] + self.seq_edge_len # Getting the predicted sequence range
        pred_end = seq_range[2] - self.seq_edge_len

        seq_bin_starts = list(range(pred_start, pred_end, self.seq_pred_binsize))
        seq_bin_ends = list(range(pred_start+self.seq_pred_binsize, pred_end+1, self.seq_pred_binsize))

        seq_bins = np.array([seq_bin_starts, seq_bin_ends]).transpose()

        return ref_features, alt_features, ref_seq, alt_seq, seq_range, seq_bins

    def infer_effects(self, chr_, pos, ref, alt, celltypes, assays,
                      index_base=0, correct_ref=False, seq_pos = None, effect_region = None,
                      batch_size=None, batch_by = None,
                      all_combinations=True,
                      return_all = False,
                      verbose=False,
                     ):
        """ Infers effects across celltypes and assays. Using batching strategy to circumvent high memory requirements.

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
        correct_ref: bool
            If False, then if the inputted ref does not match the ref at the indicate genome coordinates,
            will raise an error due to likely genome / position mis-specification. If True, this is by-passed by
            always replacing the ref bp at the indicated position in the genome. This is useful for case where the
            variant positions have been lifted over (e.g. hg19 -> hg38) but not he variant sequence (which might have
            changed between genomes). In this case, simplest to just substitute the indicate ref and alt bases as 
            indicated by the input. 
        seq_pos: int
            Specifies the position to centre the query sequence on, must be within dnacipher.seqlen_max in order to predict
            effect of the genetic variant. If None then will centre the query sequence on the inputted variant.
        effect_region: tuple, list, or None
            Specifies the region to calculated the predicted effect size, in format (start, end). The chromosome is
            taken as the same as the inputted variant chromosome (since we cannot currently predicted trans-effects).
        batch_size: int
            The number of experiments or sequence embeddings to parse at a time through the model to predict signals. Lower if run into memory errors. None means compute everything in one batch.
        batch_by: int
            Indicates how to batch the data when fed into the model, either by 'experiment', 'sequence', or None. If None, will automatically choose whichever is the larger axis.
        all_combinations: bool
            True to generate predicetiosn for all combinations of inputted cell types and assays.
            If False, then celltype and assays input must be lists of the same length,
            and only those specific combinations will be generated.
        return_all: bool
            True to return the signals across the ref and alt sequences, the ref and alt sequences, and the latent
            genome representation of the ref and alt sequences.
        verbose: bool
            True for detailed printing of progress.

        Returns
        --------
        pred_effects: pd.DataFrame
            Rows are cell types, columns are assays, values are summarised predicted effects of the variant in the particular celltype/assay combination.
        """
        # Checking batch_by input:
        batch_options = ['experiment', 'sequence', None]
        if batch_by not in batch_options:
            raise Exception(
                f"Unsupported input option, batch_by={batch_by}, but only these options supported: {batch_options}")

        # Getting the cell type-assay inputs
        if type(celltypes) == str:
            celltypes = [celltypes]

        if type(assays) == str:
            assays = [assays]

        missing_celltypes = [ct for ct in celltypes if ct not in self.celltypes]
        missing_assays = [assay_ for assay_ in assays if assay_ not in self.assays]
        if len(missing_celltypes) > 0:
            raise Exception(f"Inputted cell types not represented in the model: {missing_celltypes}")
        if len(missing_assays) > 0:
            raise Exception(f"Inputted assays not represented in the model: {missing_assays}")

        if not all_combinations and len(celltypes) != len(assays):
            raise Exception("Specified not predicting all combinations of inputted cell types and assays, yet "
                            "did not input the same number of cell types and assays to specify specific experiments."
                            f" Number of cell types inputted, number of assays inputted: {len(celltypes), len(assays)}.")

        celltype_indexes = np.array([self.celltypes.index(celltype_) for celltype_ in celltypes])
        assay_indexes = np.array([self.assays.index(assay_) for assay_ in assays])

        # Generating all pairs of the cell types and assays
        if all_combinations:
            # Checking that the cell types or assays are not duplicated, because this causes issues in the logic!
            if len(set(list(celltypes))) != len(celltypes):
                raise Exception("Inputted celltypes contain duplicates, deduplicate first e.g. list(set(celltypes)).")
            if len(set(list(assays))) != len(assays):
                raise Exception("Inputted assays contain duplicates, deduplicate first e.g. list(set(assays)).")

            experiments_to_pred = np.array(list(product(celltype_indexes, assay_indexes)), dtype=int).transpose()
            celltype_indexes = experiments_to_pred[0, :]
            assay_indexes = experiments_to_pred[1, :]

            # And getting their names
            experiment_names = np.array(list(product(celltypes, assays)), dtype=str).transpose()
            celltype_names = list( experiment_names[0, :] )
            assay_names = list( experiment_names[1, :] )

        else: # Not all combinations, just the input names:
            celltype_names = celltypes
            assay_names = assays

        # Getting the sequence embeddings input:
        ref_features, alt_features, ref_seq, alt_seq, seq_range, seq_bins = self.get_variant_embeds(chr_, pos, ref, alt,
                                                                                            index_base, seq_pos=seq_pos,
                                                                                            correct_ref=correct_ref,
                                                                               )

        # Calculating the predicted effect for a particular part of the input sequence
        subset_output_preds = False
        seq_bins_inrange = None
        if type(effect_region)!=type(None): # Only calculate effects for this range of the sequence.
            seq_bins_inrange = np.logical_and(seq_bins[:, 0] >= effect_region[0],
                                              seq_bins[:, 0] < effect_region[1])
            if sum(seq_bins_inrange) == 0: # No intersect with the outputted predictions, doesn't make sense.
                raise Exception(f"Inputted effect_region ({effect_region}) does not intersect with the region of the "
                                f"sequence for which signals can be inferred ({(seq_bins[0,0], seq_bins[-1,-1])})."
                                f"Either set effect_region=None to use whole region to estimated effect size, or "
                                f"set effect_region to be within the bounds of {(seq_bins[0,0], seq_bins[-1,-1])}.")

            if return_all:
                # Indicates the user still wants the full output of predicted signals along the sequences, and therefore
                # we need to subset the seq_bins_inrage AFTER inferring the full set of effects.
                subset_output_preds = True

            else:
                subset_output_preds = False

                # If user does not want the full predicted effects as output, so can speed up inference by only making
                # signal predictions at the locations the locations need to make the effect prediction!
                # Take the bins that have ANY overlap with the specified range of effects
                ref_features = ref_features[seq_bins_inrange, :]
                alt_features = alt_features[seq_bins_inrange, :]

        # Creating tensor representations of these inputs:
        torch.cuda.empty_cache()

        celltype_index_input = torch.tensor(celltype_indexes, device=self.device).unsqueeze(0).expand(ref_features.size(0),
                                                                                                -1)
        assay_index_input = torch.tensor(assay_indexes, device=self.device).unsqueeze(0).expand(ref_features.size(0), -1)

        ref_input = ref_features.clone().detach().to(device=self.device)
        alt_input = alt_features.clone().detach().to(device=self.device)

        # Determining number of combinations to impute:
        n_combs = len( celltype_indexes )

        # Determining the batch_axis automatically, based on whichever axis is larger!
        # batch_by: int
        #     0 or 1, 1 will batch by experiments, batch_axis = 0 will batch by sequence position.
        seq_size = ref_features.size(0)
        sizes = [seq_size, n_combs]
        if type(batch_by) == type(None):
            batch_axis = np.argmax( sizes )  # Choose the larger axis to batch by
        else:  # User defined, already checked above that it is a valid input.
            batch_axis = batch_options.index( batch_by )

        if type( batch_size ) == type(None):  # Compute everything in one batch:
            batch_size = sizes[ batch_axis ]

        with torch.no_grad():
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

            # using sum-abs-signed method to score variants
            if not subset_output_preds: # Already have the relevant set of predictions we want to infer effects for
                diff = torch.abs((alt_preds - ref_preds)).sum(axis=0).cpu().numpy()
                sign = (alt_preds - ref_preds).sum(axis=0).cpu().numpy()

            else: # Need to subset to the relevant positions
                alt_preds_sub = alt_preds[seq_bins_inrange, :]
                ref_preds_sub = ref_preds[seq_bins_inrange, :]

                diff = torch.abs((alt_preds_sub - ref_preds_sub)).sum(axis=0).cpu().numpy()
                sign = (alt_preds_sub - ref_preds_sub).sum(axis=0).cpu().numpy()

        nonzero = np.abs(sign) > 0
        sign[nonzero] = sign[nonzero] / np.abs(sign[nonzero])
        diff = diff * sign

        #### Stratifying differences to celltype, assays. Taking into account what celltype, assay were inputted.
        celltype_names_unique = list( np.unique( celltype_names ) )
        assay_names_unique = list( np.unique( assay_names ) )

        celltype_names_indexes = [celltype_names_unique.index(ct_name) for ct_name in celltype_names]
        assay_names_indexes = [assay_names_unique.index(assay_name) for assay_name in assay_names]

        stratified_result = np.full((len(celltype_names_unique), len(assay_names_unique)),
                                    fill_value=np.nan, dtype=np.float32) # Account for all_combinations = False
        stratified_result[celltype_names_indexes, assay_names_indexes] = diff
        stratified_result = pd.DataFrame(stratified_result, index=celltype_names_unique, columns=assay_names_unique)

        if not return_all:
            return stratified_result

        else:
            # Need to return the full result, so will reformat the sequence-specific information.
            seq_bin_names = [f"{chr_}_{seq_bins[i,0]}_{seq_bins[i,1]}" for i in range(seq_bins.shape[0])]
            experiment_names_flat = ['---'.join(experiment_names[:,i]) for i in range(experiment_names.shape[1])]

            ref_preds_df = pd.DataFrame(ref_preds.cpu().detach().numpy(), index=seq_bin_names,
                                        columns=experiment_names_flat)
            alt_pred_df = pd.DataFrame(alt_preds.cpu().detach().numpy(), index=seq_bin_names,
                                        columns=experiment_names_flat)

            ref_features_df = pd.DataFrame(ref_features.cpu().detach().numpy(), index=seq_bin_names)
            alt_features_df = pd.DataFrame(alt_features.cpu().detach().numpy(), index=seq_bin_names)

            return stratified_result, ref_preds_df, alt_pred_df, ref_seq, alt_seq, ref_features_df, alt_features_df

    def infer_multivariant_effects(self, vcf_df, celltypes, assays,
                                    index_base=0, correct_ref=False, seq_pos_col=None, effect_region_cols=None,
                                    batch_size=900, batch_by=None, all_combinations=True,
                                    verbose=True, log_file=sys.stdout):
        """Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per
            variant, and predicted effect sizes across the columns for all celltype/assay combinations.

        Parameters
        ----------
        vcf: pd.DataFrame
            Rows represent particular genetic variants, columns are CHR, POS, REF, ALT
        celltypes: str or list<str>
            A single cell type from within dnacipher.celltypes, or a list of such cell types.
        assays: str or list<str>
            A single assay from within dnacipher.assays, or a list of such assays.
        index_base: int
            0 or 1, specifies the index-base of the inputted variant position (pos).
        correct_ref: bool
            If False, then if the inputted ref does not match the ref at the indicate genome coordinates,
            will raise an error due to likely genome / position mis-specification. If True, this is by-passed by
            always replacing the ref bp at the indicated position in the genome. This is useful for case where the
            variant positions have been lifted over (e.g. hg19 -> hg38) but not he variant sequence (which might have
            changed between genomes). In this case, simplest to just substitute the indicate ref and alt bases as 
            indicated by the input.
        batch_size: int
            The number of experiments or sequence embeddings to parse at a time through the model to predict signals. Lower if run into memory errors.
        batch_by: int
            Indicates how to batch the data when fed into the model, either by 'experiment', 'sequence', or None. If None, will automatically choose whichever is the larger axis.
        seq_pos_col: int
            Column in vcf that specifies the position to centre the query sequence on, must be within dnacipher.seqlen_max in order to predict
            effect of the genetic variant. If None then will centre the query sequence on the inputted variant.
        effect_region_cols: tuple(str, str)
            Specifies columns in the inputted data frame, with the first specifying the start position (in genome coords)
            to measure the effect, and the second specifying end position (in genome coords) to measure the effect for
            the given variant in the sequence field-of-view across all the different celltype-assays.
        verbose: bool
            True for detailed printing of progress.

        Returns
        --------
        var_pred_effects: pd.DataFrame
            Rows are variants, and columns are predicted experiments, in format '<celltype>---<assay>'. Values are summarised predicted effects of the variant in the particular celltype/assay combination.
        """
        seq_pos = None
        if type(seq_pos_col)!=type(None):
            seq_positions = vcf_df[seq_pos_col].values

        effect_region = None
        if type(effect_region_cols)!=type(None):
            effect_regions = vcf_df[effect_region_cols].values

        # Need to determine a list of cell types and assays that will be inferred, since will output a flattened
        # dataframe.
        imputed_celltypes = []
        imputed_assays = []
        imputed_celltype_assays = []
        imputed_celltype_assays_str = []
        if all_combinations:
            iterator_ = product(celltypes, assays) # Need to generate pairs.
        else:
            iterator_ = zip(celltypes, assays) # Pairs already generated by user.

        for celltype, assay in iterator_:
            imputed_celltypes.append( celltype )
            imputed_assays.append( assay )
            imputed_celltype_assays.append( [celltype, assay] )
            imputed_celltype_assays_str.append( '---'.join( [celltype, assay] ) )

        multivariant_effects = np.zeros((vcf_df.shape[0], len(imputed_celltype_assays)), dtype=np.float32)
        var_names = []
        for i in range(vcf_df.shape[0]):

            chr_, pos, ref, alt = vcf_df.values[i, 0:4]
            if type(seq_pos_col)!=type(None):
                seq_pos = seq_positions[i]

            if type(effect_region_cols)!=type(None):
                effect_region = effect_regions[i]

            variant_effects_df = self.infer_effects(chr_, pos, ref, alt,
                                                 # This way don't need to re-compute the celltype-assay combinations
                                                 # each time.
                                                 celltypes, assays,
                                                 #imputed_celltypes, imputed_assays, all_combinations=False,
                                                 index_base=index_base, correct_ref=correct_ref,
                                                 batch_size=batch_size, batch_by=batch_by,
                                                 seq_pos=seq_pos, effect_region=effect_region,
                                                 all_combinations=all_combinations,
                                                 verbose=verbose)
            variant_effects_flat = variant_effects_df.values.ravel()

            # For debugging during developement, checked and the ravelling functioning correctly.
            # for i, (celltype, assay) in enumerate(imputed_celltype_assays):
            #     if variant_effects_flat[i] != variant_effects_df.loc[celltype, assay]:
            #         raise Exception("ORDER WRONG!")

            multivariant_effects[i, :] = variant_effects_flat
            var_names.append( f"{chr_}_{pos}_{ref}_{alt}" )

            if verbose and i % math.ceil(min([100, 0.1*vcf_df.shape[0]])) == 0:
                print(f"PROCESSED {i}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        if verbose:
            print(f"PROCESSED {vcf_df.shape[0]}/{vcf_df.shape[0]} variants.", file=log_file, flush=True)

        multivariant_effects = pd.DataFrame(multivariant_effects,
                                            # Decided not to include the var names, but rather try to keep it in-line
                                            # with the input var_df, because the user could input the same variant
                                            # BUT measure the effect using different locations (e.g. exon 1 or exon 2)
                                            #index=var_names,
                                            columns=imputed_celltype_assays_str)
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

    @staticmethod
    def normalise_and_ordered_signals(ref_signals, alt_signals,
                                    ):
        """ Performs a normalisation for the predicted effects, by scaling the predictions by the maximum observed signal
            for the reference locus.

        Parameters
        ----------
        ref_signals: pd.DataFrame
            Rows are the genome bins, columns are the experiments, and values are the predicted signals.
        alt_signals: float
            Rows are the genome bins, columns are the experiments, and values are the predicted signals.

        Returns
        --------
        ref_signals_normed: pd.DataFrame
            Same as ref_signals, but values are normalised within each assay.
        alt_signals_normed: pd.DataFrame
            Same as alt_signals, but values are normalised within each assay.
        """
        if not np.all(ref_signals.index.values == alt_signals.index.values) or \
           not np.all(ref_signals.columns.values == alt_signals.columns.values):
            raise Exception("Inputted ref_signals and alt_signals are not aligned.")

        ref_maxs = ref_signals.values.max(axis=0)
        ref_signals_normed = np.apply_along_axis(np.divide, 1, ref_signals, ref_maxs)
        alt_signals_normed = np.apply_along_axis(np.divide, 1, alt_signals, ref_maxs)

        exper_order = np.argsort( -ref_signals.values.mean(axis=0) )

        rna_seq = ['RNA-seq' in assay for assay in ref_signals.columns]
        ref_signals_normed[:, rna_seq] = ref_signals_normed[:, rna_seq]**2
        alt_signals_normed[:, rna_seq] = alt_signals_normed[:, rna_seq]**2

        ref_signals_normed = pd.DataFrame(ref_signals_normed,
                                                     index=ref_signals.index.values, columns=ref_signals.columns.values)
        alt_signals_normed = pd.DataFrame(alt_signals_normed,
                                          index=ref_signals.index.values, columns=ref_signals.columns.values)

        ref_signals_normed_and_ordered = ref_signals_normed.iloc[:,exper_order].copy()
        alt_signals_normed_and_ordered = alt_signals_normed.iloc[:,exper_order].copy()

        return ref_signals_normed_and_ordered, alt_signals_normed_and_ordered

    @staticmethod
    def get_diff_signals(ref_signals_normed, alt_signals_normed, keep_rna_squared=False):
        """ Performs a normalisation for the predicted effects, by scaling the predictions by the maximum observed signal
            for the reference locus.

        Parameters
        ----------
        ref_signals_normed: pd.DataFrame
            Rows are the genome bins, columns are the experiments, and values are the normalised predicted signals.
        alt_signals_normed: float
            Rows are the genome bins, columns are the experiments, and values are the normalised predicted signals.
        keep_rna_squared: bool
            True to keep the RNA assays squared (which was performed during normalization) or take the sqrt to convert back.

        Returns
        --------
        get_diff_signals: pd.DataFrame
            Difference between alt - ref across the sequence.
        """
        index = ref_signals_normed.index.values
        cols = ref_signals_normed.columns.values
        if not np.all(alt_signals_normed.index.values == index) or \
                    not np.all(alt_signals_normed.columns.values == cols):
            raise Exception("ref_signals_normed and alt_signals_normed are not aligned. i.e. rows and columns not same.")

        ref_signals = ref_signals_normed.values.copy()
        alt_signals = alt_signals_normed.values.copy()
        rna_assays = ['RNA' in assay_ for assay_ in cols]
        if not keep_rna_squared:
            ref_signals[:, rna_assays] = np.sqrt( ref_signals[:, rna_assays] )
            alt_signals[:, rna_assays] = np.sqrt(alt_signals[:, rna_assays])

        diff_signals = pd.DataFrame(alt_signals - ref_signals,
                                    index=list(ref_signals_normed.index),
                                    columns=list(ref_signals_normed.columns)
                                    )
        return diff_signals

    @staticmethod
    def load_intersecting_annots(pred_signals_or_range, gtf_or_bed_file, gtf=True):

        if type(pred_signals_or_range) == pd.core.frame.DataFrame:

            chrom, start, bin_end = pred_signals_or_range.index.values[0].split('_')
            _, _, end = pred_signals_or_range.index.values[-1].split('_')
            start, end = int(start), int(end)

        else:
            # Assumes the input is a range in from of (chrom, start, end)
            chrom, start, end = pred_signals_or_range

        # Convert query region to a BEDTools interval
        query_region = pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True)

        # Load BED file as a BEDTools object
        bed_or_gtf = pybedtools.BedTool( gtf_or_bed_file )

        # Find overlaps
        overlapping_features = bed_or_gtf.intersect(query_region, wa=True)

        # Convert back to pandas DataFrame
        df = pd.read_csv(overlapping_features.fn, sep="\t", header=None)

        region_start_col, region_end_col = 1, 2
        if gtf:
            df['gene_names'] = df.apply(lambda x: x.iloc[8].split('gene_name')[1].split('; ')[0].strip(' "'), 1)
            df['gene_ids'] = df.apply(lambda x: x.iloc[8].split('gene_id')[1].split('; ')[0].strip(' "'), 1)

            region_start_col, region_end_col = 3, 4

        if type(pred_signals_or_range) == pd.core.frame.DataFrame:
            bin_starts = ((df.iloc[:, region_start_col].values - start) / (end - start)) * pred_signals_or_range.shape[0] # Scaled by total
            bin_starts[bin_starts < 0] = 0 # Won't plot the starts if not in range.

            bin_ends = ((df.iloc[:, region_end_col].values - start) / (end - start)) * pred_signals_or_range.shape[0] # Scaled by total
            bin_ends[bin_ends > pred_signals_or_range.shape[0]] = pred_signals_or_range.shape[0]  # Won't plot the ends if not in range.

            df['bin_start'] = bin_starts # Fraction along the output sequence..
            df['bin_end'] = bin_ends

        return df

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
