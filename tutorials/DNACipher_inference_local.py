########################################################################################################################
                                                    # Environment Setup #
########################################################################################################################
work_dir = '/Users/bradbalderson/Desktop/projects/myPython/DNACipher/'
import os, sys
os.chdir(work_dir)

# Setting this, so that below if component not implemented in Pytorch can fall back to CPU.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

##### Importing dependencies
import sys

import torch

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

import importlib as imp

import pyfaidx

import dnacipher.dna_cipher_infer as dnaci
import dnacipher.dna_cipher_model as dnacm
import dnacipher.dna_cipher_plotting as dnapl

#### Paths
data_path = 'tutorials/data/'
weights_path = 'weights/TRAINING_DNACV5_MID-AVG-GENOME_ORIG-ALLOC_ENFORMER0_FINETUNE_STRATMSE_model_weights.pth'
fasta_file_path = f'{data_path}genome.fa'
gtex_vars_path = f'{data_path}gtex_variants_SMALL.vcf'
sample_file_path = f'weights/encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv'

########################################################################################################################
                                                    # Detecting GPU #
########################################################################################################################
if torch.cuda.is_available():
    device = 'cuda:0'
    print("Will use cuda GPU")

elif torch.backends.mps.is_available():
    # device = 'mps:0' # Currently cannot make long-sequence inference with mps:0 due to this:
    # print("Will use apple metal GPU")
    # https://github.com/pytorch/pytorch/issues/134416
    # Should be fixed in future version
    device = 'cpu'
    print('Using CPU, mps available but currently not working for long-sequence inference.')

else:
    device = 'cpu'
    print("No apparent GPU available, using CPU (will be slow).")

device

########################################################################################################################
                                            # Loading the pretrained model #
########################################################################################################################
# Some extra parameters about the model which cannot be read from the weights:
config = {'activation_function': 'gelu',
          'relu_genome_layer': True, # Is actually gelu, this just means to use activate function for genome layer.
          'layer_norm': True,
          'n_token_features': 3072,
          'relu_output': True,
          'epi_summarise_method': 'flatten',
         }

import importlib as imp
imp.reload(dnacm)
imp.reload(dnaci)

dnacipher = dnaci.DNACipher(weight_path=weights_path, sample_file=sample_file_path, config=config,
                            genome_file=fasta_file_path,
                            device=device
                           )

### The attached model
dnacipher.model_

### Also Enformer, used to generate the sequence embeddings (3072 sequence embedding input)
dnacipher.transformer_model

### The cell types / tissues represented
pd.DataFrame( dnacipher.celltypes )

### The assays represented:
pd.DataFrame( dnacipher.assays )

########################################################################################################################
                                            # Data setup #
########################################################################################################################
# !mkdir -p {data_path}
# !wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > {fasta_file_path}
# pyfaidx.Faidx(fasta_file_path)
# !ls {data_path}

########################################################################################################################
                                        # Example GTEx variant data #
########################################################################################################################
var_df = pd.read_csv(gtex_vars_path, sep='\t', header=0)
var_df.shape, var_df.head(4)

########################################################################################################################
                                        # Performing inference on variants #
########################################################################################################################
# GTEX to ENCODE.
tissue_map = {
    'Adipose_Subcutaneous': ['subcutaneous abdominal adipose tissue'],
    'Adipose_Visceral_Omentum': ['omental fat pad'],
    'Adrenal_Gland': ['adrenal gland'],
    'Artery_Aorta': ['aorta', 'ascending aorta', 'thoracic aorta'],
    'Artery_Coronary': ['coronary artery'],
    'Artery_Tibial': ['tibial artery'],
    'Brain_Amygdala': ['brain'],
    'Brain_Anterior_cingulate_cortex_BA24': ['cingulate gyrus'],
    'Brain_Caudate_basal_ganglia': ['caudate nucleus'],
    'Brain_Cerebellar_Hemisphere': ['cerebellum'],
    'Brain_Cerebellum': ['cerebellum'],
    'Brain_Cortex': ['brain'],
    'Brain_Frontal_Cortex_BA9': ['dorsolateral prefrontal cortex'],
    'Brain_Hippocampus': ['layer of hippocampus'],
    'Brain_Hypothalamus': ['brain'],
    'Brain_Nucleus_accumbens_basal_ganglia': ['brain'],
    'Brain_Putamen_basal_ganglia': ['brain'],
    'Brain_Spinal_cord_cervical_c-1': ['spinal cord'],
    'Brain_Substantia_nigra': ['substantia nigra'],
    'Breast_Mammary_Tissue': ['luminal epithelial cell of mammary gland', 'breast epithelium'],
    'Cells_Cultured_fibroblasts': ['foreskin fibroblast', 'fibroblast of dermis', 'fibroblast of lung', 'fibroblast of the aortic adventitia'],
    'Colon_Sigmoid': ['sigmoid colon'],
    'Colon_Transverse': ['transverse colon'],
    'Esophagus_Gastroesophageal_Junction': ['gastroesophageal sphincter'],
    'Esophagus_Mucosa': ['esophagus squamous epithelium'],
    'Esophagus_Muscularis': ['esophagus muscularis mucosa'],
    'Heart_Atrial_Appendage': ['right atrium auricular region'],
    'Heart_Left_Ventricle': ['heart left ventricle'],
    'Kidney_Cortex': ['kidney', 'kidney epithelial cell'],
    'Liver': ['liver', 'right lobe of liver'],
    'Lung': ['lung', 'left lung', 'right lung'],
    'Muscle_Skeletal': ['skeletal muscle tissue', 'muscle of leg', 'muscle of trunk'],
    'Nerve_Tibial': ['tibial nerve'],
    'Ovary': ['ovary'],
    'Pancreas': ['pancreas', 'body of pancreas'],
    'Prostate': ['prostate gland'],
    'Skin_Not_Sun_Exposed_Suprapubic': ['suprapubic skin'],
    'Skin_Sun_Exposed_Lower_leg': ['lower leg skin'],
    'Small_Intestine_Terminal_Ileum': ['small intestine'],
    'Spleen': ['spleen'],
    'Stomach': ['stomach', 'mucosa of stomach'],
    'Testis': ['testis'],
    'Thyroid': ['thyroid gland'],
    'Uterus': ['uterus'],
    'Vagina': ['vagina'],
    'Pituitary': ['brain'], #Edited from here to add in matches for some missing tissues!
    'Whole_Blood': ['peripheral blood mononuclear cell'],
    'Cells_EBV-transformed_lymphocytes': ['B cell'],
    'Minor_Salivary_Gland': ['bronchial epithelial cell']
}

########################################################################################################################
                    # Inferring variant effects in a particular cell type assay combination #
########################################################################################################################
#### Let's use the example of the WRN locus causal eQTL
chr_, pos, ref, alt = var_df.iloc[0, 0:4]
pos = int(pos)
chr_, pos, ref, alt

##### BECAUSE we also want to see the gene TSS in the prediction, instead of centring on the variant location
##### we can also specify a new seq centre, as long as it is within 196,608bp from the variant location!
tss_pos = int( pos - var_df.iloc[0, :]['TSS_DIST'] )
seq_pos = (pos+tss_pos) // 2

### Getting equivalent cell type represented in DNACipher
celltype = tissue_map[ var_df.values[0, 7] ][0]
celltype

# If wanted to input multiple cell types, you can set cell type as a list of cell types, like I'll do with the assays below.

#### Lets say we just want to look at plus and minus strand RNA-seq
assays = ['plus strand polyA plus RNA-seq', 'minus strand polyA plus RNA-seq'] # Choosing from dnacipher.assays

strand_effects, experiments_outputted = dnacipher.infer_specific_effect(chr_, pos, ref, alt, celltype, assays,
                                              # The positions are 1-based indexing, so have to indicate this
                                              index_base=1,
                                              # Specify a different sequence midpoint than the variant position,
                                              # defaults to centring on variant otherwise.
                                              seq_pos=seq_pos,
                                             )

### Strand-specific and celltype-specific RNA-seq effect predictions, across the whole inputted locus
for experiment, strand_effect in zip(experiments_outputted, strand_effects):
    print(f"Predicted effect of variant in {experiment}: {strand_effect}")

###### You can also see the actual effects along the sequence.
strand_effects, experiments_outputted, ref_signals, alt_signals, ref_seq, alt_seq, ref_features, alt_features = \
                                    dnacipher.infer_specific_effect(chr_, pos, ref, alt, celltype, assays,
                                              # The positions are 1-based indexing, so have to indicate this
                                              index_base=1, return_all=True, seq_pos=seq_pos,
                                             )

##### Visualising these strand effects
# From the 'experiments_outputted', can see that the first column corresponds to the plus strand.
ref_plus_signals = ref_signals[:,0:1]
alt_plus_signals = alt_signals[:,0:1]

ref_minus_signals = ref_signals[:,1:2]
alt_minus_signals = alt_signals[:,1:2]

# Re-scaling to the maximum value observed across the signal for clarity
max_ref_plus = ref_plus_signals.max(axis=0)
ref_plus_signals_normed = (ref_plus_signals / max_ref_plus)
alt_plus_signals_normed = (alt_plus_signals / max_ref_plus)

max_ref_minus = ref_minus_signals.max(axis=0)
ref_minus_signals_normed = (ref_minus_signals / max_ref_minus)
alt_minus_signals_normed = (alt_minus_signals / max_ref_minus)

ref_signals_normed = np.concat((ref_plus_signals_normed, ref_minus_signals_normed), axis=1)
alt_signals_normed = np.concat((alt_plus_signals_normed, alt_minus_signals_normed), axis=1)

assay_colors = {'plus strand polyA plus RNA-seq': 'magenta',
                'minus strand polyA plus RNA-seq': 'dodgerblue',
                }
assay_labels = ['plus strand polyA plus RNA-seq', 'minus strand polyA plus RNA-seq']

# Squaring since originals were log2, and can see the exons better if put to the unlogged signals.
dnapl.plot_signals(ref_signals_normed**2, assay_labels, assay_colors, show=True,
                   title="DNACipher reference sequence predictions")
dnapl.plot_signals(alt_signals_normed**2, assay_labels, assay_colors, show=True,
                   title="DNACipher alternate sequence predictions")

diff_signals = alt_signals_normed - ref_signals_normed
dnapl.plot_signals(diff_signals, assay_labels, assay_colors, show=True, plot_delta=True,
                   title="DNACipher alt-ref sequence predictions", y_step=-np.max(np.abs(diff_signals)))
# NOTE the delta value shown as a percentage, refers to the percentage of the maximum signal seen across the region,
# it is not relative to the total signal across the region. In other words, across the whole region we see a net
# increase of 10% of the maximum signal experession for the plus-strand, and negligible change for the minus strand.

########################################################################################################################
                # We can also infer effects across ALL cell type and assay combinations  #
########################################################################################################################

# If you have the memory, can predict for all available cell types and assays, by setting
# celltype=dnacipher.celltypes and assays=dnacipher.assays above, but you need ALOT of memory.

# Instead below I have implemented a batched version, so can go through and predict the full matrix of effects in
# a batched fashion, since otherwise likely to run out of GPU memory.
pred_effects = dnacipher.infer_effects(chr_, pos, ref, alt,
                                       batch_axis=1,
                                       batch_size=900, # Impute this number of experiments per batch.
                                       # The positions are 1-based indexing, so have to indicate this
                                       index_base=1,
                                       seq_pos=seq_pos,
                                       verbose=True,
                                             )

pred_effects_normed_and_ordered = dnacipher.normalise_and_order_effects(pred_effects)

fig, ax = plt.subplots(figsize=(8,8))
sb.heatmap(pred_effects_normed_and_ordered, cmap='PiYG', vmin=-2, vmax=2, #ax=ax
              )
plt.subplots_adjust(bottom=0.35, top=.95, left=0.4, right=1)
plt.show()

########################################################################################################################
                # We can also infer effects across ALL cell type and assay combinations  #
########################################################################################################################
#?dnacipher.infer_multivariant_effects

var_df.head(3)

### Can add a column representing the position we want to centre the sequences, if different from the variant position.
### Otherwise will default to centring the sequence on the variant.
tss_positions = (var_df['POS'].values - var_df['TSS_DIST'].values).astype(int)

var_df['seq_pos'] = np.array([var_df['POS'].values, tss_positions]).mean(axis=0).astype( int )

var_pred_effects = dnacipher.infer_multivariant_effects(var_df.head(3),
                                              # The positions are 1-based indexing, so have to indicate this
                                              index_base=1, seq_pos_col='seq_pos',
                                             verbose=True,
                                             )

var_pred_effects.shape, var_pred_effects.iloc[:, 0:10]

### You can reshape to then show the effects of a particular variant:
var0_pred_effects = var_pred_effects.values[2,:].reshape(len(dnacipher.celltypes), len(dnacipher.assays))
var0_pred_effects = pd.DataFrame(var0_pred_effects, index=dnacipher.celltypes, columns=dnacipher.assays)

var0_pred_effects_normed_and_ordered = dnacipher.normalise_and_order_effects(var0_pred_effects)

fig, ax = plt.subplots(figsize=(8,8))
sb.heatmap(var0_pred_effects_normed_and_ordered, cmap='PiYG', vmin=-2, vmax=2, #ax=ax
              )
plt.subplots_adjust(bottom=0.35, top=.95, left=0.4, right=1)
plt.show()


