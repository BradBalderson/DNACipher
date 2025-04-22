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
gtf_file_path = f'{data_path}gencode.v26.annotation.gtf'
encode_cres_path = f'{data_path}GRCh38-cCREs.bed'
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
#### Downloading reference genome
# !mkdir -p {data_path}
# !wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > {fasta_file_path}
# pyfaidx.Faidx(fasta_file_path)
# !ls {data_path}

#### Downloading the GTF of the gene annotations
# !wget -O - https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.annotation.gtf.gz | gunzip -c > {gtf_file_path}

#### Downloading ENCODE candidate cis-regulatory elements bed file
# !wget https://downloads.wenglab.org/V3/GRCh38-cCREs.bed -O {encode_cres_path}

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

# strand_effects = dnacipher.infer_effects(chr_, pos, ref, alt, celltype, assays,
#                                               # The positions are 1-based indexing, so have to indicate this
#                                               index_base=1,
#                                               # Specify a different sequence midpoint than the variant position,
#                                               # defaults to centring on variant otherwise.
#                                               seq_pos=seq_pos,
#                                              )
#
# ### Strand-specific and celltype-specific RNA-seq effect predictions, across the whole inputted locus
# print( strand_effects )

### Can also calculate the effects localised to a particular region, for example if we want to focus on local effects
# around the variant or focus on effects on a particular gene / exon / regulatory region
effect_region = (pos-100, pos+100) # chr already implied
# strand_effects_local = dnacipher.infer_effects(chr_, pos, ref, alt, celltype, assays,
#                                               # The positions are 1-based indexing, so have to indicate this
#                                               index_base=1,
#                                               # Specify a different sequence midpoint than the variant position,
#                                               # defaults to centring on variant otherwise.
#                                               seq_pos=seq_pos, effect_region=effect_region,
#                                              )
# print(strand_effects_local)

###### You can also see the effects along the sequence.
# strand_effects, ref_signals, alt_signals, ref_seq, alt_seq, ref_features, alt_features = \
#                                     dnacipher.infer_effects(chr_, pos, ref, alt, celltype, assays,
#                                               # The positions are 1-based indexing, so have to indicate this
#                                               index_base=1, return_all=True, seq_pos=seq_pos,
#                                              )

##### Visualising these strand effects
# Can load intersecting genes with this region for visualisation.

# Can load intersecting regions
# genes = dnacipher.load_intersecting_annots(ref_signals, gtf_file_path)
# cres = dnacipher.load_intersecting_annots(ref_signals, encode_cres_path, gtf=False)
#
# # Normalising so it is clearer to visualise multiple assays at once.
# ref_signals_normed_and_ordered, alt_signals_normed_and_ordered = dnacipher.normalise_and_ordered_signals(ref_signals,
#                                                                                                          alt_signals
#                                                                                                          )
#
# # Plotting the normalized and ranked tracks
# dnapl.plot_signals(ref_signals_normed_and_ordered, title="DNACipher REF seq predictions", genes=genes, regions=cres)
# dnapl.plot_signals(alt_signals_normed_and_ordered, title="DNACipher ALT seq predictions", genes=genes, regions=cres)
#
# variant_loc = (chr_, pos, ref, alt)
# diff_signals = dnacipher.get_diff_signals(ref_signals_normed_and_ordered, alt_signals_normed_and_ordered)
# dnapl.plot_signals(diff_signals, plot_delta=True, variant_loc=variant_loc,
#                    genes=genes, regions=cres,
#                    title="DNACipher alt-ref sequence predictions"
#                    )
# NOTE the delta value shown as a percentage, refers to the percentage of the maximum signal seen across the region,
# it is not relative to the total signal across the region. In other words, across the whole region we see a net
# increase of 10% of the maximum signal experession for the plus-strand, and negligible change for the minus strand.

########################################################################################################################
                # We can also infer effects across ALL cell type and assay combinations  #
########################################################################################################################
# I have implemented a batched generation of predictions, so can go through and predict the full matrix of effects in
# a batched fashion, since otherwise likely to run out of GPU memory.
# pred_effects = dnacipher.infer_effects(chr_, pos, ref, alt, dnacipher.celltypes, dnacipher.assays,
#                                        batch_by='experiment',
#                                        batch_size=12, # Impute this number of experiments per batch.
#                                        # The positions are 1-based indexing, so have to indicate this
#                                        index_base=1,
#                                        seq_pos=seq_pos,
#                                        verbose=True,
#                                              )
#
# pred_effects_normed_and_ordered = dnacipher.normalise_and_order_effects(pred_effects)
#
# fig, ax = plt.subplots(figsize=(8,8))
# sb.heatmap(pred_effects_normed_and_ordered, cmap='PiYG', vmin=-2, vmax=2, #ax=ax
#               )
# plt.subplots_adjust(bottom=0.35, top=.95, left=0.4, right=1)
# plt.show()

########################################################################################################################
                # We can also infer effects across ALL cell type and assay combinations  #
########################################################################################################################
#?dnacipher.infer_multivariant_effects
var_df.head(3)

### Can add a column representing the position we want to centre the sequences, if different from the variant position.
### Otherwise will default to centring the sequence on the variant.
tss_positions = (var_df['POS'].values - var_df['TSS_DIST'].values).astype(int)

var_df['seq_pos'] = np.array([var_df['POS'].values, tss_positions]).mean(axis=0).astype( int )

# Can also load in the gene locations, so that we are just measuring effects for the associated eGene of each eQTL,
# or for whatever feature you like at each given variant.
gtf_df = pd.read_csv(gtf_file_path, sep="\t", header=None, comment='#')
gtf_df = gtf_df.loc[gtf_df[2].values=='gene', :]
gtf_df['gene_names'] = gtf_df.apply(lambda x: x.iloc[8].split('gene_name')[1].split('; ')[0].strip(' "'), 1)
gtf_df['gene_ids'] = gtf_df.apply(lambda x: x.iloc[8].split('gene_id')[1].split('; ')[0].strip(' "'), 1)
gene_ids = gtf_df['gene_ids'].values

# Getting the annotation locations of the eGenes associated with each eQTL.
# egene_row_indices = [np.where(gene_ids==id_)[0][0] for id_ in var_df['GENE_ID']]
# egenes_gtf = gtf_df.iloc[egene_row_indices, :]
#
# var_df['egene_start'] = egenes_gtf[3].values
# var_df['egene_end'] = egenes_gtf[4].values
#
# # NOTE that any of these regions that fully lie outside of the sequence input (seq_pos +/- 98,304bp) will raise an error!
#
# # Put in all celltypes, assays to get all combinations
# var_pred_effects = dnacipher.infer_multivariant_effects(var_df.head(3),
#                                                         dnacipher.celltypes[0:5], dnacipher.assays[0:5],
#                                              seq_pos_col='seq_pos', effect_region_cols=['egene_start', 'egene_end'],
#                                              verbose=True, batch_size=15, index_base=1
#                                              )
# var_pred_effects.shape, var_pred_effects.iloc[:, 0:10]
#
# ### You can reshape to then show the effects of a particular variant:
# var0_pred_effects = var_pred_effects.values[2,:].reshape(len(dnacipher.celltypes), len(dnacipher.assays))
# var0_pred_effects = pd.DataFrame(var0_pred_effects, index=dnacipher.celltypes, columns=dnacipher.assays)
#
# var0_pred_effects_normed_and_ordered = dnacipher.normalise_and_order_effects(var0_pred_effects)
#
# fig, ax = plt.subplots(figsize=(8,8))
# sb.heatmap(var0_pred_effects_normed_and_ordered, cmap='PiYG', vmin=-2, vmax=2, #ax=ax
#               )
# plt.subplots_adjust(bottom=0.35, top=.95, left=0.4, right=1)
# plt.show()

########################################################################################################################
              # Functional fine-mapping - predicting if, where, and what effect of genetic variants   #
########################################################################################################################
# Loading a locus with high LD that has association to a particular phenotype..
#
# Download from Chiou 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC10560508/#S30

# Fine-mapping variants surrounding BCL11A locus
gene_ = 'RUNX3' #'BCL11A' # BCL11A associated locus outide of the field-of-view!
gene_entry = gtf_df.loc[gtf_df['gene_names'].values == gene_, :]
gene_range = list(gene_entry[[0,3,4,6]].values[0,:])
tss = gene_range[1] if gene_range[3]=="+" else gene_range[2] # Taking start position depending on strand
gene_midpoint = np.mean(gene_range[1:3])

# Defining a range to take variants from the file
#variants_range = (tss-(dnacipher.seq_pred_range//2), tss+(dnacipher.seq_pred_range//2))
var_pos = 24970252
locus_midpoint = var_pos #24970252 #60409281 #60633463 # Think these is hg19 coords
variants_range = (locus_midpoint-round(dnacipher.seq_pred_range*.7), locus_midpoint+round(dnacipher.seq_pred_range*.2))
#variants_range = (locus_midpoint-1000, locus_midpoint+1000)

var_dir = '/Users/bradbalderson/Desktop/projects/MRFF/data/variant/'
# Download from here:
# https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90014001-GCST90015000/GCST90014023/
# then
# gzip -d -c GCST90014023_buildGRCh38.tsv.gz | awk 'NR==1 || $3 == 2' | gzip -c > GCST90014023_buildGRCh38.tsv.chr2.gz
# chr2_gwas_stats = pd.read_csv(f"{var_dir}GCST90014023_buildGRCh38.tsv.chr2.gz", sep='\t')

# Turns out need to use the harmonize version for the meta analysis from here:
# https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90014001-GCST90015000/GCST90014023/harmonised/
# then
# gzip -d -c 34012112-GCST90014023-EFO_0001359-Build38.f.tsv.gz | awk 'NR==1 || $3 == 1' | gzip -c > 34012112-GCST90014023-EFO_0001359-Build38.f.tsv.chr1.gz
chr_gwas_stats = pd.read_csv(f"{var_dir}34012112-GCST90014023-EFO_0001359-Build38.f.tsv.{gene_range[0]}.gz", sep='\t')
# gzip -d -c 34012112-GCST90014023-EFO_0001359-Build38.f.tsv.gz | awk 'NR==1 || $3 == 2' | gzip -c > 34012112-GCST90014023-EFO_0001359-Build38.f.tsv.chr2.gz
#chr2_gwas_stats = pd.read_csv(f"{var_dir}34012112-GCST90014023-EFO_0001359-Build38.f.tsv.chr2.gz", sep='\t')

#### Loading all the variants and saving as a .bedGraph so I can seee if there is a better locus I should look at that
#### has signals that are within-range of the gene of interest.
# gwas_stats = pd.read_csv(f"{var_dir}34012112-GCST90014023-EFO_0001359-Build38.f.tsv.gz", sep='\t')
# gwas_stats['-log10_pval'] = -np.log10( gwas_stats['p_value'].values+1e-30 )
# gwas_stats.to_csv(f"{var_dir}34012112-GCST90014023-EFO_0001359-Build38.f.tsv.log10.gz", compression='gzip')
# Converting to .bedGraph..
# gzip -d -c 34012112-GCST90014023-EFO_0001359-Build38.f.tsv.log10.gz | cut -d',' -f4,5,14 | awk -F',' 'NR>1 {print "chr"$1, $2, $2+1, $3}' OFS='\t' > t1_gwas_log1pvals.bedGraph

gene_gwas_variants_bool = np.logical_and(chr_gwas_stats['base_pair_location'].values > variants_range[0],
                                         chr_gwas_stats['base_pair_location'].values < variants_range[1])

gene_gwas_stats = chr_gwas_stats.loc[gene_gwas_variants_bool, :].copy()
gene_gwas_stats['-log10_pval'] = -np.log10( gene_gwas_stats['p_value'].values )
gene_gwas_stats = gene_gwas_stats.iloc[ np.argsort(gene_gwas_stats['base_pair_location'].values), :]

### GREAT! This looks like a very good locus to study.
# Let's select variants as the candidate causal variants as those that are significant, and those
# that are as close as possible but NOT associated as the background variants.
p_cut = 5e-8 #5e-7 Good cutoff for real analysis, but too slow so make it more stringent to just do a few.
lowsig_cut = .9 #.1 Good cutoff for decent number of variants, but is too slow so will just try a few...
sig_var_bool = gene_gwas_stats['p_value'] < p_cut
sig_locs = gene_gwas_stats['base_pair_location'].values[sig_var_bool]
sig_range = (np.min(sig_locs), np.max(sig_locs))

lowsig_var_bool = gene_gwas_stats['p_value'] > lowsig_cut

locs = gene_gwas_stats['base_pair_location'].values
lowsig_var_bool = np.logical_and(lowsig_var_bool,
                                 locs >= sig_range[0])
lowsig_var_bool = np.logical_and(lowsig_var_bool,
                                 locs <= sig_range[1])

### Remove indels
other_allele = [len(allele) > 1 for allele in gene_gwas_stats['other_allele']]
effect_allele = [len(allele) > 1 for allele in gene_gwas_stats['effect_allele']]
remove_vars = np.logical_or(effect_allele, other_allele)

sig_var_bool[remove_vars] = False
lowsig_var_bool[remove_vars] = False
other_var_bool = np.logical_and(~lowsig_var_bool, ~sig_var_bool)

print('No. candidate variants:', sum(sig_var_bool))
print('No. background variants:', sum(lowsig_var_bool))

lowsig_gwas_stats = gene_gwas_stats.loc[lowsig_var_bool, :].copy()
candidate_gwas_stats = gene_gwas_stats.loc[sig_var_bool, :].copy()
other_gwas_stats = gene_gwas_stats.loc[other_var_bool, :].copy()

plt.scatter(other_gwas_stats['base_pair_location'].values, other_gwas_stats['-log10_pval'].values,
            c='orange')
plt.scatter(candidate_gwas_stats['base_pair_location'].values,
            candidate_gwas_stats['-log10_pval'].values,
            c='red')
plt.scatter(lowsig_gwas_stats['base_pair_location'].values,
            lowsig_gwas_stats['-log10_pval'].values,
            c='black')
ylims = plt.ylim()
midpoint = np.mean(ylims)
plt.hlines(midpoint, gene_range[1], gene_range[2], color='k')
plt.vlines(var_pos, 0, ylims[1]*.75, color='red')
plt.show()

##### Now let's see what we can do!
candidate_gwas_stats['var_label'] = 'candidate'
lowsig_gwas_stats['var_label'] = 'background'

selected_gwas_stats = pd.concat([candidate_gwas_stats, lowsig_gwas_stats])

##### Reformatting these to the required input for DNACipher
selected_df = selected_gwas_stats[['chromosome', 'base_pair_location', 'other_allele', 'effect_allele', 'var_label']].copy()
selected_df.index = selected_gwas_stats['variant_id'].values.astype( str )
selected_df['chromosome'] = [f"chr{chr_}" for chr_ in selected_df['chromosome']]
# Will centre the sequence to the same position for each variant
selected_df['seq_pos'] = var_pos

# Will use the whole locus to test effects, so no need to set the
runx3_celltypes_chiou = ['monocyte', 'dendritic', 'T', 'NK', 'Natural Killer']
runx3_celltypes_dnacipher = [ct for ct in dnacipher.celltypes
                             if np.any([ct in chiou_ct for chiou_ct in runx3_celltypes_chiou])]
runx3_celltypes_dnacipher = list(set(['peripheral blood mononuclear cell', 'natural killer cell', 'natural killer cell',
                             'naive thymus-derived CD8-positive, alpha-beta T cell',
                             'naive thymus-derived CD4-positive, alpha-beta T cell',
                             'pancreas', 'effector memory CD8-positive, alpha-beta T cell',
                             'effector memory CD4-positive, alpha-beta T cell',
                             'central memory CD8-positive, alpha-beta T cell',
                             'body of pancreas', 'activated naive CD8-positive, alpha-beta T cell',
                             'T-helper 17 cell', 'T-cell',  'CD14-positive monocyte',
                             'CD4-positive, CD25-positive, alpha-beta regulatory T cell',
                             'CD4-positive, alpha-beta T cell', 'CD4-positive, alpha-beta memory T cell',
                             'CD8-positive, alpha-beta T cell', 'CD8-positive, alpha-beta memory T cell']))

# For testing the current method (before going to GPU) will only try a couple of promising cell types.
test_assays = [assay for assay in dnacipher.assays if not ('minus strand' in assay or 'ChIP-seq' in assay)]

# selected_pred_effects = dnacipher.infer_multivariant_effects(selected_df,
#                                                         runx3_celltypes_dnacipher, test_assays,
#                                              seq_pos_col='seq_pos',
#                                              verbose=True, #batch_size=15,
#                                              index_base=1
#                                              )
# Saving to analyse later..
save_dir = '/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/validation/functional_finemapping/'
# selected_pred_effects.to_csv(f"{save_dir}selected_pred_effects_RUNX3_chiou_locus.txt", sep='\t')

# TODO now try to do some kind of statistical analysis to identify which of these is the causal variant!!!
selected_pred_effects = pd.read_csv(f"{save_dir}selected_pred_effects_RUNX3_chiou_locus.txt",
                                    sep='\t', index_col=0)

print("here")

# Testing if there is any statistical difference between the causal- and non-causal effects
bg_effects = selected_pred_effects.values[selected_df['var_label'].values=='background', :]
candidate_effects = selected_pred_effects.values[selected_df['var_label'].values=='candidate', :]

bg_abs_effects = np.abs( bg_effects )
candidate_abs_effects = np.abs( candidate_effects )

import scipy.stats
plt.hist(bg_abs_effects[:,3], bins=100)
plt.show()

#### Getting the t-stats for each observation in the candidate causal variants
means = np.mean(bg_abs_effects, axis=0)
stderr = scipy.stats.sem(bg_abs_effects, axis=0)

t_stats, pvals = np.zeros(candidate_abs_effects.shape), np.zeros(candidate_abs_effects.shape)
for coli in range(bg_abs_effects.shape[1]):
    n = bg_abs_effects.shape[0]
    t_stat = (candidate_abs_effects[:,coli]-means[coli])/stderr[coli] # t-statistic for mean
    pval = scipy.stats.t.sf(np.abs(t_stat), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)

    t_stats[:,coli] = t_stat
    pvals[:,coli] = pval

padjs = pvals * (pvals.shape[0]*pvals.shape[1])
padjs[padjs>1] = 1

####### Let's look at which of these look significant...
fcs = np.apply_along_axis(np.divide, 1, candidate_effects, means)

plt.scatter(fcs.ravel(), -np.log10(pvals.ravel()))
plt.xlabel('fold-change')
plt.ylabel('-log10(p_val)')
plt.show()

#### Perhaps could plot the count of the number of significant effects per variant.
var_count = (padjs < .01).sum(axis=1)

# Have clear winners for the causal variants, let's see what they look like...
candidates_df = selected_df.loc[selected_df['var_label'].values=='candidate', :]
causal_variants = candidates_df.loc[var_count==108, :] # ONE OF THESE IS THE ACTUAL FINE_MAPPED CAUSAL VARIANT!

# COOL so let's see if we can plot this on the same LD-plot, so can visualise the effects to show the fine-mapping
order = np.argsort(var_count)

plt.scatter(other_gwas_stats['base_pair_location'].values, other_gwas_stats['-log10_pval'].values,
            c='grey', alpha=.5)
plt.scatter(candidate_gwas_stats['base_pair_location'].values[order],
            candidate_gwas_stats['-log10_pval'].values[order], c=var_count[order], cmap='viridis')
plt.scatter(lowsig_gwas_stats['base_pair_location'].values,
            lowsig_gwas_stats['-log10_pval'].values,
            c='black')
ylims = plt.ylim()
midpoint = np.mean(ylims)
plt.hlines(midpoint, gene_range[1], gene_range[2], color='k')
plt.vlines(var_pos, 0, ylims[1]*.75, color='red')
plt.show()





