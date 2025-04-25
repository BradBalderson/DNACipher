DNACipher
==================
## Prediction of genomic measurements from DNA sequence in observed and unobserved cell types and assays
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_logo.png" alt="DNACipher Example" width="600">

**DNACipher is a DNA sequence deep learning model that also includes cell type and assay information on the model input.**

 Tutorials - Python Interface
-------
### The following tutorials are written in google collab, showing how to use the DNACipher Python interface.

* Tutorial 1: DNACipher inference of genetic variant effects
https://colab.research.google.com/gist/BradBalderson/c4389baa0d789314259b8479cfd35747/dnacipher_inference_local.ipynb
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_tutorial1_figure.png" alt="DNACipher Tut1" width="1000">

* Tutorial 2: DeepVariantImpactMapping (DVIM) with DNACipher to infer common and rare genetic variants with significant effects at GWAS loci:
https://colab.research.google.com/drive/17GiWLt_SigpVa6hl6A9yP_edM4IcQeEy?usp=sharing
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_DVIM_example.png" alt="DNACipher Tut2" width="1000">

 Tutorials - Command-line Interface
-------
### DNACipher DVIM command-line-interface

DNACipher and DVIM analysis using a command-line interface, so that R
users and non-Python programmers can utilize the model and analysis.

Install
-------

Please replace 'mamba' with 'conda' if not installed, mamba much faster however (recommend installing mamba!).

Expected install time is approximately X-minutes. 

The current version has been tested with python 3.10 using the conda environment setup specified below, 
on system Ubuntu 22.04.3 LTS.

To install from source:

    mamba create -n dnac_env python=3.10
    mamba activate sheriff_env
    mamba install matplotlib seaborn pandas scipy pytorch-lightning ipykernel zlib bioconda::bedtools 
    pip install pyfaidx kipoiseq enformer-pytorch pybedtools

    git clone https://github.com/BradBalderson/DNACipher.git
    cd DNACipher
    pip install .

Usage
-----

    dnacipher --help
     Usage: dnacipher [OPTIONS] COMMAND [ARGS]...                                                                                                          
                                                                                                                                                       
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                             │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                      │
│ --help                        Show this message and exit.                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ infer-effects                Infers a single varaints effects across celltypes and assays, and optionally across the sequence.                      │
│ infer-multivariant-effects   Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per variant, and    │
│                              predicted effect sizes across the columns for all celltype/assay combinations.                                         │
│ stratify-variants            Performs stratification of variants at GWAS loci to categories:  * 'candidate' variants (common significant variants), │
│                              * 'rare' variants (non-significant rare variants in the same region as the candidate variants),  * 'background'        │
│                              variants (common non-significant variants), and 'other' variants (rare variants outside of the hit locus).             │
│ effect-pvals                 Calculates variant effect p-values for non-background variants against background variants.                            │
│ impact-map                   Calls 'impact' variants - variants with significant predicted effects in particular cell types / assays compared with  │
│                              background variants.                                                                                                   │
│ plot-signals                 Plots DNACipher signal tracks and optional gene/cCRE annotations.                                                      │
│ plot-variant-stats           Manhattan-like plot for variant statistics.                                                                            │
│ plot-volcano                 Volcano plot for Deep Variant Impact Mapping predicted molecular effects.                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

DNACipher variant effect inference
------

***Just inferring the effects for a single variant returning summed-effect across given region***
    out_prefix="/home/jovyan/data4/bbalderson_runAI/MRFF/data/dnacipher_test/WRN_eQTL_"
    fasta_path="/home/jovyan/data4/bbalderson_runAI/MRFF/data/enformer/genome.fa"
    
    chr_="chr8"
    pos="31119876"
    index_base="1"
    ref="T"
    alt="C"
    seq_pos="31076838"
    effect_start="31119776"
    effect_end="31119976"
    
    echo "prostate gland" > celltypes.txt
    echo "plus strand polyA plus RNA-seq,,minus strand polyA plus RNA-seq" > assays.txt

    # Scores the effects for the varaint in the inputted contexts, with scoring referring to the SUM(ALT-REF) at the indicated positions
    dnacipher infer-effects ${chr_} ${pos} ${ref} ${alt} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base} -s ${seq_pos} -ers ${effect_start} -ere ${effect_end}

***Inferring effects for a single variant with signals along sequence outputted***
python -m dnacipher.main infer-effects ${chr_} ${pos} ${ref} ${alt} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base} -s ${seq_pos} -ers ${effect_start} -ere ${effect_end} -return_all

Annotation download
wget https://downloads.wenglab.org/V3/GRCh38-cCREs.bed
wget -O - https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.annotation.gtf.gz | gunzip -c > gencode.v26.annotation.gtf

python -m dnacipher.main plot-signals ${out_prefix}diff_signals.txt ${out_prefix} -gtf gencode.v26.annotation.gtf -cres GRCh38-cCREs.bed -chr ${chr_} -pos ${pos} -ref ${ref} -alt ${alt}

***Inferring effects across variants just outputting the summed effect across locus***
out_prefix="/home/jovyan/data4/bbalderson_runAI/MRFF/data/dnacipher_test/eQTLs_"
vcf_path="/home/jovyan/data4/bbalderson_runAI/myPython/DNACipher/tutorials/data/gtex_variants_SMALL.vcf"

python -m dnacipher.main infer-multivariant-effects ${vcf_path} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base}

***Deep Variant Impact Mapping***
out_prefix="/home/jovyan/data4/bbalderson_runAI/MRFF/data/dnacipher_test/dvim_runx3_"
runx3_gwas_stats_path="/home/jovyan/data4/bbalderson_runAI/myPython/DNACipher/tutorials/data/Chiou-2021-T1D-GWAS_RUNX3-signal_variant_stats.txt.gz"

python -m dnacipher.main stratify-variants ${runx3_gwas_stats_path} other_allele effect_allele base_pair_location p_value effect_allele_frequency ${out_prefix}

python -m dnacipher.main plot-variant-stats -gtf gencode.v26.annotation.gtf -- ${out_prefix}stratified_gwas_stats.txt "-log10_pval" var_label ${out_prefix}

echo 'effector memory CD4-positive, alpha-beta T cell,,activated naive CD8-positive, alpha-beta T cell,,T-helper 17 cell,,T-cell,,CD8-positive, alpha-beta memory T cell,,effector memory CD8-positive, alpha-beta T cell,,pancreas,,peripheral blood mononuclear cell,,naive thymus-derived CD8-positive, alpha-beta T cell,,CD4-positive, alpha-beta memory T cell,,central memory CD8-positive, alpha-beta T cell,,naive thymus-derived CD4-positive, alpha-beta T cell,,body of pancreas,,CD4-positive, CD25-positive, alpha-beta regulatory T cell,,CD4-positive, alpha-beta T cell,,CD8-positive, alpha-beta T cell,,CD14-positive monocyte,,natural killer cell' > t1d_runx3_celltypes.txt

echo 'ATAC-seq,,DNase-seq,,plus strand polyA minus RNA-seq,,plus strand polyA plus RNA-seq,,plus strand total RNA-seq,,signal of polyA plus RNA-seq' > t1d_runx3_assays.txt

***Need to re-arrange the frame to make sure is in format CHR, POS, REF, ALT***

awk -F'\t' 'NR==1 || $NF != "other"' ${out_prefix}stratified_gwas_stats.txt > ${out_prefix}selected_gwas_stats.txt

awk 'BEGIN {OFS="\t"} {print $3, $4, $6, $5, $NF}' ${out_prefix}selected_gwas_stats.txt > ${out_prefix}selected_gwas_stats.reformatted.txt
awk 'BEGIN {OFS=FS="\t"} NR==1 || $1 ~ /^chr/ {print $0; next} { $1 = "chr" $1; print }' ${out_prefix}selected_gwas_stats.reformatted.txt > ${out_prefix}selected_gwas_stats.reformatted.chr_named.txt

***Need to also subset the variants to just the required variants for testing!***

awk -F'\t' 'NR==1 || $5 != "other"' ${out_prefix}stratified_gwas_stats.reformatted.chr_named.txt > ${out_prefix}selected_gwas_stats.reformatted.chr_named.txt

***IMPORTANT need to set the seq_pos column, so that each variant is being scored consistently for DVIM***
runx3_stats=${out_prefix}selected_gwas_stats.reformatted.chr_named.txt
runx3_stats_dvim=${out_prefix}selected_gwas_stats.reformatted.chr_named.seq_pos.txt

awk 'BEGIN {OFS=FS="\t"} NR==1 {print $0, "seq_pos"; next} {print $0, "24970252"}' ${runx3_stats} > ${runx3_stats_dvim}

python -m dnacipher.main infer-multivariant-effects ${runx3_stats_dvim} t1d_runx3_celltypes.txt t1d_runx3_assays.txt ${fasta_path} ${out_prefix} -i 1 -seq_pos_col seq_pos

***Calculating p-values***

runx3_effects=${out_prefix}var_context_effects.txt

python -m dnacipher.main effect-pvals ${runx3_stats} ${runx3_effects} ${out_prefix}

***Calling significant effects***

runx3_pvals=${out_prefix}boot_pvals.txt

python -m dnacipher.main impact-map ${runx3_stats} ${runx3_effects} ${runx3_pvals} ${out_prefix} -fc 2

python -m dnacipher.main plot-variant-stats -gtf gencode.v26.annotation.gtf -- ${out_prefix}selected_gwas_stats.reformatted.chr_named.impact_calls.txt n_sig_effects var_label ${out_prefix}

python -m dnacipher.main plot-volcano candidate ${runx3_stats} ${out_prefix}sig_effects.txt ${out_prefix}fold_changes.txt ${runx3_pvals} ${out_prefix}

python -m dnacipher.main plot-volcano rare ${runx3_stats} ${out_prefix}sig_effects.txt ${out_prefix}fold_changes.txt ${runx3_pvals} ${out_prefix}

# TODO now need to write the install instructions for the commandline tool, and make it so can call without the python -m!









    


