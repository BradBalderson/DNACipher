🧬DNACipher
==================
## Prediction of genomic measurements from DNA sequence in observed and unobserved cell types and assays
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_logo.png" alt="DNACipher Example" width="600">

**DNACipher is a DNA sequence deep learning model that also includes cell type and assay information on the model input.**

🦾DNACipher, a deep learning framework that integrates long-range sequence modeling (🧬) with biological context imputation (🧠🫀🫁), 
enabling variant effect prediction across 38,000+ experimental contexts🎯, 
a 7-fold improvement over previous state-of-the-art models such as Enformer📈.

🎯💥Deep Variant Impact Mapping (DVIM) is an analysis framework built on DNACipher that calls significant variant effects
at GWAS loci, to call common and rare 'impact' variants. These impact variants have the following properties:

    👉 They occur within the GWAS hit locus.
    👉 Have significant predicted molecular effects in >=1 biological contexts above non-significant common variants at a similar genome location.
    👉 They can be common or rare variants - enabling the study of rare variants previously inaccessible to classical statistical methods.

DNACipher is made available here as both a Python interface 🐍 and a command-line-interface 🖥, with tutorials below reproducing
key results from the DNACipher manuscript 📖.

 1.0 Tutorials - 🐍Python Interface🐍
-------
### The following tutorials are written in google collab, showing how to use the DNACipher Python interface.

* Tutorial 1: 📊DNACipher inference of genetic variant effects 
https://colab.research.google.com/gist/BradBalderson/c4389baa0d789314259b8479cfd35747/dnacipher_inference_local.ipynb
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_tutorial1_figure.png" alt="DNACipher Tut1" width="1000">

* Tutorial 2: 🎯💥Deep Variant Impact Mapping (DVIM) with DNACipher to infer common and rare genetic variants with significant effects at GWAS loci:
https://colab.research.google.com/drive/17GiWLt_SigpVa6hl6A9yP_edM4IcQeEy?usp=sharing
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_DVIM_example.png" alt="DNACipher Tut2" width="1000">

 2.0 Tutorials -🖥Command-line Interface🖥
-------
### DNACipher DVIM command-line-interface

DNACipher and DVIM analysis using a command-line interface, so that R
users and non-Python programmers can utilize the model and analysis.

2.1 Install 📍
-------

Please replace 'mamba' with 'conda' if not installed, mamba much faster however (recommend installing mamba!).

Expected install time is approximately 3-minutes. 

The current version has been tested with python 3.10 using the conda environment setup specified below, 
on a linux Ubuntu 22.04.3 LTS with a Nvidia A40 GPU (40Gb GPU RAM, 40Gb of CPU RAM) running Cuda 12.4 driver version 550.90.07. 

To install from source:

    mamba create -n dnac_env python=3.10
    mamba activate dnac_env
    mamba install matplotlib seaborn pandas scipy pytorch-lightning zlib ipykernel bioconda::bedtools 
    
    # ensure zlib is found
    export LDFLAGS="-L$CONDA_PREFIX/lib"
    export CPPFLAGS="-I$CONDA_PREFIX/include"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    
    pip install numba typer pyfaidx kipoiseq enformer-pytorch pybedtools

    git clone https://github.com/BradBalderson/DNACipher.git
    cd DNACipher
    pip install .

Usage 💻
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


2.2 DNACipher variant effect inference 📊
------

***Most steps below need the reference genome in order to load the sequences for variant effect inference***

    wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > hg38.fa

    mamba install bioconda::samtools
    samtools faidx hg38.fa

***2.2.1 Inferring the effects for a single variant***

The minimal inputs here are just the variant CHR, POS, REF, ALT of the variant, the celltypes and assays to infer the 
effects for, the path to the fasta file and the prefix for the output files.

    dnacipher infer-effects --help

     Usage: dnacipher infer-effects [OPTIONS] CHR_ POS REF ALT CELLTYPES ASSAYS                                                                              
                                    FASTA_FILE_PATH OUT_PREFIX                                                                                               
                                                                                                                                                             
     Infers a single varaints effects across celltypes and assays, and optionally across the sequence.                                                       
                                                                                                                                                             
    ╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    chr_                 TEXT     Chromosome of variant. [default: None] [required]                                                                  │
    │ *    pos                  INTEGER  Position of variant on chromosome. [default: None] [required]                                                      │
    │ *    ref                  TEXT     Reference allele of variant. [default: None] [required]                                                            │
    │ *    alt                  TEXT     Alternate allele of variant. [default: None] [required]                                                            │
    │ *    celltypes            TEXT     Celltypes to infer effects for. File in format: ct1,,ct2,,ct3 [default: None] [required]                           │
    │ *    assays               TEXT     Assays to infer effects for. File in format: assay1,,assay2,,assay3 [default: None] [required]                     │
    │ *    fasta_file_path      TEXT     FASTA file path for the reference genome. Must have .fai index. [default: None] [required]                         │
    │ *    out_prefix           TEXT     Prefix for all outputs files. [default: None] [required]                                                           │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │         -d,-device                                         TEXT     Device to run model on. [default: None]                                           │
    │         -i,-index_base                                     INTEGER  Whether the variant position is 0-based or 1-based indexing. [default: 0]         │
    │         -correct_ref                 -no-correct_ref                Correct the reference genome sequence if disagrees with the inputted ref allele.  │
    │                                                                     [default: no-correct_ref]                                                         │
    │         -s,-seq_pos                                        INTEGER  Where to centre the query sequence. [default: None]                               │
    │         -ers,-effect_region_start                          INTEGER  Where to start measuring the effect in the genome. [default: None]                │
    │         -ere,-effect_region_end                            INTEGER  Where to end measuring the effect in the genome. [default: None]                  │
    │         -b,-batch_size                                     INTEGER  How many effects to infer at a time. [default: None]                              │
    │         -by,-batch_by                                      TEXT     Indicates how to batch the data when fed into the model, either by 'experiment',  │
    │                                                                     'sequence', or None. If None, will automatically choose whichever is the larger   │
    │                                                                     axis.                                                                             │
    │                                                                     [default: None]                                                                   │
    │         -all_combinations            -no-all_combinations           Generate predicetions for all combinations of inputted cell types and assays.     │
    │                                                                     [default: all_combinations]                                                       │
    │         -return_all                  -no-return_all                 Return the signals across the ref and alt sequences [default: no-return_all]      │
    │         -verbose                     -quiet                         Enable or disable verbose output [default: verbose]                               │
    │ --help                                                              Show this message and exit.                                                       │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

***🏃Running example, showing the variant effect inference for a causal eQTL at the WRN gene locus:***

    out_prefix="WRN_eQTL_"
    fasta_path="hg38.fa"
    
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

    # Inferring effects for a single variant with signals along sequence outputted
    dnacipher infer-effects ${chr_} ${pos} ${ref} ${alt} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base} -s ${seq_pos} -ers ${effect_start} -ere ${effect_end} -return_all

***2.2.2 🖼Plotting results***

    dnacipher plot-signals --help
                                                                                                                                                             
     Usage: dnacipher plot-signals [OPTIONS] SIGNALS_PATH OUT_PREFIX                                                                                         
                                                                                                                                                             
     Plots DNACipher signal tracks and optional gene/cCRE annotations.                                                                                       
                                                                                                                                                             
    ╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    signals_path      TEXT  Path to the signal predictions across contexts for a given variant. [default: None] [required]                           │
    │ *    out_prefix        TEXT  Prefix for all outputs files. [default: None] [required]                                                                 │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │         -gtf,-gtf_file_path                        TEXT     Optional: GTF file path for gene annotation overlap. [default: None]                      │
    │         -cres,-encode_cres_path                    TEXT     Optional: BED file path for cCREs annotation overlap. [default: None]                     │
    │         -chr,-variant_chr                          TEXT     Chromosome of the variant (for plotting vertical line). [default: None]                   │
    │         -pos,-variant_pos                          INTEGER  Position of the variant (for plotting vertical line). [default: None]                     │
    │         -ref,-variant_ref                          TEXT     Reference allele of the variant. [default: None]                                          │
    │         -alt,-variant_alt                          TEXT     Alternate allele of the variant. [default: None]                                          │
    │         -plot_delta                -no-plot_delta           Whether to plot the difference between ref and alt signals. [default: no-plot_delta]      │
    │         -xtick_freq                                INTEGER  Spacing between x-ticks in base pairs. [default: 200]                                     │
    │         -title                                     TEXT     Plot title. [default: DNACipher track predictions]                                        │
    │         -verbose                   -quiet                   Enable or disable verbose output. [default: verbose]                                      │
    │ --help                                                      Show this message and exit.                                                               │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

The ***signals_path*** points to one of the signal files outputted from the infer-effects command above with the -return-all option,
which could be the ref signals, the alt signals, or the diff signals which refers to the difference between ALT-REF signals.

These outputted signal files have this format:

        prostate gland---plus strand polyA plus RNA-seq prostate gland---minus strand polyA plus RNA-seq
    chr8_31019494_31019622  -1.1451542e-05  1.0021031e-05
    chr8_31019622_31019750  -3.5203993e-05  -1.9833446e-05
    chr8_31019750_31019878  -8.381903e-06   6.67572e-06
    chr8_31019878_31020006  -1.9155443e-05  3.27453e-05

These are tab-separated files, with columns in format ***CELLTYPE---ASSAY***, and rows are in format ***CHR_START_END*** referring to positions in the genome
where the inferred difference occurs. The rows are at 128bp resolution bins, for 896 bins (total ~114kb, in the middle of the inputted
196kb sequence). The values in this case are ALT-REF signals.

The below code produces signal plots for the diff-signals shown above, which is also produced in Tutorial 1 of the Python API.

    # OPTIONAL annotation downloads, to make visualization more compelling
    wget https://downloads.wenglab.org/V3/GRCh38-cCREs.bed
    wget -O - https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.annotation.gtf.gz | gunzip -c > gencode.v26.annotation.gtf
    
    dnacipher plot-signals ${out_prefix}diff_signals.txt ${out_prefix} -gtf gencode.v26.annotation.gtf -cres GRCh38-cCREs.bed -chr ${chr_} -pos ${pos} -ref ${ref} -alt ${alt}
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/WRN_eQTL_signals_plot.png" alt="DNACipher Tut1" width="1000">

***2.2.3 👐Inferring effects for multiple variants outputting the summed effect across the locus***

This shows how to infer effects for multiple variants, just outputing the SIGN*SUM(ABS(ALT-REF)), where SIGN refers to 
if most positions were negative or positive along the sequence (SIGN = {-1, +1}).

The input file to this command is a VCF-like tab-separated file, that looks like this:

    CHR     POS     REF     ALT
    chr8    31119876        T       C
    chr13   50909867        G       C
    chr13   50909048        T       C
    chr14   24036431        C       T
    chr13   50909867        G       C

👉Each line is a 'query' to the DNACipher model. Can include additional columns that specify where to centre the query
sequence (otherwise defaults to centring ON the variant location), and also where to determine ALT-REF predictions. 
For example, can include an addition two columns specifying where to take the difference (SUM(ALT[start_effect:end_effect]-REF[start_effect:end_effect])).
These coordinates are given relative to the reference genome, with the chromosome assumed to be the same as the specified 
chromosome for the variant. For example, could have an input file like this:

    CHR     POS     REF     ALT SEQ_POS START_EFFECT    END_EFFECT
    chr8    31119876        T       C   31119866    31119856    31119886

In this above example, would infer the effect for the variant, with the query sequence centred at SEQ_POS, and measuring
the effect only at positions START_EFFECT:END_EFFECT. So with this input, for the same variant, we could alter SEQ_POS
and START_EFFECT and END_EFFECT in order to measure different predicted effects along the genome for the same variant.

Inferring effects for multiple variants like this is achieved with the command below:

    dnacipher infer-multivariant-effects --help
                                                                                                                                                         
     Usage: dnacipher infer-multivariant-effects [OPTIONS] VCF_PATH CELLTYPES                                                                                
                                                 ASSAYS FASTA_FILE_PATH OUT_PREFIX                                                                           
                                                                                                                                                             
     Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per variant, and predicted effect sizes across the   
     columns for all celltype/assay combinations.                                                                                                            
                                                                                                                                                             
    ╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    vcf_path             TEXT  Rows represent particular genetic variants, columns are CHR, POS, REF, ALT [default: None] [required]                 │
    │ *    celltypes            TEXT  Celltypes to infer effects for. File in format: ct1,ct2,ct3 [default: None] [required]                                │
    │ *    assays               TEXT  Assays to infer effects for. File in format: assay1,assay2,assay3 [default: None] [required]                          │
    │ *    fasta_file_path      TEXT  FASTA file path for the reference genome. Must have .fai index. [default: None] [required]                            │
    │ *    out_prefix           TEXT  Prefix for all outputs files. [default: None] [required]                                                              │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │         -d,-device                                              TEXT     Device to run model on. [default: None]                                      │
    │         -i,-index_base                                          INTEGER  Whether the variant position is 0-based or 1-based indexing. [default: 0]    │
    │         -correct_ref                      -no-correct_ref                Correct the reference genome sequence if disagrees with the inputted ref     │
    │                                                                          allele.                                                                      │
    │                                                                          [default: no-correct_ref]                                                    │
    │         -sc,-seq_pos_col                                        TEXT     Column in vcf that specifies the position to centre the query sequence on,   │
    │                                                                          must be within dnacipher.seqlen_max in order to predict effect of the        │
    │                                                                          genetic variant. If None then will centre the query sequence on the inputted │
    │                                                                          variant.                                                                     │
    │                                                                          [default: None]                                                              │
    │         -ersc,-effect_region_start_col                          TEXT     Specifies column in the inputted data frame specifying the start position    │
    │                                                                          (in genome coords) to measure the effect                                     │
    │                                                                          [default: None]                                                              │
    │         -erec,-effect_region_end_col                            TEXT     Specifies column in the inputted data frame specifying the end position (in  │
    │                                                                          genome coords) to measure the effect                                         │
    │                                                                          [default: None]                                                              │
    │         -b,-batch_size                                          INTEGER  How many effects to infer at a time. [default: None]                         │
    │         -by,-batch_by                                           TEXT     Indicates how to batch the data when fed into the model, either by           │
    │                                                                          'experiment', 'sequence', or None. If None, will automatically choose        │
    │                                                                          whichever is the larger axis.                                                │
    │                                                                          [default: None]                                                              │
    │         -all_combinations                 -no-all_combinations           Generate predicetions for all combinations of inputted cell types and        │
    │                                                                          assays.                                                                      │
    │                                                                          [default: all_combinations]                                                  │
    │         -verbose                          -quiet                         Enable or disable verbose output [default: verbose]                          │
    │ --help                                                                   Show this message and exit.                                                  │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Working example using provided tutorial data, this will take ~0.6 minutes to run.

    out_prefix="eQTLs_"
    vcf_path="tutorials/data/gtex_variants_SMALL.vcf"
    
    dnacipher infer-multivariant-effects ${vcf_path} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base}

The output file looks like this:

    prostate gland---plus strand polyA plus RNA-seq prostate gland---minus strand polyA plus RNA-seq
    0.46152464      -3.3929667
    -11.987795      30.90546
    2.1517787       3.6421638
    6.2597466       7.8572655
    -11.987795      30.90546
    6.0217276       -5.532748
    0.77098024      -1.0102781

Each row refers to a row in the ***vcf_path*** input, and each column is the ***CELLTYPE---ASSAY***, with SUM(ALT-REF)
for the given query outputted.

3.0 🎯💥Deep Variant Impact Mapping (DVIM)
------

DVIM analysis is performed one locus at a time, and assumes that the inputted VCF-like file has been
subsetted to the variants surrounding a given GWAS hit locus.

The below will reproduce the DVIM at the RUNX3 locus of T1D.

The first input file looks like this, but it is flexible of the order and naming of the columns. Required information is
the chromosome, position, ref, alt, allele_frequency, p_value. This is necessary to perform the variant stratification
to decide what are the different variant types present at the locus for DVIM.

    variant_id      p_value chromosome      base_pair_location      effect_allele   other_allele    effect_allele_frequency beta    standard_error  odds_ratio       ci_lower        ci_upper        -log10_pval
    rs6657823       0.774   1       24890003        G       A       0.138   0.00593 0.020636                                0.11125903931710739
    rs6663476       0.539   1       24890041        C       T       0.831   -0.012048       0.019613  

***3.1 🤹‍Stratifying the variants at the RUNX3 T1D GWAS loci into the the common, rare, background variants and other variants***

    dnacipher stratify-variants --help
                                                                                                                                                             
     Usage: dnacipher stratify-variants [OPTIONS] SIGNAL_GWAS_STATS_PATH                                                                                     
                                        VAR_REF_COL VAR_ALT_COL VAR_LOC_COL P_COL                                                                            
                                        ALLELE_FREQ_COL OUT_PREFIX                                                                                           
                                                                                                                                                             
     Performs stratification of variants at GWAS loci to categories:  * 'candidate' variants (common significant variants),  * 'rare' variants               
     (non-significant rare variants in the same region as the candidate variants),  * 'background' variants (common non-significant variants), and 'other'   
     variants (rare variants outside of the hit locus).                                                                                                      
                                                                                                                                                             
    ╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    signal_gwas_stats_path      TEXT  Path to GWAS summary statistics for each of the variants at a GWAS locus, within range model predictions.      │
    │                                        [default: None]                                                                                                │
    │                                        [required]                                                                                                     │
    │ *    var_ref_col                 TEXT  Column in the input dataframe that specifies the variant reference sequence as a string. [default: None]       │
    │                                        [required]                                                                                                     │
    │ *    var_alt_col                 TEXT  Column in the input dataframe that specifies the variant alternate sequence as a string. [default: None]       │
    │                                        [required]                                                                                                     │
    │ *    var_loc_col                 TEXT  Column in the input dataframe that specifies the variant position as an integer. [default: None] [required]    │
    │ *    p_col                       TEXT  Name of column in the input dataframe that contains the p-values of the variant-trait associations.            │
    │                                        [default: None]                                                                                                │
    │                                        [required]                                                                                                     │
    │ *    allele_freq_col             TEXT  Column in the input dataframe that specifies the variant allele frequency as a float. [default: None]          │
    │                                        [required]                                                                                                     │
    │ *    out_prefix                  TEXT  Prefix for all outputs files. [default: None] [required]                                                       │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │         -pc,-p_cut                       FLOAT    P-value cutoff to consider a variant significantly associated with the trait. [default: 5e-07]      │
    │         -lc,-lowsig_cut                  FLOAT    Cutoff to consider variants confidently not-significant. [default: 0.001]                           │
    │         -nt,-n_top                       INTEGER  If no significant variants, will take this many as the top candidates. [default: 10]                │
    │         -afc,-allele_freq_cut            FLOAT    If a variant is above this minor allele frequency AND is considered confidently non-significant,    │
    │                                                   then is considered a 'background' variant. If is below this allele frequency, then considered a     │
    │                                                   rare variant.                                                                                       │
    │                                                   [default: 0.05]                                                                                     │
    │         -mbv,-min_bg_variants            INTEGER  If have less than this number of background variants, will rank-order potential background variants │
    │                                                   by scoring allele frequency and significance, and take this many variants as significant.           │
    │                                                   [default: 100]                                                                                      │
    │         -verbose                 -quiet           Enable or disable verbose output [default: verbose]                                                 │
    │ --help                                            Show this message and exit.                                                                         │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Now running the example at the RUNX3 T1D locus, note that the columns with the required information are specified on
input to enable flexible input formats:

    out_prefix="dvim_runx3_"
    runx3_gwas_stats_path="tutorials/data/Chiou-2021-T1D-GWAS_RUNX3-signal_variant_stats.txt.gz"
    
    dnacipher stratify-variants ${runx3_gwas_stats_path} other_allele effect_allele base_pair_location p_value effect_allele_frequency ${out_prefix}

As output, this gives the same input file, but with an extra column 'variant_label' specifying the stratification of the variant:

    variant_id      p_value chromosome      base_pair_location      effect_allele   other_allele    effect_allele_frequency beta    standard_error  odds_ratio       ci_lower        ci_upper        -log10_pval     var_label
    rs72657048      1.11e-07        1       24963243        G       C       0.509   0.074632        0.01406                         6.954677021213342       candidate
    rs6672420       1.74e-07        1       24964519        T       A       0.483   0.07369 0.014103                                6.7594507517174005      candidate

The labels are explained in the ***dnacipher stratify-variants --help*** documentation above.

***🎨Plotting the stratifications***

Easy to check this result with the following command:

    dnacipher plot-variant-stats --help
                                                                                                                                                             
     Usage: dnacipher plot-variant-stats [OPTIONS] STRATIFIED_GWAS_STATS_PATH                                                                                
                                         Y_AXIS_COL COLOR_BY OUT_PREFIX                                                                                      
                                                                                                                                                             
     Manhattan-like plot for variant statistics.                                                                                                             
                                                                                                                                                             
    ╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    stratified_gwas_stats_path      TEXT  Path to stratified GWAS statistics file. [default: None] [required]                                        │
    │ *    y_axis_col                      TEXT  Column name to use for y-axis values (e.g., -log10_pval). [default: None] [required]                       │
    │ *    color_by                        TEXT  Column name to color points by (e.g., var_label). [default: None] [required]                               │
    │ *    out_prefix                      TEXT  Prefix for output files. [default: None] [required]                                                        │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │         -gtf,-gtf_file_path                       TEXT   Optional GTF file path for plotting gene annotations. [default: None]                        │
    │         -cmap,-color_map                          TEXT   Colormap name for continuous coloring. [default: magma]                                      │
    │         -alpha                                    FLOAT  Opacity of the scatter plot points. [default: 0.5]                                           │
    │         -order_points          -no-order_points          Whether to plot points ordered by statistic. [default: order_points]                         │
    │         -reverse_order         -no-reverse_order         Whether to reverse point order. [default: no-reverse_order]                                  │
    │         -show_legend           -no-show_legend           Whether to display the plot legend. [default: show_legend]                                   │
    │         -verbose               -quiet                    Enable or disable verbose output. [default: verbose]                                         │
    │ --help                                                   Show this message and exit.                                                                  │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

We can now plot to see if the variant stratification make sense:
 
    dnacipher plot-variant-stats -gtf gencode.v26.annotation.gtf -- ${out_prefix}stratified_gwas_stats.txt "-log10_pval" var_label ${out_prefix}
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dvim_runx3_-log10_pval_var_label_variant_stats.png" alt="DVIM RUNX3 locus" width="500">

***3.2. 🧠🫀🫁Performing the variant effect inference in the relevant cell types***

Creating the input files required for the DNACipher variant effect inference for cell types and assays.

    echo 'effector memory CD4-positive, alpha-beta T cell,,activated naive CD8-positive, alpha-beta T cell,,T-helper 17 cell,,T-cell,,CD8-positive, alpha-beta memory T cell,,effector memory CD8-positive, alpha-beta T cell,,pancreas,,peripheral blood mononuclear cell,,naive thymus-derived CD8-positive, alpha-beta T cell,,CD4-positive, alpha-beta memory T cell,,central memory CD8-positive, alpha-beta T cell,,naive thymus-derived CD4-positive, alpha-beta T cell,,body of pancreas,,CD4-positive, CD25-positive, alpha-beta regulatory T cell,,CD4-positive, alpha-beta T cell,,CD8-positive, alpha-beta T cell,,CD14-positive monocyte,,natural killer cell' > t1d_runx3_celltypes.txt
    
    echo 'ATAC-seq,,DNase-seq,,plus strand polyA minus RNA-seq,,plus strand polyA plus RNA-seq,,plus strand total RNA-seq,,signal of polyA plus RNA-seq' > t1d_runx3_assays.txt

Need to re-arrange the frame to make sure is in format CHR, POS, REF, ALT, required by ***dnacipher infer-multivariant-effects***

    awk -F'\t' 'NR==1 || $NF != "other"' ${out_prefix}stratified_gwas_stats.txt > ${out_prefix}selected_gwas_stats.txt
    
    awk 'BEGIN {OFS="\t"} {print $3, $4, $6, $5, $NF}' ${out_prefix}selected_gwas_stats.txt > ${out_prefix}selected_gwas_stats.reformatted.txt
    awk 'BEGIN {OFS=FS="\t"} NR==1 || $1 ~ /^chr/ {print $0; next} { $1 = "chr" $1; print }' ${out_prefix}selected_gwas_stats.reformatted.txt > ${out_prefix}selected_gwas_stats.reformatted.chr_named.txt

IMPORTANT need to set the seq_pos column, so that each variant is being scored consistently for DVIM and comparing variant effects in a apples-to-apples fashion.

    runx3_stats=${out_prefix}selected_gwas_stats.reformatted.chr_named.txt
    runx3_stats_dvim=${out_prefix}selected_gwas_stats.reformatted.chr_named.seq_pos.txt
    
    awk 'BEGIN {OFS=FS="\t"} NR==1 {print $0, "seq_pos"; next} {print $0, "24970252"}' ${runx3_stats} > ${runx3_stats_dvim}

Now running the dnacipher effect inference for these variants, which will then compare statistically.
~7mins run time for 543 variants and 108 contexts.

    dnacipher infer-multivariant-effects ${runx3_stats_dvim} t1d_runx3_celltypes.txt t1d_runx3_assays.txt ${fasta_path} ${out_prefix} -i 1 -seq_pos_col seq_pos

***👩‍💻Calculating p-values***

    dnacipher effect-pvals --help
                                                                                                                                                             
     Usage: dnacipher effect-pvals [OPTIONS] SELECTED_GWAS_STATS_PATH                                                                                        
                                   PRED_EFFECTS_PATH OUT_PREFIX                                                                                              
                                                                                                                                                             
     Calculates variant effect p-values for non-background variants against background variants.                                                             
                                                                                                                                                             
    ╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    selected_gwas_stats_path      TEXT  Path to GWAS summary statistics for each of the variants at a GWAS locus, with a column 'var_labels'         │
    │                                          indicating candidate, rare, and background variants. Each row is a variant.                                  │
    │                                          [default: None]                                                                                              │
    │                                          [required]                                                                                                   │
    │ *    pred_effects_path             TEXT  Path to predicted effects for each variant. Each row is a variant, and each column is a predicted molecular  │
    │                                          effect for that variant.                                                                                     │
    │                                          [default: None]                                                                                              │
    │                                          [required]                                                                                                   │
    │ *    out_prefix                    TEXT  Prefix for all outputs files. [default: None] [required]                                                     │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │         -nb,-n_boots         INTEGER  No. of boot-straps of re-selecting the background variants. [default: 10000]                                    │
    │         -pc,-p_cutoff        FLOAT    P-value below which a non-background variant predicted molecular effect is considered significantly different   │
    │                                       to the background vars.                                                                                         │
    │                                       [default: 0.05]                                                                                                 │
    │         -p,-pseudocount      INTEGER  Value added to boot-strap counts to prevent 0 p-values, defines lower-bound for minimum p-values, should be set │
    │                                       to 1.                                                                                                           │
    │                                       [default: 1]                                                                                                    │
    │         -std,-min_std        FLOAT    Minimum standard deviation for the background variant effects. Set to avoid 0 std for 0 effects of background   │
    │                                       variants causing infinite z-scores.                                                                             │
    │                                       [default: 0.01]                                                                                                 │
    │         -v,-verbosity        INTEGER  Verbosity levels. 0 errors only, 1 prints processing progress, 2 prints debugging information. [default: 1]     │
    │ --help                                Show this message and exit.                                                                                     │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

This command will perform boot-strapping of the background variant effects, and perform a one-sample z-test for each 
predicted effect of the candidate common and rare variants to determine p-values with the measured
mean and std of the background variant effects as the null distribution. 
Very fast due to Numba just-in-time (JIT) C-compilation (~0.1min for 10,000 boot-straps).

    runx3_effects=${out_prefix}var_context_effects.txt
    
    dnacipher effect-pvals ${runx3_stats} ${runx3_effects} ${out_prefix}

The output files include {out_prefix}_boot_pvals.txt and {out_prefix}_boot_counts.txt. Former is the p-values of 
the predicted molecular effect significance. Output looks like this:

    effector memory CD4-positive, alpha-beta T cell---ATAC-seq      effector memory CD4-positive, alpha-beta T cell---DNase-seq
    1.0     1.0
    0.9998000199980002      0.9998000199980002

Rows correspond to a variant, and the columns correspond to the context in format ***CELLTYPE---ASSAY***, and the values
are the p-values for the inferred effect being significantly different from the background variants.

***🎯Calling significant effects***

Now we can call the 'impact'💥 variants, setting our desired fold-change and p-value cutoff!

    dnacipher impact-map --help
                                                                                                                                                             
     Usage: dnacipher impact-map [OPTIONS] SELECTED_GWAS_STATS_PATH                                                                                          
                                 PRED_EFFECTS_PATH BOOT_PVALS_PATH OUT_PREFIX                                                                                
                                                                                                                                                             
     Calls 'impact' variants - variants with significant predicted effects in particular cell types / assays compared with background variants.              
                                                                                                                                                             
    ╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    selected_gwas_stats_path      TEXT  Path to GWAS summary statistics for each of the variants at a GWAS locus, with a column 'var_labels'         │
    │                                          indicating candidate, rare, and background variants. Each row is a variant.                                  │
    │                                          [default: None]                                                                                              │
    │                                          [required]                                                                                                   │
    │ *    pred_effects_path             TEXT  Path to predicted effects for each variant. Each row is a variant, and each column is a predicted molecular  │
    │                                          effect for that variant.                                                                                     │
    │                                          [default: None]                                                                                              │
    │                                          [required]                                                                                                   │
    │ *    boot_pvals_path               TEXT  Path to predicted effects for each variant. Each row is a variant, and each column is a predicted molecular  │
    │                                          effect for that variant.                                                                                     │
    │                                          [default: None]                                                                                              │
    │                                          [required]                                                                                                   │
    │ *    out_prefix                    TEXT  Prefix for all outputs files. [default: None] [required]                                                     │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │         -pc,-p_cutoff             FLOAT  P-value below which a non-background variant predicted molecular effect is considered significantly          │
    │                                          different to the background vars.                                                                            │
    │                                          [default: 0.05]                                                                                              │
    │         -fc,-fc_cutoff            FLOAT  Fold-change cutoff to be considered significant. [default: 0]                                                │
    │         -verbose          -quiet         Enable or disable verbose output [default: verbose]                                                          │
    │ --help                                   Show this message and exit.                                                                                  │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Now for the RUNX3 example, inputting the original stats file, the predicted effects for each variant, and p-values.
Run time ~10secs.

    runx3_pvals=${out_prefix}boot_pvals.txt
    
    dnacipher impact-map ${runx3_stats} ${runx3_effects} ${runx3_pvals} ${out_prefix} -fc 2

Most important output is {runx3_stats}.impact_calls.txt, which specifies which of the variants are 'impact' variants and
the number of significant effects they have:

    chromosome      base_pair_location      other_allele    effect_allele   var_label       n_sig_effects   impact_variant
    chr1    24963243        C       G       candidate       0       False
    chr1    24964519        A       T       candidate       2       True
    chr1    24966177        C       T       candidate       18      True

***👩‍🎨Plotting the results***

Can now-replot the variant stats, this time with the n_sig_effects to show the impact variants!

    dnacipher plot-variant-stats -gtf gencode.v26.annotation.gtf -- ${out_prefix}selected_gwas_stats.reformatted.chr_named.impact_calls.txt n_sig_effects var_label ${out_prefix}
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dvim_runx3_n_sig_effects_var_label_variant_stats.png" alt="DVIM RUNX3 locus" width="500">

Also volcano plots to assess the cutoffs used to call the impact variants:

    dnacipher plot-volcano candidate ${runx3_stats} ${out_prefix}sig_effects.txt ${out_prefix}fold_changes.txt ${runx3_pvals} ${out_prefix}
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dvim_runx3_candidate_volcano.png" alt="DVIM RUNX3 locus" width="500">

    dnacipher plot-volcano rare ${runx3_stats} ${out_prefix}sig_effects.txt ${out_prefix}fold_changes.txt ${runx3_pvals} ${out_prefix}
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dvim_runx3_rare_volcano.png" alt="DVIM RUNX3 locus" width="500">

Citation 🙇‍
--------

Coming soon..

Contact ☎
-------

Authors: Brad Balderson

Contact:  bbalderson@salk.edu








    


