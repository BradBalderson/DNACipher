""" CLI for performing DNACipher analysis of genetic variants.
"""

from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import typer
import sys

import numpy as np
import pandas as pd

from . import command_line_helpers as clh
from .helpers import data_manage
from .signals import signal_normalization as sn, signal_processing as sp

from .embed import run_embed_gen as reg

from .train import train_model as tm
from .train import train_eval as te

from .infer import run_effect_inference as infer
from .infer import run_signal_inference as si
import dnacipher_train.infer.deep_variant_impact_mapping as dvim
import dnacipher_train.visual.dna_cipher_plotting as dnapl

app = typer.Typer(pretty_exceptions_short=False)

__version__ = '1.5.0'
__version_name__ = '🦾MetalGrip'

@app.command()
def version():
    """Prints the DNACipher version."""
    print(f"DNACipher train {__version__}: {__version_name__}", file=sys.stdout, flush=True)

@app.command()
def device(training: Annotated[
        Optional[bool],
        typer.Option(
            "--training/--inference",
            help=(
                "Whether referring to device for training or for inference."
                )
            )
        ] = True,):
    """Prints the device that is detected and will be used by DNACipher."""
    clh.get_best_device(training)

@app.command()
def precompute_embeds(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations for training, with atleast two columns CHR, START. All other columns ignored.")],
    fasta_file: Annotated[str, typer.Argument(help="Path to indexed .fasta file.")],
    model_name: Annotated[str, typer.Argument(help='Name of the model to use for precomputing the embeddings.')],
    device: Annotated[str, typer.Argument(help='Name of the device to put the model on when computing the embeddings.')],
    batch_size: Annotated[Optional[int], typer.Option("-b", "-batch_size", help=("Number of regions to embeddings for at a time."))] = 56,
    region_pad: Annotated[Optional[int], typer.Option("-r", "-region_pad",
                                                      help=("+/- base-pairs to average around the inputted regions to generate the mean signal values for the region."
                                                            "So it means every input position by the genome_file, will average signals from position-region_pad to position+region_pad."))] = 147,
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
    model_config_yaml: Annotated[Optional[str], typer.Option("-y", "-yaml", help=("Path to a ymal file, that can be read with pyyaml to create necessary inputs to initialize the particular model."))] = None,
):
    """ Precomputes the sequence embeddings at desired genome sequence locations, as mean embeddings around the input coordinates.
    """
    reg.generate_embeddings(run_name, genome_file, fasta_file, region_pad, model_name, device, batch_size, out_dir,
                            model_config_yaml=model_config_yaml)

@app.command()
def precompute_signals(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations for training, with atleast two columns CHR, START. All other columns ignored.")],
    sample_folder: Annotated[str, typer.Argument(help="Directory containing the .bigWig files, from which will read the signals.")],
    sample_file: Annotated[str, typer.Argument(help='Experiment metadata file, that specifies each of the experiments per row that need to be computed. A tab-separated metadata file, with atleast two columns; "File accession", "celltype_assay". File accession is the prefix to the .bigWig file.')],
    batch_size: Annotated[Optional[int], typer.Option("-b", "-batch_size", help=("Number of regions to compute signals for across all experiments at a time."))] = 2,
    cpu: Annotated[Optional[int], typer.Option("-c", "-cpu", help=("Number of CPUs to use for computing regions."))] = 1,
    region_pad: Annotated[Optional[int], typer.Option("-r", "-region_pad",
                                                      help=("+/- base-pairs to average around the inputted regions to generate the mean signal values for the region."
                                                            "So it means every input position by the genome_file, will average signals from position-region_pad to position+region_pad."))] = 147,
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
    method: Annotated[Optional[str], typer.Option("-m", "-method", help=("Method to compute signals. Options are 'dataloader', 'deeptools', or 'bwtool'"))] = 'bwtool',
):
    """ Precomputes the functional signals to be predicted with given input sequence, as mean signal across the input locations.
    """
    ### This version was VERY slow when I tried to apply it to the full dataset. Therefore trying a version that
    ### uses MultiBigWigSummary from DeepTools instead, to see if this can get better performance!
    sp.run_signal_precomputation(run_name, genome_file, sample_folder, sample_file, batch_size, cpu, region_pad,
                                 out_dir, method=method)

@app.command()
def split_h5(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations for training, with atleast two columns CHR, START. All other columns ignored.")],
    h5_file: Annotated[str, typer.Argument(help="Path to the h5 file to convert to a split_h5.")],
    n_splits: Annotated[int, typer.Argument(help=("Number of splits to create inside the h5 file."))],
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
):
    """ Converts a .h5 file containing a single matrix, to a split.h5 which has an index, allows for faster random indexing.
    """
    data_manage.split_h5(run_name, genome_file, h5_file, n_splits, out_dir)

@app.command()
def fit_norm(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations. with atleast three columns CHR, POS, ALLOC. The column names are ignored, so can be named anything, assumes first 3 columns contain this information. ALLOC specifices 'train' and 'test' sequences.")],
    sample_file: Annotated[str, typer.Argument(help='Experiment metadata file, that specifies each of the experiments per row that need to be computed. A tab-separated metadata file, with atleast two columns; "File accession", "celltype_assay".')],
    signal_file: Annotated[str, typer.Argument(help='h5 file containing the matrix, that has the regions on the rows and the experiments on the columns, with values populated with the signal values.')],
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
):
    """ Fits a normalization to the data - so that the mean signal values across assays are scaled to the median observed value.
        Because the data is very sparse, this mean signal is relative to the non-zero entries in the dataset, and also the
        normalization is only applied to the non-zero entries.
        RNA-seq is dealt with in a specific way, because of the different representation of this assay.
    """
    sn.fit_signal_normalization(run_name, genome_file, sample_file, signal_file, out_dir)

@app.command()
def apply_norm(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations. with atleast three columns CHR, POS, ALLOC. The column names are ignored, so can be named anything, assumes first 3 columns contain this information. ALLOC specifices 'train' and 'test' sequences.")],
    sample_file: Annotated[str, typer.Argument(help='Experiment metadata file, that specifies each of the experiments per row that need to be computed. A tab-separated metadata file, with atleast two columns; "File accession", "celltype_assay".')],
    signal_file: Annotated[str, typer.Argument(help='h5 file containing the matrix, that has the regions on the rows and the experiments on the columns, with values populated with the signal values.')],
    norm_file: Annotated[str, typer.Argument(help='.pkl file that was outputted from running dnacipher_train fit-norm.')],
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
):
    """ Applies a pre-fitted normalization to the inputted data.
    """
    sn.apply_signal_normalization(run_name, genome_file, sample_file, signal_file, norm_file, out_dir)

@app.command()
def norm(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations. with atleast three columns CHR, POS, ALLOC. The column names are ignored, so can be named anything, assumes first 3 columns contain this information. ALLOC specifices 'train' and 'test' sequences.")],
    sample_file: Annotated[str, typer.Argument(help='Experiment metadata file, that specifies each of the experiments per row that need to be computed. A tab-separated metadata file, with atleast two columns; "File accession", "celltype_assay".')],
    signal_file: Annotated[str, typer.Argument(help='h5 file containing the matrix, that has the regions on the rows and the experiments on the columns, with values populated with the signal values.')],
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
):
    """ Performs both the fitting and applying of the normalization.
    """
    sn.fit_signal_normalization(run_name, genome_file, sample_file, signal_file, out_dir)

    norm_file = out_dir + f'{run_name}_norm_stats.pkl'
    sn.apply_signal_normalization(run_name, genome_file, sample_file, signal_file, norm_file, out_dir)

@app.command()
def train(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations. with atleast three columns CHR, POS, ALLOC. The column names are ignored, so can be named anything, assumes first 3 columns contain this information. ALLOC specifices 'train' and 'test' sequences.")],
    sample_file: Annotated[str, typer.Argument(help='Experiment metadata file, that specifies each of the experiments per row that need to be computed. A tab-separated metadata file, with atleast two columns; "File accession", "celltype_assay".')],
    signal_file: Annotated[str, typer.Argument(help='h5 file containing the matrix, that has the regions on the rows and the experiments on the columns, with values populated with the signal values.')],
    embed_file: Annotated[str, typer.Argument(help='h5 file containing the matrix, that has the regions on the rows and the sequence embedding features on the columns.')],
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
    model_config_file: Annotated[Optional[str], typer.Option("-m", "-model_config", help=(".yaml file specifying the model parameters, such as layer sizes, number of layers, etc. If not provided, defaults to an internal config specifying the original model."))] = None,
    batch_size: Annotated[Optional[int], typer.Option("-b", "-batch_size", help=("Number of regions to train on for each batch."))] = 256,
    cpu: Annotated[Optional[int], typer.Option("-c", "-cpu", help=("Number of cpus to use for loading batches."))] = 1,
    device: Annotated[Optional[str], typer.Option("-d", "-device", help=("Device to train model. e.g. cuda:0, mps:0, cpu. If not specified will detect the available GPU."))] = None,
    epochs_warmup: Annotated[Optional[int], typer.Option("-ew", "-epochs_warmup", help=("Number of epochs for training the model WITHOUT relu on output or weighting the different experiments."))] = 25,
    epochs_final: Annotated[Optional[int], typer.Option("-ef", "-epochs_final", help=("Number of epochs for training the model WITH relu on output & weighting the different experiments."))] = 25,
    eval_batches: Annotated[Optional[int], typer.Option("-eb", "-eval_batches", help=( "Number of batches for evaluating the final trained model."))] = 50,
    learning_rate: Annotated[Optional[float], typer.Option("-lr", "-learning_rate", help=("Learning rate for the model on each batch."))] = 0.0014526,
    dropout_rate: Annotated[Optional[float], typer.Option("-dr", "-dropout_rate", help=("Dropout rate for regularisation the model during training."))] = 0.05,
    n_regions: Annotated[Optional[int], typer.Option("-nr", "-n_regions", help=("Number of regions to use for train and test, useful for testing/debuggin."))] = None,
):
    """ Trains the DNACipher model.
    """
    if type(device) == type(None):
        device = clh.get_best_device( True )

    if model_config_file=="None" or model_config_file=="":
        model_config_file = None

    tm.train_dnacipher_model(run_name, genome_file, sample_file, signal_file, embed_file, out_dir,
                             model_config_file,
                             # train parameters, need to change depending on resources.
                             batch_size, cpu, device, epochs_warmup, epochs_final, eval_batches,
                             learning_rate=learning_rate, dropout_rate=dropout_rate, n_regions=n_regions,
                             )

@app.command()
def eval(
    run_name: Annotated[str, typer.Argument(help="Name of the precompute signal run, used to prefix output files.")],
    genome_file: Annotated[str, typer.Argument(help="Tsv file specifying the genome locations. with atleast three columns CHR, POS, ALLOC. The column names are ignored, so can be named anything, assumes first 3 columns contain this information. ALLOC specifices 'train' and 'test' sequences.")],
    sample_file: Annotated[str, typer.Argument(help='Experiment metadata file, that specifies each of the experiments per row that need to be computed. A tab-separated metadata file, with atleast two columns; "File accession", "celltype_assay".')],
    signal_file: Annotated[str, typer.Argument(help='h5 file containing the matrix, that has the regions on the rows and the experiments on the columns, with values populated with the signal values.')],
    embed_file: Annotated[str, typer.Argument(help='h5 file containing the matrix, that has the regions on the rows and the sequence embedding features on the columns.')],
    weights_path: Annotated[str, typer.Argument(help=(".pth specifying the pretrained DNACipher model weights. if 'default' inputted will use pretrained DNACipher model to reproduce key benchmarks, but this assumes the same samples used to train the original model are provided on input."))],
    out_dir: Annotated[Optional[str], typer.Option("-o", "-out", help=("Output directory."))] = '',
    ct_knn_file: Annotated[Optional[str], typer.Option("-ct_knn", "-celltype_knn", help=(".json file specifying the celltypes as keys, and a ranked list of all other cell types as values. Enables kNN benchmark, where k=10."))] = None,
    model_config_file: Annotated[Optional[str], typer.Option("-m", "-model_config", help=(".yaml file specifying the model parameters, such as layer sizes, number of layers, etc. If not provided, defaults to an internal config specifying the original model."))] = None,
    batch_size: Annotated[Optional[int], typer.Option("-b", "-batch_size", help=("Number of regions to train on for each batch."))] = 256,
    cpu: Annotated[Optional[int], typer.Option("-c", "-cpu", help=("Number of cpus to use for loading batches."))] = 1,
    device: Annotated[Optional[str], typer.Option("-d", "-device", help=("Device to train model. e.g. cuda:0, mps:0, cpu. If not specified will detect the available GPU."))] = None,
    eval_batches: Annotated[Optional[int], typer.Option("-eb", "-eval_batches", help=( "Number of batches for evaluating the final trained model."))] = 50,
):
    """ Runs evaluation on a trained the DNACipher model.
    """
    if type(device) == type(None):
        device = clh.get_best_device( True )

    if ct_knn_file == "" or ct_knn_file=='None':
        ct_knn_file = None

    if model_config_file == "" or model_config_file=='None':
        model_config_file = None

    # Determining the pre-trained weights.
    if weights_path == 'default':
        git_path = Path(__file__).parent  # Should be the DNACipher path

        model_path = f'{git_path}/weights/'
        weights_path = f'{model_path}TRAINING_DNACV5_MID-AVG-GENOME_ORIG-ALLOC_ENFORMER0_FINETUNE_STRATMSE_model_weights.pth'

    te.eval(run_name, genome_file, sample_file, signal_file, embed_file,
            weights_path, out_dir, model_config_file, ct_knn_file,
            # train parameters, need to change depending on resources.
            batch_size, cpu, device,eval_batches,
           )

@app.command()
def celltypes():
    """Prints the available celltypes to infer effects for."""

    celltype_assays, celltype_assay_labels, celltypes, assays = clh.load_context_data()

    for celltype in celltypes:
        print(celltype, file=sys.stdout, flush=True)

@app.command()
def assays():
    """Prints the available assays to infer effects for."""

    celltype_assays, celltype_assay_labels, celltypes, assays = clh.load_context_data()

    for assay in assays:
        print(assay, file=sys.stdout, flush=True)

@app.command()
def infer_signals(
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all output files.")],
    bed_path: Annotated[str, typer.Argument(help="BED file with regions to infer signals for.")],
    celltypes: Annotated[str, typer.Argument(help="Celltypes to infer signals for. File in format: ct1,,ct2,...")],
    assays: Annotated[str, typer.Argument(help="Assays to infer signals for. File in format: assay1,,assay2,...")],
    fasta_file_path: Annotated[str, typer.Argument(help="FASTA file path for the reference genome. Must have .fai index.")],

    sample_file: Annotated[Optional[str], typer.Option("-sf", "-sample_file", help="Sample metadata used to train the model.")] = None,
    model_config_file: Annotated[Optional[str], typer.Option("-m", "-model_config", help="YAML file specifying model parameters.")] = None,
    weights_path: Annotated[Optional[str], typer.Option("-w", "-weights", help="Path to pretrained model weights (.pth).")] = None,
    model_name: Annotated[Optional[str], typer.Option("-mn", "-model_name", help="Model name to use (e.g. borzoi).")] = None,
    device: Annotated[Optional[str], typer.Option("-d", "-device", help="Device to run model on.")] = None,

    batch_size: Annotated[Optional[int], typer.Option("-b", "-batch_size", help="Batch size for inference.")] = None,

    all_combinations: Annotated[bool, typer.Option("--all_combinations/--no-all_combinations", help="All combinations of inputted celltype/assays, or specific experiments are already specified by the celltype/assay inputs.")] = True,
    verbose: Annotated[bool, typer.Option("--verbose/--silent", help="Verbose output.")] = True,
):
    """
    Infer signals across genomic regions for given celltype/assays.
    """

    si.infer_signals(
        out_prefix, bed_path, celltypes, assays, fasta_file_path,
        sample_file, model_config_file, weights_path, model_name,
        device, batch_size,
        all_combinations, verbose,
    )

@app.command()
def infer_multivariant_effects(
        out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
        vcf_path: Annotated[str, typer.Argument(help="Rows represent particular genetic variants, columns are CHR, POS, REF, ALT")],
        celltypes: Annotated[str, typer.Argument(help="Celltypes to infer effects for. File in format: ct1,,ct2,,ct3")],
        assays: Annotated[str, typer.Argument(help="Assays to infer effects for. File in format: assay1,,assay2,,assay3")],
        fasta_file_path: Annotated[str, typer.Argument(help="FASTA file path for the reference genome. Must have .fai index.")],
        sample_file: Annotated[Optional[str], typer.Option("-sf", "-sample_file", help="Sample metadata used to train the DNACipher model to be used for inference. If not provided uses default DNACipher.")] = None,
        model_config_file: Annotated[Optional[str], typer.Option("-m", "-model_config", help=(".yaml file specifying the model parameters, such as layer sizes, number of layers, etc. If not provided, defaults to an internal config specifying the original model."))] = None,
        weights_path: Annotated[Optional[str], typer.Option("-w", "-weights", help=(".pth specifying the pretrained DNACipher model weights. if None will use default pretrained DNACipher model."))] = None,
        model_name: Annotated[Optional[str], typer.Option("-mn", "-model_name", help='Name of the model to use for precomputing the embeddings for training. If None defaults to pretrained DNACipher model.')] = None,
        device: Annotated[Optional[str], typer.Option("-d", "-device", help="Device to run model on.")] = None,
        index_base: Annotated[Optional[int], typer.Option("-i", "-index_base", help=("Whether the variant position is 0-based or 1-based indexing."))] = 0,
        correct_ref: Annotated[bool, typer.Option("-correct_ref/-no-correct_ref", help="Correct the reference genome sequence if disagrees with the inputted ref allele.")] = False,
        seq_pos_col: Annotated[Optional[str], typer.Option("-sc", "-seq_pos_col", help=("Column in vcf that specifies the position to centre the query sequence on, must be within dnacipher.seqlen_max in order to predict effect of the genetic variant. If None then will centre the query sequence on the inputted variant."))] = None,
        effect_region_start_col: Annotated[Optional[str], typer.Option("-ersc", "-effect_region_start_col", help=("Specifies column in the inputted data frame specifying the start position (in genome coords) to measure the effect"))] = None,
        effect_region_end_col: Annotated[Optional[str], typer.Option("-erec", "-effect_region_end_col", help=("Specifies column in the inputted data frame specifying the end position (in genome coords) to measure the effect"))] = None,
        batch_size: Annotated[Optional[int], typer.Option("-b", "-batch_size", help=("How many effects to infer at a time."))] = None,
        batch_by: Annotated[Optional[str], typer.Option("-by", "-batch_by", help=("Indicates how to batch the data when fed into the model, either by 'experiment', 'sequence', or None. If None, will automatically choose whichever is the larger axis."))] = None,
        all_combinations: Annotated[bool, typer.Option("-all_combinations/-no-all_combinations",help="Generate predictions for all combinations of inputted cell types and assays.")] = True,
        scoring_method: Annotated[str, typer.Option("-sm", "-scoring_method",help="Refers to how to compare the ref and alt prediction to score variant effects. Supported methods are 'signed_sum_abs' which will take the SUM(|ALT-REF|)*{-1 if SUM(ALT-REF) < 0}. 'sum_logfc' which will score as log2( (SUM(ALT) / SUM(REF)) + 1).")] = "signed_sum_abs",
        res_reduce_factor: Annotated[Optional[int], typer.Option("-rrf", "-res_reduce_factor", help=("Decrease the resolution of the imputed signals by this factor. E.g. rrf=4, 32bp/bin goes to 128bp/bin. If None maintains same resolution as embedding model."))] = None,
        verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output")] = True
):
    """ Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per
            variant, and predicted effect sizes across the columns for celltype/assay combinations.
    """
    # TODO:
    #     * Should implement outputting the positional information as well, so could parameterize moleculear effects as (pos, celltype, assay)

    infer.infer_multivariant_effects(out_prefix, vcf_path, celltypes, assays, fasta_file_path,
                                     sample_file, model_config_file, weights_path, model_name,
                                     device, index_base,
                                     correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col,
                                     batch_size,
                                     batch_by, all_combinations, scoring_method, verbose, res_reduce_factor=res_reduce_factor,
                               )

@app.command()
def stratify_variants(
        out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
        signal_gwas_stats_path: Annotated[str, typer.Argument(help="Path to GWAS summary statistics for each of the variants at a GWAS locus, within range model predictions.")],
        var_ref_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant reference sequence as a string.")],
        var_alt_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant alternate sequence as a string.")],
        var_loc_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant position as an integer.")],
        p_col: Annotated[str, typer.Argument(help="Name of column in the input dataframe that contains the p-values of the variant-trait associations.")],
        allele_freq_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant allele frequency as a float.")],
        p_cut: Annotated[Optional[float], typer.Option("-pc", "-p_cut", help="P-value cutoff to consider a variant significantly associated with the trait.")] = 5e-07,
        lowsig_cut: Annotated[Optional[float], typer.Option("-lc", "-lowsig_cut", help=("Cutoff to consider variants confidently not-significant."))] = 0.001,
        n_top: Annotated[Optional[int], typer.Option("-nt", "-n_top", help=("If no significant variants, will take this many as the top candidates."))] = 10,
        allele_freq_cut: Annotated[Optional[float], typer.Option("-afc", "-allele_freq_cut", help=("If a variant is above this minor allele frequency AND is considered confidently non-significant, then is considered a 'background' variant. If is below this allele frequency, then considered a rare variant."))] = 0.05,
        min_bg_variants: Annotated[Optional[int], typer.Option("-mbv", "-min_bg_variants", help=("If have less than this number of background variants, will rank-order potential background variants by scoring allele frequency and significance, and take this many variants as significant."))] = 100,
        verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output")] = True
):
    """ Performs stratification of variants at GWAS loci to categories:
        * 'candidate' variants (common significant variants),
        * 'rare' variants (non-significant rare variants in the same region as the candidate variants),
        * 'background' variants (common non-significant variants), and 'other' variants (rare variants outside of the hit locus).
    """

    # TODO NEED TO IMPLEMENT THIS SO THAT IT CAN ACCEPT VARIANTS FROM MULTIPLE LOCI, AND STRATIFY THEM INDEPENDENTLY
    #  BY LOCI.

    signal_gwas_stats = pd.read_csv(signal_gwas_stats_path, sep='\t')

    stratified_gwas_stats = dvim.stratify_variants(signal_gwas_stats, var_ref_col=var_ref_col, var_alt_col=var_alt_col,
                                                   var_loc_col=var_loc_col, p_col=p_col,
                                                   allele_freq_col=allele_freq_col,
                                                   p_cut=p_cut, lowsig_cut=lowsig_cut, n_top=n_top,
                                                   allele_freq_cut=allele_freq_cut,
                                                   min_bg_variants=min_bg_variants, verbose=verbose)

    out_path = f"{out_prefix}stratified_gwas_stats.txt"
    stratified_gwas_stats.to_csv(out_path, index=False, sep='\t')

    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)


@app.command()
def impact_map(
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
    selected_gwas_stats_path: Annotated[str, typer.Argument(help="Path to GWAS summary statistics for each of the variants at a GWAS locus, with a column 'var_labels' indicating candidate, rare, and background variants. Each row is a variant.")],
    pred_effects_path: Annotated[str, typer.Argument(help="Path to predicted effects for each variant. Each row is a variant, and each column is a predicted molecular effect for that variant.")],
    cpu: Annotated[Optional[int], typer.Option("-c", "-cpu", help=("No. of CPUs to use. Parallelizes across the number of loci, so no effect if only one locus."))] = 3,
    n_boots: Annotated[Optional[int], typer.Option("-nb", "-n_boots", help=("No. of boot-straps of re-selecting the background variants."))] = 10_000,
    p_cutoff: Annotated[Optional[float], typer.Option("-pc", "-p_cutoff", help="P-value below which a non-background variant predicted molecular effect is considered significantly different to the background vars.")] = 0.05,
    fc_cutoff: Annotated[Optional[float], typer.Option("-fc", "-fc_cutoff", help=("Fold-change cutoff to be considered significant."))] = 0,
    min_std: Annotated[Optional[float], typer.Option("-std", "-min_std", help=("Minimum standard deviation for the background variant effects. Set to avoid 0 std for 0 effects of background variants causing infinite z-scores."))] = 0.01,
    pseudocount: Annotated[Optional[int], typer.Option("-p", "-pseudocount", help=("Value added to boot-strap counts to prevent 0 p-values, defines lower-bound for minimum p-values, should be set to 1."))] = 1,
    verbosity: Annotated[Optional[int], typer.Option("-v", "-verbosity", help=("Verbosity levels. 0 errors only, 1 prints processing progress, 2 prints debugging information."))] = 1,
):
    """ Calls 'impact' variants - variants with significant predicted effects in particular cell types / assays compared with background variants.
    """

    verbose = False
    if verbosity > 0:
        verbose = True

    selected_gwas_stats = pd.read_csv(selected_gwas_stats_path, sep='\t')
    selected_pred_effects = pd.read_csv(pred_effects_path, sep='\t')

    selected_gwas_stats, sig_effects, foldchange_effects, boot_pvals_df, boot_counts_df = \
                                                        dvim.impact_map(selected_gwas_stats, selected_pred_effects,
                                                                        cpu=cpu, p_cutoff=p_cutoff, min_std=min_std,
                                                                        n_boots=n_boots, verbose=verbose,
                                                                        pseudocount=pseudocount, fc_cutoff=fc_cutoff
                                                                        )

    out_paths = [selected_gwas_stats_path.replace('.txt', '.impact_calls.txt'), f"{out_prefix}sig_effects.txt",
                 f"{out_prefix}fold_changes.txt"]+[f"{out_prefix}boot_pvals.txt", f"{out_prefix}boot_counts.txt"]
    out_dfs = [selected_gwas_stats, sig_effects, foldchange_effects, boot_pvals_df, boot_counts_df]
    for out_path, dataframe in zip(out_paths, out_dfs):

        dataframe.to_csv(out_path, index=False, sep='\t')

        if verbose:
            print(f"Wrote {out_path}", file=sys.stdout, flush=True)

    if verbose:
        loci_common_sig = len(np.where(selected_gwas_stats['locus_common_padj'].values < p_cutoff)[0])
        loci_rare_sig = len(np.where(selected_gwas_stats['locus_rare_padj'].values < p_cutoff)[0])

        print(f"\nFound {loci_common_sig} loci with significant effects for common variants.", file=sys.stdout, flush=True)
        print(f"\nFound {loci_rare_sig} loci with significant effects for rare variants.", file=sys.stdout,flush=True)

@app.command()
def plot_variant_stats(
        out_prefix: Annotated[str, typer.Argument(help="Prefix for output files.")],
        stratified_gwas_stats_path: Annotated[str, typer.Argument(help="Path to stratified GWAS statistics file.")],
        y_axis_col: Annotated[str, typer.Argument(help="Column name to use for y-axis values (e.g., -log10_pval).")],
        color_by: Annotated[str, typer.Argument(help="Column name to color points by (e.g., var_label).")],
        var_loc_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant position as an integer.")],
        locus_index: Annotated[Optional[int], typer.Option("-l", "-locus", help="Which locus to plot, as an integer specifying the locus ordering, if multiple loci represented by a seq_pos column. If None will plot first locus are all variants, for latter case assuming file represents single locus.")] = None,
        chrom_col: Annotated[Optional[str], typer.Option("-chr", "-chrom_col", help="Name of column that contains the chromosome.")] = None,
        gtf_file_path: Annotated[Optional[str], typer.Option("-gtf", "-gtf_file_path", help="Optional GTF file path for plotting gene annotations.")] = None,
        color_map: Annotated[Optional[str], typer.Option("-cmap", "-color_map", help="Colormap name for continuous coloring.")] = "magma",
        alpha: Annotated[Optional[float], typer.Option("-alpha", help="Opacity of the scatter plot points.")] = 0.5,
        order_points: Annotated[bool, typer.Option("-order_points/-no-order_points", help="Whether to plot points ordered by statistic.")] = True,
        reverse_order: Annotated[bool, typer.Option("-reverse_order/-no-reverse_order", help="Whether to reverse point order.")] = False,
        show_legend: Annotated[bool, typer.Option("-show_legend/-no-show_legend", help="Whether to display the plot legend.")] = True,
        verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output.")] = True
):
    """Manhattan-like plot for variant statistics."""

    # Load data
    locus_gwas_stats = pd.read_csv(stratified_gwas_stats_path, sep='\t')

    # TODO NEED TO TEST THIS FUNCTIONALIY OF SELECTING A LOCUS.
    locus = ''
    if 'seq_pos' in locus_gwas_stats.columns: # Subsetting to just the first locus in case has multiple loci.

        if type(locus_index)==type(None):
            locus = locus_gwas_stats['seq_pos'].values[0]

        else:
            loci = list({locus: index for locus, index in enumerate(locus_gwas_stats['seq_pos'].values)}.keys())
            locus = loci[locus_index]

        locus_gwas_stats = locus_gwas_stats.loc[locus_gwas_stats['seq_pos'].values == locus, :]

    gtf_df = None
    if gtf_file_path:
        #### Loading the gtf
        gtf_cols = [
            "chrom", "source", "feature", "start", "end",
            "score", "strand", "frame", "attribute"
        ]
        gtf_df = pd.read_csv(gtf_file_path, sep="\t", header=None, comment='#', names=gtf_cols)
        gtf_df = gtf_df.loc[gtf_df["feature"].values == 'gene', :]
        gtf_df['gene_names'] = gtf_df.apply(lambda x: x.iloc[8].split('gene_name')[1].split('; ')[0].strip(' "'), 1)
        gtf_df['gene_ids'] = gtf_df.apply(lambda x: x.iloc[8].split('gene_id')[1].split('; ')[0].strip(' "'), 1)

        if verbose:
            print(f"Loaded {len(gtf_df)} genes from {gtf_file_path}", file=sys.stdout, flush=True)

    # Predefined colors for variant labels
    variant_colors = {'other': 'grey', 'candidate': 'orange', 'background': 'black', 'rare': 'magenta'}

    dnapl.plot_variant_stats(
        locus_gwas_stats,
        y_axis_col=y_axis_col,
        color_by=color_by,
        color_dict=variant_colors,
        color_map=color_map,
        alpha=alpha,
        order_points=order_points,
        reverse_order=reverse_order,
        gtf_df=gtf_df,
        show_legend=show_legend,
        var_chrom_col=chrom_col, var_loc_col=var_loc_col,
    )

    out_path = f"{out_prefix}{locus}{y_axis_col}_{color_by}_variant_stats.png"
    clh.dealWithPlot(True, False, True, '', out_path, 300)

    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)


@app.command()
def plot_volcano(
        out_prefix: Annotated[str, typer.Argument(help="Prefix for output files.")],
        variant_type: Annotated[str, typer.Argument(help="Variant label category to highlight in the volcano plot (e.g., candidate).")],
        selected_gwas_stats_path: Annotated[str, typer.Argument(help="Path to selected GWAS statistics.")],
        sig_effects_path: Annotated[str, typer.Argument(help="Path to significant effects matrix.")],
        foldchange_effects_path: Annotated[str, typer.Argument(help="Path to fold-change effects matrix.")],
        boot_pvals_path: Annotated[str, typer.Argument(help="Path to bootstrapped p-values matrix.")],
        alpha: Annotated[Optional[float], typer.Option("-alpha", help="Opacity of the scatter plot points.")] = 0.4,
        verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output.")] = True
):
    """Volcano plot for Deep Variant Impact Mapping predicted molecular effects."""

    # TODO NEED TO ADD FUNCTIONALIY OF SELECTING A LOCUS, like in the previous function.

    selected_gwas_stats = pd.read_csv(selected_gwas_stats_path, sep='\t')
    sig_effects = pd.read_csv(sig_effects_path, sep='\t')
    foldchange_effects = pd.read_csv(foldchange_effects_path, sep='\t')
    boot_pvals_df = pd.read_csv(boot_pvals_path, sep='\t')

    dnapl.plot_volcano(
        variant_type,
        selected_gwas_stats,
        sig_effects,
        foldchange_effects,
        boot_pvals_df,
        down_color='dodgerblue',
        up_color='tomato',
        nonsig_color='grey',
        alpha=alpha,
        show=False
    )

    out_path = f"{out_prefix}{variant_type}_volcano.png"
    clh.dealWithPlot(True, False, True, '', out_path, 300)

    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)

def main():
    root_dir = Path(__file__).parent
    sys.path.append(str(root_dir))
    app()

if __name__ == "__main__":
    main()



