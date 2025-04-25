""" CLI for performing DNACipher analysis of genetic variants.
"""

import sys

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated

import typer
import sys

from . import command_line_helpers as clh
from . import dna_cipher_infer as dnaci
from . import deep_variant_impact_mapping as dvim
from . import dna_cipher_plotting as dnapl

app = typer.Typer(pretty_exceptions_short=False)

@app.command()
def infer_effects(
    chr_: Annotated[str, typer.Argument(help="Chromosome of variant.")],
    pos: Annotated[int, typer.Argument(help="Position of variant on chromosome.")],
    ref: Annotated[str, typer.Argument(help="Reference allele of variant.")],
    alt: Annotated[str, typer.Argument(help="Alternate allele of variant.")],
    celltypes: Annotated[str, typer.Argument(help="Celltypes to infer effects for. File in format: ct1,,ct2,,ct3")],
    assays: Annotated[str, typer.Argument(help="Assays to infer effects for. File in format: assay1,,assay2,,assay3")],
    fasta_file_path: Annotated[str, typer.Argument(help="FASTA file path for the reference genome. Must have .fai index.")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
    device: Annotated[Optional[str], typer.Option( "-d", "-device", help="Device to run model on.")] = None,
    index_base: Annotated[Optional[int], typer.Option( "-i", "-index_base", help=( "Whether the variant position is 0-based or 1-based indexing.") )] = 0,
    correct_ref: Annotated[bool, typer.Option("-correct_ref/-no-correct_ref", help="Correct the reference genome sequence if disagrees with the inputted ref allele.")] = False,
    seq_pos: Annotated[Optional[int], typer.Option("-s", "-seq_pos", help=( "Where to centre the query sequence.") )] = None,
    effect_region_start: Annotated[Optional[int], typer.Option( "-ers", "-effect_region_start", help=( "Where to start measuring the effect in the genome.") )] = None,
    effect_region_end: Annotated[Optional[int], typer.Option( "-ere", "-effect_region_end", help=( "Where to end measuring the effect in the genome.") )] = None,
    
    batch_size: Annotated[Optional[int], typer.Option( "-b", "-batch_size", help=( "How many effects to infer at a time.") )] = None,
    batch_by: Annotated[Optional[str], typer.Option( "-by", "-batch_by", help=( "Indicates how to batch the data when fed into the model, either by 'experiment', 'sequence', or None. If None, will automatically choose whichever is the larger axis.") )] = None,
    all_combinations: Annotated[bool, typer.Option("-all_combinations/-no-all_combinations", help="Generate predicetions for all combinations of inputted cell types and assays.")] = True,
    return_all: Annotated[bool, typer.Option("-return_all/-no-return_all", help="Return the signals across the ref and alt sequences")] = False,
    verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output")] = True
                 ):
    """ Infers a single varaints effects across celltypes and assays, and optionally across the sequence.
    """

    dnacipher, celltypes, assays = clh.parse_general_input(celltypes, assays, device, fasta_file_path, verbose)
    
    if type(effect_region_start)!=type(None) and type(effect_region_end)!=type(None):
        effect_region = (effect_region_start, effect_region_end)
    else:
        effect_region = None
        
    if return_all: # Detailed predictions along the strand !
        context_effects, ref_signals, alt_signals, ref_seq, alt_seq, ref_features, alt_features = \
                                    dnacipher.infer_effects(chr_, pos, ref, alt, celltypes, assays,
                                              index_base=index_base, correct_ref=correct_ref, seq_pos=seq_pos, 
                                              effect_region=effect_region, batch_size=batch_size, batch_by=batch_by,
                                              all_combinations=all_combinations, verbose=verbose,
                                              return_all=True, 
                                             )
        
        ref_signals_normed_and_ordered, alt_signals_normed_and_ordered = dnacipher.normalise_and_ordered_signals(
                                                                                            ref_signals, alt_signals
                                                                                                          )
        diff_signals = dnacipher.get_diff_signals(ref_signals_normed_and_ordered, alt_signals_normed_and_ordered)
        
        frames = [context_effects, ref_signals, alt_signals, ref_signals_normed_and_ordered, alt_signals_normed_and_ordered,
                 diff_signals]
        
        for i, (dataframe, name_) in enumerate(zip(frames, ['context_effects', 'ref_signals', 'alt_signals',
                                             'ref_signals_normed_and_ordered', 'alt_signals_normed_and_ordered', 
                                             'diff_signals'])):

            out_path = f"{out_prefix}{name_}.txt"
            dataframe.to_csv(out_path, index=i!=0, sep='\t') #all of the signal frames need the index
            
            if verbose:
                print(f"Wrote {out_path}", file=sys.stdout, flush=True)
                
    else: # Just the overall locus effects.
        context_effects = dnacipher.infer_effects(chr_, pos, ref, alt, celltypes, assays,
                                              index_base=index_base, correct_ref=correct_ref, seq_pos=seq_pos, 
                                              effect_region=effect_region, batch_size=batch_size, batch_by=batch_by,
                                              all_combinations=all_combinations, verbose=verbose,
                                              return_all=False, 
                                             )
        
        out_path = f"{out_prefix}context_effects.txt"
        context_effects.to_csv(out_path, index=False, sep='\t')
        
        if verbose:
                print(f"Wrote {out_path}", file=sys.stdout, flush=True)
    

@app.command()
def infer_multivariant_effects(
    vcf_path: Annotated[str, typer.Argument(help="Rows represent particular genetic variants, columns are CHR, POS, REF, ALT")],
    celltypes: Annotated[str, typer.Argument(help="Celltypes to infer effects for. File in format: ct1,ct2,ct3")],
    assays: Annotated[str, typer.Argument(help="Assays to infer effects for. File in format: assay1,assay2,assay3")],
    fasta_file_path: Annotated[str, typer.Argument(help="FASTA file path for the reference genome. Must have .fai index.")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
    device: Annotated[Optional[str], typer.Option( "-d", "-device", help="Device to run model on.")] = None,
    index_base: Annotated[Optional[int], typer.Option( "-i", "-index_base", help=( "Whether the variant position is 0-based or 1-based indexing.") )] = 0,
    correct_ref: Annotated[bool, typer.Option("-correct_ref/-no-correct_ref", help="Correct the reference genome sequence if disagrees with the inputted ref allele.")] = False,
    seq_pos_col: Annotated[Optional[str], typer.Option("-sc", "-seq_pos_col", help=( "Column in vcf that specifies the position to centre the query sequence on, must be within dnacipher.seqlen_max in order to predict effect of the genetic variant. If None then will centre the query sequence on the inputted variant.") )] = None,
    effect_region_start_col: Annotated[Optional[str], typer.Option( "-ersc", "-effect_region_start_col", help=( "Specifies column in the inputted data frame specifying the start position (in genome coords) to measure the effect") )] = None,
    effect_region_end_col: Annotated[Optional[str], typer.Option( "-erec", "-effect_region_end_col", help=( "Specifies column in the inputted data frame specifying the end position (in genome coords) to measure the effect") )] = None,
    batch_size: Annotated[Optional[int], typer.Option( "-b", "-batch_size", help=( "How many effects to infer at a time.") )] = None,
    batch_by: Annotated[Optional[str], typer.Option( "-by", "-batch_by", help=( "Indicates how to batch the data when fed into the model, either by 'experiment', 'sequence', or None. If None, will automatically choose whichever is the larger axis.") )] = None,
    all_combinations: Annotated[bool, typer.Option("-all_combinations/-no-all_combinations", help="Generate predicetions for all combinations of inputted cell types and assays.")] = True,
    verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output")] = True
     ):
    """ Takes as input a vcf file, in format, CHR, POS, REF, ALT as columns. Outputs a dataframe with rows per
            variant, and predicted effect sizes across the columns for all celltype/assay combinations.
    """
    # TODO: 
    #     * Should implement outputting the positional information as well, so could parameterize moleculear effects as (pos, celltype, assay)

    dnacipher, celltypes, assays = clh.parse_general_input(celltypes, assays, device, fasta_file_path, verbose)

    if type(effect_region_start_col)!=type(None) and type(effect_region_end_col)!=type(None):
        effect_region_cols = (effect_region_start_col, effect_region_end_col)
    else:
        effect_region_cols = None

    var_df = pd.read_csv(vcf_path, sep='\t', header=0)

    var_pred_effects = dnacipher.infer_multivariant_effects(var_df, celltypes, assays,
                                              index_base=index_base, correct_ref=correct_ref, seq_pos_col=seq_pos_col, 
                                              effect_region_cols=effect_region_cols, batch_size=batch_size, batch_by=batch_by,
                                              all_combinations=all_combinations, verbose=verbose,
                                             )

    out_path = f"{out_prefix}var_context_effects.txt"
    var_pred_effects.to_csv(out_path, index=False, sep='\t')
    
    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)

@app.command()
def stratify_variants(
    signal_gwas_stats_path: Annotated[str, typer.Argument(help="Path to GWAS summary statistics for each of the variants at a GWAS locus, within range model predictions.")],
    var_ref_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant reference sequence as a string.")],
    var_alt_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant alternate sequence as a string.")],
    var_loc_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant position as an integer.")],
    p_col: Annotated[str, typer.Argument(help="Name of column in the input dataframe that contains the p-values of the variant-trait associations.")],
    allele_freq_col: Annotated[str, typer.Argument(help="Column in the input dataframe that specifies the variant allele frequency as a float.")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
    p_cut: Annotated[Optional[float], typer.Option( "-pc", "-p_cut", help="P-value cutoff to consider a variant significantly associated with the trait.")] = 5e-07,
    lowsig_cut: Annotated[Optional[float], typer.Option( "-lc", "-lowsig_cut", help=( "Cutoff to consider variants confidently not-significant.") )] = 0.001,
    n_top: Annotated[Optional[int], typer.Option( "-nt", "-n_top", help=( "If no significant variants, will take this many as the top candidates.") )] = 10,
    allele_freq_cut: Annotated[Optional[float], typer.Option( "-afc", "-allele_freq_cut", help=( "If a variant is above this minor allele frequency AND is considered confidently non-significant, then is considered a 'background' variant. If is below this allele frequency, then considered a rare variant.") )] = 0.05,
    min_bg_variants: Annotated[Optional[int], typer.Option( "-mbv", "-min_bg_variants", help=( "If have less than this number of background variants, will rank-order potential background variants by scoring allele frequency and significance, and take this many variants as significant.") )] = 100,
    verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output")] = True
):
    """ Performs stratification of variants at GWAS loci to categories: 
        * 'candidate' variants (common significant variants), 
        * 'rare' variants (non-significant rare variants in the same region as the candidate variants), 
        * 'background' variants (common non-significant variants), and 'other' variants (rare variants outside of the hit locus).
    """
    
    signal_gwas_stats = pd.read_csv(signal_gwas_stats_path, sep='\t')

    stratified_gwas_stats = dvim.stratify_variants(signal_gwas_stats, var_ref_col=var_ref_col, var_alt_col=var_alt_col,
                                                    var_loc_col=var_loc_col, p_col=p_col, allele_freq_col=allele_freq_col,
                                                    p_cut=p_cut, lowsig_cut=lowsig_cut, n_top=n_top, allele_freq_cut=allele_freq_cut,
                                                    min_bg_variants=min_bg_variants, verbose=verbose)

    out_path = f"{out_prefix}stratified_gwas_stats.txt"
    stratified_gwas_stats.to_csv(out_path, index=False, sep='\t')

    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)
    
@app.command()
def effect_pvals(
selected_gwas_stats_path: Annotated[str, typer.Argument(help="Path to GWAS summary statistics for each of the variants at a GWAS locus, with a column 'var_labels' indicating candidate, rare, and background variants. Each row is a variant.")],
pred_effects_path: Annotated[str, typer.Argument(help="Path to predicted effects for each variant. Each row is a variant, and each column is a predicted molecular effect for that variant.")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
    n_boots: Annotated[Optional[int], typer.Option( "-nb", "-n_boots", help=( "No. of boot-straps of re-selecting the background variants.") )] = 10_000,
    p_cutoff: Annotated[Optional[float], typer.Option( "-pc", "-p_cutoff", help="P-value below which a non-background variant predicted molecular effect is considered significantly different to the background vars.")] = 0.05,
    pseudocount: Annotated[Optional[int], typer.Option( "-p", "-pseudocount", help=( "Value added to boot-strap counts to prevent 0 p-values, defines lower-bound for minimum p-values, should be set to 1.") )] = 1,
    min_std: Annotated[Optional[float], typer.Option( "-std", "-min_std", help=( " Minimum standard deviation for the background variant effects. Set to avoid 0 std for 0 effects of background variants causing infinite z-scores.") )] = 0.01,
    verbosity: Annotated[Optional[int], typer.Option( "-v", "-verbosity", help=( "Verbosity levels. 0 errors only, 1 prints processing progress, 2 prints debugging information.") )] = 1,
):
    """ Calculates variant effect p-values for non-background variants against background variants.
    """

    selected_gwas_stats = pd.read_csv(selected_gwas_stats_path, sep='\t')
    selected_pred_effects = pd.read_csv(pred_effects_path, sep='\t')

    # print(selected_gwas_stats.head(2))
    # print(selected_pred_effects.head(2))

    boot_pvals_df, boot_counts_df = dvim.calc_variant_effect_pvals(selected_gwas_stats, selected_pred_effects, 
                                                                   p_cutoff=p_cutoff, min_std=min_std,
                                                                   n_boots=n_boots, verbosity=verbosity,
                                                                  pseudocount=pseudocount)

    out_paths = [f"{out_prefix}boot_pvals.txt", f"{out_prefix}boot_counts.txt"]
    for out_path, dataframe in zip(out_paths, [boot_pvals_df, boot_counts_df]):
        
        dataframe.to_csv(out_path, index=False, sep='\t')
    
        if verbosity >= 1:
            print(f"Wrote {out_path}", file=sys.stdout, flush=True)

@app.command()
def impact_map(
selected_gwas_stats_path: Annotated[str, typer.Argument(help="Path to GWAS summary statistics for each of the variants at a GWAS locus, with a column 'var_labels' indicating candidate, rare, and background variants. Each row is a variant.")],
pred_effects_path: Annotated[str, typer.Argument(help="Path to predicted effects for each variant. Each row is a variant, and each column is a predicted molecular effect for that variant.")],
boot_pvals_path: Annotated[str, typer.Argument(help="Path to predicted effects for each variant. Each row is a variant, and each column is a predicted molecular effect for that variant.")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
    p_cutoff: Annotated[Optional[float], typer.Option( "-pc", "-p_cutoff", help="P-value below which a non-background variant predicted molecular effect is considered significantly different to the background vars.")] = 0.05,
    fc_cutoff: Annotated[Optional[float], typer.Option( "-fc", "-fc_cutoff", help=( "Fold-change cutoff to be considered significant.") )] = 0,
    verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output")] = True
):
    """ Calls 'impact' variants - variants with significant predicted effects in particular cell types / assays compared with background variants.
    """

    selected_gwas_stats = pd.read_csv(selected_gwas_stats_path, sep='\t')
    selected_pred_effects = pd.read_csv(pred_effects_path, sep='\t')
    boot_pvals_df = pd.read_csv(boot_pvals_path, sep='\t')

    sig_effects, foldchange_effects = dvim.call_sig_effects(selected_gwas_stats, selected_pred_effects, boot_pvals_df, 
                                                            p_cutoff=p_cutoff, fc_cutoff=fc_cutoff)

    ### Calling the impact variants
    n_sig_effects = (sig_effects).sum(axis=1)
    selected_gwas_stats['n_sig_effects'] = n_sig_effects
    selected_gwas_stats['impact_variant'] = n_sig_effects > 0

    out_paths = [selected_gwas_stats_path.replace('.txt', '.impact_calls.txt'), f"{out_prefix}sig_effects.txt", f"{out_prefix}fold_changes.txt"]
    for out_path, dataframe in zip(out_paths, [selected_gwas_stats, sig_effects, foldchange_effects]):
        
        dataframe.to_csv(out_path, index=False, sep='\t')
    
        if verbose:
            print(f"Wrote {out_path}", file=sys.stdout, flush=True)

@app.command()
def plot_signals(
    signals_path: Annotated[str, typer.Argument(help="Path to the signal predictions across contexts for a given variant.")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for all outputs files.")],
    gtf_file_path: Annotated[Optional[str], typer.Option("-gtf", "-gtf_file_path", help="Optional: GTF file path for gene annotation overlap.")] = None,
    encode_cres_path: Annotated[Optional[str], typer.Option("-cres", "-encode_cres_path", help="Optional: BED file path for cCREs annotation overlap.")] = None,
    variant_chr: Annotated[Optional[str], typer.Option("-chr", "-variant_chr", help="Chromosome of the variant (for plotting vertical line).")] = None,
    variant_pos: Annotated[Optional[int], typer.Option("-pos", "-variant_pos", help="Position of the variant (for plotting vertical line).")] = None,
    variant_ref: Annotated[Optional[str], typer.Option("-ref", "-variant_ref", help="Reference allele of the variant.")] = None,
    variant_alt: Annotated[Optional[str], typer.Option("-alt", "-variant_alt", help="Alternate allele of the variant.")] = None,
    plot_delta: Annotated[bool, typer.Option("-plot_delta/-no-plot_delta", help="Whether to plot the difference between ref and alt signals.")] = False,
    xtick_freq: Annotated[Optional[int], typer.Option("-xtick_freq", help="Spacing between x-ticks in base pairs.")] = 200,
    title: Annotated[Optional[str], typer.Option("-title", help="Plot title.")] = "DNACipher track predictions",
    verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output.")] = True
):
    """Plots DNACipher signal tracks and optional gene/cCRE annotations."""

    # Load predicted effects
    signals_normed = pd.read_csv(signals_path, index_col=0, sep='\t')

    # print(signals_normed.head(3), file=sys.stdout, flush=True)
    # return

    # Optional annotations
    genes = None
    if gtf_file_path:
        genes = dnaci.DNACipher.load_intersecting_annots(signals_normed, gtf_file_path)
        if verbose:
            print(f"Loaded {len(genes)} genes from {gtf_file_path}", file=sys.stdout, flush=True)

    cres = None
    if encode_cres_path:
        cres = dnaci.DNACipher.load_intersecting_annots(signals_normed, encode_cres_path, gtf=False)
        if verbose:
            print(f"Loaded {len(cres)} cCREs from {encode_cres_path}", file=sys.stdout, flush=True)

    # Variant information
    variant_loc = None
    if variant_chr and variant_pos and variant_ref and variant_alt:
        variant_loc = (variant_chr, variant_pos, variant_ref, variant_alt)

    # Call plotting function
    dnapl.plot_signals(
        signals_normed,
        variant_loc=variant_loc,
        plot_delta=plot_delta,
        title=title,
        genes=genes,
        regions=cres,
        xlabel="Genomic position",
        xtick_freq=xtick_freq,
        show=False
    )

    out_path = f"{out_prefix}signals_plot.png"
    clh.dealWithPlot(True, False, True, '', out_path, 300)

    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)

# @app.command()
# def plot_effects_matrix(

# ):
#     """TODO: Plots DNACipher variant effect predictions across cell type / assay combinations as a heatmap."""
#     print("NOT YET IMPLEMENTED", file=sys.stdout, flush=True)

@app.command()
def plot_variant_stats(
    stratified_gwas_stats_path: Annotated[str, typer.Argument(help="Path to stratified GWAS statistics file.")],
    y_axis_col: Annotated[str, typer.Argument(help="Column name to use for y-axis values (e.g., -log10_pval).")],
    color_by: Annotated[str, typer.Argument(help="Column name to color points by (e.g., var_label).")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for output files.")],
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

    gtf_df = None
    if gtf_file_path:
        #### Loading the gtf
        gtf_cols = [
                "chrom", "source", "feature", "start", "end",
                "score", "strand", "frame", "attribute"
            ]
        gtf_df = pd.read_csv(gtf_file_path, sep="\t", header=None, comment='#', names=gtf_cols)
        gtf_df = gtf_df.loc[gtf_df["feature"].values=='gene', :]
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
        show_legend=show_legend
    )

    out_path = f"{out_prefix}{y_axis_col}_{color_by}_variant_stats.png"
    clh.dealWithPlot(True, False, True, '', out_path, 300)

    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)

@app.command()
def plot_volcano(
    variant_type: Annotated[str, typer.Argument(help="Variant label category to highlight in the volcano plot (e.g., candidate).")],
    selected_gwas_stats_path: Annotated[str, typer.Argument(help="Path to selected GWAS statistics.")],
    sig_effects_path: Annotated[str, typer.Argument(help="Path to significant effects matrix.")],
    foldchange_effects_path: Annotated[str, typer.Argument(help="Path to fold-change effects matrix.")],
    boot_pvals_path: Annotated[str, typer.Argument(help="Path to bootstrapped p-values matrix.")],
    out_prefix: Annotated[str, typer.Argument(help="Prefix for output files.")],
    alpha: Annotated[Optional[float], typer.Option("-alpha", help="Opacity of the scatter plot points.")] = 0.4,
    verbose: Annotated[bool, typer.Option("-verbose/-quiet", help="Enable or disable verbose output.")] = True
):
    """Volcano plot for Deep Variant Impact Mapping predicted molecular effects."""

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



