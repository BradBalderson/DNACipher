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
from . import deep_variant_impact_mapping as dvim

app = typer.Typer(pretty_exceptions_short=False)

@app.command()
def infer_effects(
    chr_: Annotated[str, typer.Argument(help="Chromosome of variant.")],
    pos: Annotated[int, typer.Argument(help="Position of variant on chromosome.")],
    ref: Annotated[str, typer.Argument(help="Reference allele of variant.")],
    alt: Annotated[str, typer.Argument(help="Alternate allele of variant.")],
    celltypes: Annotated[str, typer.Argument(help="Celltypes to infer effects for. File in format: ct1,ct2,ct3")],
    assays: Annotated[str, typer.Argument(help="Assays to infer effects for. File in format: assay1,assay2,assay3")],
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
        
        for dataframe, name_ in zip(frames, ['context_effects', 'ref_signals', 'alt_signals',
                                             'ref_signals_normed_and_ordered', 'alt_signals_normed_and_ordered', 
                                             'diff_signals']):
            
            out_path = f"{out_prefix}{name_}.txt"
            dataframe.to_csv(out_path, sep='\t')
            
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
        context_effects.to_csv(out_path, sep='\t')
        
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

    """ TODO: 
        * Should implement outputting the positional information as well, so could parameterize moleculear effects as (pos, celltype, assay)
    """

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
    var_pred_effects.to_csv(out_path, sep='\t')
    
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
    
    signal_gwas_stats = pd.read_csv(signal_gwas_stats_path, sep='\t')

    stratified_gwas_stats = dvim.stratify_variants(signal_gwas_stats, var_ref_col=var_ref_col, var_alt_col=var_alt_col,
                                                    var_loc_col=var_loc_col, p_col=p_col, allele_freq_col=allele_freq_col,
                                                    p_cut=p_cut, lowsig_cut=lowsig_cut, n_top=n_top, allele_freq_cut=allele_freq_cut,
                                                    min_bg_variants=min_bg_variants, verbose=verbose)

    out_path = f"{out_prefix}stratified_gwas_stats.txt"
    stratified_gwas_stats.to_csv(out_path, sep='\t')

    if verbose:
        print(f"Wrote {out_path}", file=sys.stdout, flush=True)
    
@app.command()
def impact_map(

):
    pass

@app.command()
def plot_signals(

):
    pass

@app.command()
def plot_effects_matrix(

):
    pass

@app.command()
def plot_variant_stats(

):
    pass

@app.command()
def plot_volcano(

):
    pass

def main():
    root_dir = Path(__file__).parent
    sys.path.append(str(root_dir))
    app()

if __name__ == "__main__":
    main()



