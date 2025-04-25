"""Helper functions for plotting."""
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_signals(signals_normed, variant_loc=None,
                 exper_colors=None, color_map='Accent', label_size=9, fontweight=700, y_step=None,
                 title='DNACipher track predictions', plot_delta=False, figsize=None,
                 genes=None, regions=None, regions_color='gold', region_names='cCREs',
                 xlabel=None, xtick_freq=200, plot_origins=True, show=True):

    if type(y_step) == type(None):
        y_step = -np.max( np.abs(signals_normed) )

    # Getting required label information from the pandas dataframe index and columns
    exper_labels = list( signals_normed.columns.values )
    if type(exper_colors)==type(None):
        exper_colors = get_colors( np.array(exper_labels), color_map=color_map )

    # Determining xtick labels, which will be the mid-point of each bin
    chr_ = signals_normed.index.values[0].split('_')[0]
    bin_regions = np.array([region.split('_')[1:] for region in signals_normed.index], dtype=int)
    bin_labels = bin_regions.mean(axis=1).astype(int)

    # The chromosome will be the xlabel by default
    if type( xlabel ) == type(None):
        xlabel = signals_normed.index.values[0].split('_')[0]

    # Can now just use the data from signals_normed
    x = np.arange(signals_normed.shape[0])
    signals_normed = signals_normed.values

    #### Plotting the ref sequence predictions
    # The height should depend on the number of assays we are plotting.
    if type(figsize) == type(None):
        figsize = (17, math.ceil(2.5*signals_normed.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle( title )

    #### Plotting the signal tracks
    y_offset = 0
    for i, exper_ in enumerate(exper_labels):
        exper_color = exper_colors[exper_]
        ax.plot(x, signals_normed[:,i] + y_offset, label=exper_, lw=2, c=exper_color, alpha=0.9)

        if plot_delta: # Assumes inputs represent difference tracks
            delta = round(float( sum(signals_normed[:, i]*100) ), 2) # To make it a percentage
            ax.text(-30, y_offset, f'∆={delta}%',
                    color=exper_color, fontsize=label_size * 0.8, fontweight=fontweight)

        if plot_origins:
            ax.plot(x, [y_offset]*len(x), c='k', linestyle=':')

        y_offset += y_step

    text_dist = get_text_height(ax, "ABC", label_size) * .6
    if type(genes) != type(None):
        gene_bodies = genes.iloc[:,2].values == 'gene'
        exons = genes.iloc[:,2].values == 'exon'

        genes_ = genes.loc[gene_bodies,:].copy()
        exons_ = genes.loc[exons,:].copy()

        # Plotting the genes:
        y_min, y_max = plt.ylim()
        y_offset_ = y_max + (y_max-y_min)*.1
        y_step_ = text_dist * 3
        plot_annots(genes_, 'genes', y_offset_, y_step_, fig, ax, label_size * 0.8, fontweight)
        # And the exons in an arrow style:
        plot_annots(exons_, 'exons', y_offset_, y_step_, fig, ax, label_size * 0.8, fontweight)

        #y_offset_genes = y_offset_ + ((genes_.shape[0]*y_step_) / 2)
        ax.text(max(x), y_offset_, 'genes', color='k', fontsize=label_size, fontweight=fontweight)

    #### Adding the variant if provided
    if type(variant_loc)!=type(None):
        if variant_loc[0] != chr_:
            raise Exception("Inputted variant not on the same chromosome.")

        y_min, ymax = plt.ylim()
        y_span = abs(ymax)-abs(y_min)
        y_pad = y_span *.025
        y_start, y_end = (y_min-y_pad), ymax+y_pad

        start, end = bin_regions[0,0], bin_regions[-1,-1]

        variant_pos = ((variant_loc[1] - start) / (end - start)) * signals_normed.shape[0]

        plt.vlines(variant_pos, y_start, y_end, color='k')
        plt.text(variant_pos, y_start, f"{variant_loc[2]}>{variant_loc[3]}",
                 c='k', fontsize=label_size * 0.8,
                 fontweight=fontweight)

    if type(regions)!=type(None):
        # Plotting the regions
        y_offset_regions = ax.get_ylim()[1]
        plot_annots(regions, 'regions', y_offset_regions, 0, fig, ax, label_size * 0.8, fontweight,
                    color=regions_color, width=10)
        ax.text(max(x), y_offset_regions, region_names, color=regions_color, fontsize=label_size, fontweight=fontweight)

    y_offset = 0
    text_dist = get_text_height(ax, "ABC", label_size) * .6
    for i, exper_ in enumerate(exper_labels):
        exper_color = exper_colors[exper_]

        # So the text does not take up much space, will plot cell type then assay..
        celltype, assay = exper_.split('---')

        ax.text(max(x), y_offset+text_dist, celltype, color=exper_color, fontsize=label_size,
                fontweight=fontweight)
        ax.text(max(x), y_offset-text_dist, assay, color=exper_color, fontsize=label_size,
                fontweight=fontweight)

        y_offset += y_step

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticklabels([])
    ax.grid(False, axis='y')

    # Adding the location labels
    xticks = list(range(0, signals_normed.shape[0], xtick_freq))
    xtick_labels = [bin_labels[i] for i in xticks]
    ax.set_xticks(xticks, xtick_labels)

    # Add the x-axis label
    ax.set_xlabel( xlabel, fontdict={'fontsize': label_size*1.2}, #labelpad=10
                   )

    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig, ax

def plot_annots(annots, style, y_offset_, y_step_, fig, ax, label_size, fontweight, color='k', width=None):
    """Adds annotations to the plot."""
    for rowi, (_, row) in enumerate(annots.iterrows()):

        # Line indicating the gene location
        start_end = [row['bin_start'], row['bin_end']]
        if style in ['genes', 'exons']:
            strand_ = row.iloc[6]
            current_gene = row['gene_names']
            if rowi == 0:
                prev_gene = current_gene # Initialize

        else:
            strand_ = '.'

        if style == 'genes':
            ax.hlines(y_offset_, start_end[0], start_end[1], colors=color, linestyle='--')

            # Adding label
            name_bin = start_end[1] if strand_ == '-' else start_end[0]
            plt.text(name_bin, y_offset_, row['gene_names'],
                     c='red', fontsize=label_size, fontweight=fontweight)

            y_offset_ += y_step_

        elif style == 'exons':

            # Put all the exons on the gene annotation, so only move when change gene.
            if current_gene != prev_gene:
                y_offset_ += y_step_
                prev_gene = current_gene

            # Commented out code was for debugging the arrow direction.
            # import copy
            # fig_copy = copy.deepcopy(fig)
            # ax_copy = fig_copy.axes[0]

            range_ = ax.get_ylim()[1] - ax.get_ylim()[0]
            width = range_ / 120
            ax.arrow(
                start_end[-int(strand_=='-')],
                y_offset_,
                (start_end[1]-start_end[0]) * [1, -1][-int(strand_=='-')], # reverses the arrow depending on strand
                0,
                width=width,#.00019,
                facecolor=color,
                head_length=5
            )
            # fig_copy.show()
            # print("here")

        elif style == 'regions':
            # Commented out code was for debugging
            # import copy
            # fig_copy = copy.deepcopy(fig)
            # ax_copy = fig_copy.axes[0]

            range_ = ax.get_ylim()[1] - ax.get_ylim()[0]
            #width = range_ * 4 # 10 looked good.
            ax.hlines(y_offset_, start_end[0], start_end[1], colors=color, linestyle='solid', linewidth=width)
            # fig_copy.show()
            # print("here")

def plot_genes_in_region(genes, chrom, region_start, region_end, y_delta=0, size = 1, size2=1, size3=1):
    """
    Plots gene annotations (gene body + TSS) for all genes within a specified region,
    using the matplotlib scripting interface.
    
    Parameters:
        gtf_file (str): Path to the GTF file.
        chrom (str): Chromosome name (e.g., "chr1").
        region_start (int): Start coordinate of region (1-based).
        region_end (int): End coordinate of region (1-based).
    """
    
    genes = genes[
        (genes['feature'] == 'gene') &
        (genes['chrom'] == chrom) &
        (genes['end'] >= region_start) &
        (genes['start'] <= region_end)
    ].copy()

    # Extract gene names
    genes['gene_name'] = genes['attribute'].str.extract('gene_name "([^"]+)"')

    # Create plot
    #plt.figure(figsize=(8, 2))
    y_delta_original = y_delta
    for _, row in genes.iterrows():
        start = row['start']
        end = row['end']
        strand = row['strand']
        gene_name = row['gene_name']
        tss = start if strand == '+' else end

        # Gene body line
        plt.hlines(y=y_delta, xmin=start, xmax=end, color='black', linewidth=2)

        # TSS tick mark (short vertical line)
        arrow_start_y = (y_delta+0.1)*size
        plt.vlines(x=tss, ymin=y_delta, ymax=arrow_start_y, color='k', linewidth=1)

        # TSS arrow indicating transcription direction
        arrow_dx = 1000 if strand == '+' else -1000
        arrow_start_x = tss
        plt.arrow(
            arrow_start_x, arrow_start_y,
            arrow_dx*size2, 0,
            head_width=0.05*size3, head_length=400*size2,
            fc='k', ec='k', linewidth=0.5*size, length_includes_head=True
        )

        # Gene label
        plt.text(tss, arrow_start_y+(.01*y_delta_original), gene_name, ha='center', fontsize=8)
        
        y_delta += (y_delta*.05)
            
def plot_variant_stats(locus_gwas_stats, y_axis_col, color_by, color_dict=None, color_map='magma', 
                       var_chrom_col='chromosome', var_loc_col='base_pair_location',
                      alpha=.5, order_points=True, reverse_order=False, gtf_df=None, show_legend=True):
    """ Manhattan-like plot of variant information.
    
    Parameters
    ----------
    signal_gwas_stats: pd.DataFrame
        GWAS summary statistics for each of the variants at the GWAS locus, within range model predictions.
    y_axis_col: str
        Column in signal_gwas_stats referring to int or float statistics to plot for each variant.
    color_by: str
        Column in signal_gwas_stats to color each variant by.
    color_dict: dict
        Only relevant if color_by refers to discrete annotations of the variants. Keys are the variant categories, values are color names.
    color_map: str
        If color_by is int or float or color_dict=None, then will use this to determine point colors.
    alpha: float
        Opacity of the points.
    order_points: bool
        Whether to plot the points in an ordered way, most to least frequent label, largest to smallest stat, for example.
    reverse_order: bool
        Reverse default orderering of the points.
    gtf_df: pd.DataFrame
        DataFrame of a gtf file, if provided will plot the genes that intersect the region.
    show_legend: bool
        Whether to show a color legend or not.
    """
    
    # Input checking
    if y_axis_col not in locus_gwas_stats.columns:
        raise Exception(f'{y_axis_col} not in available columns: {list(locus_gwas_stats.columns)}')
        
    elif type(locus_gwas_stats[y_axis_col].values[0]) not in [float, int, np.float64, np.int64]:
        
        raise Exception(f'{y_axis_col} refers to non float or int data: {locus_gwas_stats[y_axis_col]}')
    
    if color_by not in locus_gwas_stats.columns:
        raise Exception(f'{color_by} not in available columns: {list(locus_gwas_stats.columns)}')

    # Setting the colors if not set
    is_discrete_colors = type(locus_gwas_stats[color_by].values[0])==str
    if type(color_dict)==type(None) and is_discrete_colors:
        color_dict = get_colors(locus_gwas_stats[color_by].values, color_map=color_map)
    
    color_stats = locus_gwas_stats[color_by].values
    if type(color_dict) != type(None) and is_discrete_colors:
        point_colors = [color_dict[color_stat] for color_stat in color_stats]
    else:
        point_colors = color_stats # Color by the continuous stats.
    
    # Plotting discrete variant labels.
    if is_discrete_colors:
        
        labels = locus_gwas_stats[color_by].values
        label_set = np.unique(labels)
        
        if order_points:
            label_counts = np.array([len(np.where(labels==label_)[0]) for label_ in label_set])
            label_set = label_set[np.argsort(-label_counts)]
            if reverse_order:
                label_set = label_set[::-1]
        
        for label_ in label_set:
            plt.scatter(locus_gwas_stats[ var_loc_col ].values[labels==label_], 
                        locus_gwas_stats[y_axis_col].values[labels==label_],
                        c=color_dict[label_], alpha=alpha, label=label_)
        
        if show_legend:
            plt.legend(loc='upper left', #bbox_to_anchor=(1, 0.5), 
                       fontsize=8)
            
    else:
        
        values_ = locus_gwas_stats[color_by].values
        if order_points:
            index_order = np.argsort(values_)
            if reverse_order:
                index_order = index_order[::-1]
                
        else:
            index_order = list(range(len(values_)))
            
        sc = plt.scatter(locus_gwas_stats[ var_loc_col ].values[index_order], 
                         locus_gwas_stats[ y_axis_col ].values[index_order],
                        c=values_[index_order], alpha=alpha, cmap=color_map)
        if show_legend:
            plt.colorbar(sc, label=color_by)
        
    ylims = plt.ylim()
    midpoint = np.mean(ylims)
    
    xlims = plt.xlim()
    
    chromosome = locus_gwas_stats[ var_chrom_col ].values[0]
    if type(gtf_df) != type(None):
        chromosome = f"chr{ str(chromosome).strip('chr').strip('chromosome') }"
        plot_genes_in_region(gtf_df, chromosome, xlims[0], xlims[1], y_delta=midpoint)

    plt.ylabel( y_axis_col )
    plt.xlabel( chromosome )
    
def plot_volcano(plot_label, selected_gwas_stats, sig_effects, foldchange_effects, boot_pvals_df,
                down_color='dodgerblue', up_color='tomato', nonsig_color='grey', alpha=.4, show=True):
    """ Volcano plot of significant effects.
    """ 
    
    plot_bool = selected_gwas_stats['var_label'].values == plot_label

    foldchange_flat = foldchange_effects.values[plot_bool,:].ravel()
    signal_ps_flat = -np.log10( boot_pvals_df.values[plot_bool,:].ravel() )
    sig_bool_flat = sig_effects.values[plot_bool,:].ravel()

    up_bool = np.logical_and(sig_bool_flat, foldchange_flat > 0)
    down_bool = np.logical_and(sig_bool_flat, foldchange_flat < 0)

    plt.title(f"{plot_label} variants significant effects")
    plt.scatter(foldchange_flat[~sig_bool_flat], signal_ps_flat[~sig_bool_flat], color=nonsig_color, alpha=alpha)
    plt.scatter(foldchange_flat[down_bool], signal_ps_flat[down_bool], color=down_color, alpha=alpha)
    plt.scatter(foldchange_flat[up_bool], signal_ps_flat[up_bool], color=up_color, alpha=alpha)
    plt.xlabel('fold-change')
    plt.ylabel('-log10(p_val)')
    if show:
        plt.show()
    

def get_text_height(ax, text, fontsize):
    """Estimate text width in data coordinates."""
    text_obj = ax.text(0, 0, text, fontsize=fontsize, alpha=0)  # Invisible text
    renderer = ax.figure.canvas.get_renderer()
    bbox = text_obj.get_window_extent(renderer)
    text_height = bbox.height / ax.figure.dpi  # Convert from pixels to inches
    text_obj.remove()

    # Convert inches to data coordinates
    ylim = ax.get_ylim()
    fig_height = ax.figure.get_size_inches()[1]
    return (text_height / fig_height) * (ylim[1] - ylim[0])

def get_colors(labels, color_map='tab20', rgb=False):
    """ Gets an OrderedDict of colors; the order indicates the frequency of \
        labels from largest to smallest. Apologies for snakeCase here, older code I wrote.

        Args:
            labels (numpy.array<str>): Indicates a set of labels for observations.

            color_map (str): A matplotlib colormap.

            rgb (bool): If True, colors indicated by rgb value, if false hexcode.

        Returns:
            dict<str, tuple or str>: An ordered dict indicating the labels which \
                        occur most to least frequently and the associated colors.
    """
    # Determining the set of labels
    labelSet = get_ordered_label_set( labels )

    # Initialising the ordered dict #
    cellTypeColors = {}

    # Ordering the cells according to their frequency and obtaining colors #
    nLabels = len(labelSet)
    cmap = plt.cm.get_cmap(color_map, nLabels)
    rgbs = [cmap(i)[:3] for i in range(nLabels)]

    # Populating the color dictionary with rgb values or hexcodes #
    for i in range( len(labelSet) ):
        cellType = labelSet[i]
        rgbi = rgbs[i]
        if not rgb:
            cellTypeColors[cellType] = matplotlib.colors.rgb2hex(rgbi)
        else:
            cellTypeColors[cellType] = rgbi

    return cellTypeColors

def get_ordered_label_set( labels ):
	""" Gets the set of labels associated with labels ordered from most to \
		least frequent.
	"""
	label_set = list(set(labels))
	freqs = np.array([len(np.where(labels == label)[0])
						 for label in label_set])
	order = np.array(list((-freqs).argsort()))
	label_set_ordered = list(np.array(label_set)[order])

	return label_set_ordered

