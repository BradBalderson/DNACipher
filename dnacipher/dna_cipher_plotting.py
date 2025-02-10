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
        exper_colors = get_colors( exper_labels, color_map=color_map )

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

