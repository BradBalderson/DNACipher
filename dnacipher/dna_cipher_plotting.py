"""Helper functions for plotting."""

import numpy as np
import matplotlib.pyplot as plt

def plot_signals(signals_normed, assay_labels, assay_colors, label_size=9, fontweight=700,
                 y_step=-0.7, title='DNACipher track predictions', plot_delta=False, show=False):
    x = np.arange(signals_normed.shape[0])

    #### Plotting the ref sequence predictions
    fig, ax = plt.subplots(figsize=(17, 5))
    fig.suptitle( title )

    y_offset = 0
    for i, assay_ in enumerate(assay_labels):
        ax.plot(x, signals_normed[:,i]+y_offset, label=assay_, lw=2, c=assay_colors[assay_], alpha=0.9)
        ax.text(max(x), y_offset, assay_, color=assay_colors[assay_], fontsize=label_size,
                fontweight=fontweight)

        if plot_delta: # Assumes inputs represent difference tracks
            delta = round(float( sum(signals_normed[:,i]*100) ), 2) # To make it a percentage
            ax.text(-30, y_offset, #y_offset + np.quantile(signals_normed[:,i], 0.8),
                    f'∆={delta}%',
                    color=assay_colors[assay_], fontsize=label_size * 0.8, fontweight=fontweight)

        y_offset += y_step

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticklabels([])
    ax.grid(False, axis='y')
    if show:
        plt.show()
    else:
        return fig, ax

