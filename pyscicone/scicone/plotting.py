from scicone.constants import *
from scicone.utils import cluster_clones
from scipy.cluster.hierarchy import ward, leaves_list
from scipy.spatial.distance import pdist
import numpy as np
import string
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import seaborn as sns

# sns.set_style("ticks", {"axes.grid": True})
sns.set_style("white")

datacmap = matplotlib.colors.LinearSegmentedColormap.from_list("cmap", BLUE_WHITE_RED)

def get_cnv_cmap(vmax=4, vmid=2):
    # Extend amplification colors beyond 4
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cnvcmap", BLUE_WHITE_RED[:2], vmid+1)

    l = []
    # Deletions
    for i in range(vmid): # deletions
        rgb = cmap(i)
        l.append(matplotlib.colors.rgb2hex(rgb))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cnvcmap", BLUE_WHITE_RED[1:], (vmax-vmid)+1)

    # Amplifications
    for i in range(0, cmap.N):
        rgb = cmap(i)
        l.append(matplotlib.colors.rgb2hex(rgb))
    cmap = matplotlib.colors.ListedColormap(l)

    return cmap


class MatrixPlotter():

    def __init__(self, data, cbar_title="", mode='data', chr_stops_dict=None,
                    labels=None,
                    cluster=False,
                    textfontsize=24,
                    tickfontsize=22,
                    bps=None,
                    figsize=(24,8),
                    dpi=100, # should be defined when writing the table, not when initialising right?
                    vmax=None,
                    vmid=2,
                    bbox_to_anchor=(1.065, 0.0, 1, 1),
                    labels_title='Subclones',
                    clone_label_rotation=0,
                    linewidth_bps = 0.7,
                    width_ratios = [1, 40],
                    gs_wspace = 0.05,
                    gs_index_clone_bar = 0,
                    gs_index_heatmap = 1,
                    tickfontsmallthreshold = 0.015

                    ):

        self.data = data
        self.cbar_title = cbar_title
        self.mode = mode
        self.chr_stops_dict = chr_stops_dict
        self.clone_labels = np.array(labels).ravel() if labels is not None else None
        self.cluster = cluster
        self.textfontsize = textfontsize
        self.tickfontsize = tickfontsize
        self.bps = bps
        self.dpi = dpi
        self.figsize = figsize
        self.vmax = vmax
        self.vmid = vmid
        self.bbox_to_anchor = bbox_to_anchor
        self.labels_title = labels_title
        self.clone_label_rotation = clone_label_rotation
        self.linewidth_bps = linewidth_bps
        self.width_ratios=width_ratios
        self.gs_wspace = gs_wspace
        self.gs_index_clone_bar = gs_index_clone_bar
        self.gs_index_heatmap = gs_index_heatmap
        self.tickfontsmallthreshold  = tickfontsmallthreshold

    def initialise_cnv_colormap(self):

        if self.vmax is None or self.vmax < 4:
            self.vmax = 4
        self.vmid = min(self.vmid, self.vmax - 1)
        self.vmax = int(self.vmax)
        self.vmid = int(self.vmid)
        self.cmap = get_cnv_cmap(vmax=self.vmax, vmid=self.vmid)


    def plot(self):

        if self.mode == 'data':
            self.cmap = datacmap #@todo porbably should be not stored as global
        elif self.mode == 'cnv':
            self.initialise_cnv_colormap()
        else:
            raise AttributeError('mode argument must be one of \'data\' or \'cnv\'')

        self.cmap.set_bad(color='black') # for NaN

        self.data_ = np.array(self.data, copy=True)

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        if self.clone_labels is not None: # Clustering was supplied

            self.labels_ = np.array(self.clone_labels, copy=True)

            if self.mode == 'cnv':
                self.data_, self.labels_ = cluster_clones(self.data_, self.labels_, within_clone=False)
            else:
                self.data_, self.labels_ = cluster_clones(self.data_, self.labels_, within_clone=self.cluster)

            self.clone_ticks = dict()
            self.unique_clone_labels = np.unique(self.labels_)
            self.n_unique_clone_labels = len(self.unique_clone_labels)
            for label in self.unique_clone_labels: # sorted
                # Get last pos
                t = np.where(self.labels_ == label)[0][-1]
                self.clone_ticks[label] = t + 1


            self.gs = GridSpec(1, len(self.width_ratios), wspace=self.gs_wspace, width_ratios=self.width_ratios)

            self.ax_clones = self.fig.add_subplot(self.gs[self.gs_index_clone_bar])
            self.clone_bounds = [0] + list(self.clone_ticks.values())
            self.subnorm = matplotlib.colors.BoundaryNorm(self.clone_bounds, self.n_unique_clone_labels)
            self.clone_colors = list(LABEL_COLORS_DICT.values())[:self.n_unique_clone_labels]

            if '-' in self.unique_clone_labels:
                self.clone_colors = ['black'] + self.clone_colors

            self.clonecmap = matplotlib.colors.ListedColormap(self.clone_colors)

            self.clone_colorbar = matplotlib.colorbar.ColorbarBase(
                self.ax_clones,
                cmap=self.clonecmap,
                norm=self.subnorm,
                boundaries=self.clone_bounds,
                spacing="proportional",
                orientation="vertical",
            )

            self.clone_colorbar.outline.set_visible(False)
            self.clone_colorbar.ax.set_ylabel(self.labels_title, fontsize=self.textfontsize)
            self.clone_colorbar.ax.yaxis.set_label_position("left")

            bounds = self.clone_bounds
            for j, lab in enumerate(self.clone_ticks.keys()):
                self.clone_colorbar.ax.text(
                    self.clone_colorbar.ax.get_xlim()[-1]*0.5,
                    ((bounds[j + 1] + bounds[j]) / 2) ,
                    lab,
                    ha="center",
                    va="center",
                    rotation=self.clone_label_rotation,
                    color="w",
                    fontsize=self.tickfontsize,
                )
            self.clone_colorbar.set_ticks([])
            ax = self.fig.add_subplot(self.gs[self.gs_index_heatmap])
        else:
            ax = plt.gca()

        if self.clone_labels is None and self.cluster:
            Z = ward(pdist(self.data_))
            self.hclust_index = leaves_list(Z)
            self.data_ = self.data_[self.hclust_index]
            self.cell_clustering_linkage = Z

        im = plt.pcolormesh(self.data_, cmap=self.cmap, rasterized=True)
        self.ax_heatmap = ax = plt.gca()
        self.ax_heatmap.set_ylabel('Cells', fontsize=self.textfontsize)
        self.ax_heatmap.set_xlabel('Bins', fontsize=self.textfontsize)

        if self.clone_ticks is not None:
            # Add horizontal lines for clones:
            for loc in self.clone_ticks.values():
                self.ax_heatmap.axhline(loc,c='grey',linewidth=0.5)

        # Plot breakpoints/segments when available:
        if self.bps is not None:
            self.ax_heatmap.vlines(self.bps, *self.ax_heatmap.get_ylim(), colors="k", linestyles="dashed", linewidth=self.linewidth_bps)

        # Plot chromosome boundaries:
        if self.chr_stops_dict is not None:
            self.chr_stops_chrs = list(self.chr_stops_dict.keys())
            self.chr_stops_bins = list(self.chr_stops_dict.values())
            self.chr_means = []
            self.chr_means.append(self.chr_stops_bins[0] / 2)
            self.chr_sizes = []
            prev=0
            for c in range(1, len(self.chr_stops_bins)):
                aux = (self.chr_stops_bins[c] + self.chr_stops_bins[c - 1]) / 2
                self.chr_means.append(aux)
                self.chr_sizes.append(self.chr_stops_bins[c-1]-prev)
                prev = self.chr_stops_bins[c-1]

            # Chromosome labels
            self.chromosome_labels = deepcopy(self.chr_stops_chrs)

            self.major_contig_ticks = ticker.FixedLocator(self.chr_means)
            self.ax_heatmap.xaxis.set_major_locator(self.major_contig_ticks)

            #plt.xticks(major_contig_ticks, np.array(chrs_), rotation=0, fontsize=tickfontsize)
            self.ax_heatmap.set_xticklabels(self.chromosome_labels,
                                            minor=False,
                                            fontsize=self.tickfontsize)

            # Plot chromosome boundaries:
            self.ax_heatmap.vlines(
                list(self.chr_stops_dict.values())[:-1],
                *self.ax_heatmap.get_ylim(),
                linewidth=2.5,
                colors='k')

            self.ax_heatmap.set_xlabel('Chromosomes', fontsize=self.textfontsize)

            # Plot small contig label a bit smaller;;
            for i,(contig,size) in enumerate(zip(self.chromosome_labels, self.chr_sizes)):
                chrom_label = self.ax_heatmap.xaxis.get_major_ticks()[i]
                if type(self.tickfontsmallthreshold) is set:
                    if contig in self.tickfontsmallthreshold:
                        chrom_label.label.set_fontsize( self.tickfontsize*0.5 )
                else:
                    if size/sum(self.chr_sizes)<self.tickfontsmallthreshold:
                        chrom_label.label.set_fontsize( self.tickfontsize*0.5 )

        for tick in self.ax_heatmap.yaxis.get_major_ticks():
            tick.label.set_fontsize(self.tickfontsize)

        if self.clone_labels is not None:
            plt.yticks([])

        # Add CNV colorbar
        axins = inset_axes(
                self.ax_heatmap,
                width="2%",  # width = 5% of parent_bbox width
                height="85%",
                loc="lower left",
                bbox_to_anchor=self.bbox_to_anchor,
                bbox_transform=ax.transAxes,
                borderpad=0,
            )

        self.colorbar = cb = plt.colorbar(im, cax=axins)

        if self.vmax is not None:
            im.set_clim(vmin=0, vmax=self.vmax)

        if self.mode == 'cnv':
            im.set_clim(vmin=0, vmax=self.vmax)
            tick_locs = (np.arange(self.vmax+1) + 0.5)*(self.vmax)/(self.vmax+1)
            cb.set_ticks(tick_locs)
            # cb.set_ticks([0.4, 1.2, 2, 2.8, 3.6])
            ticklabels = np.arange(0, self.vmax+1).astype(int).astype(str)
            ticklabels[-1] = f"{ticklabels[-1]}+"
            cb.set_ticklabels(ticklabels)
        elif mode == 'data':
            if vmax == 2:
                cb.set_ticks([0, 1, 2])
                cb.set_ticklabels(["0", "1", "2+"])

        cb.ax.tick_params(labelsize=self.tickfontsize)
        cb.outline.set_visible(False)
        cb.ax.set_title(self.cbar_title, y=1.01, fontsize=self.textfontsize)




def plot_matrix(data, cbar_title="", mode='data', chr_stops_dict=None,
                labels=None, cluster=False, textfontsize=24, tickfontsize=22,
                bps=None, figsize=(24,8), dpi=100, vmax=None, vmid=2, bbox_to_anchor=(1.065, 0.0, 1, 1),
                labels_title='Subclones',clone_label_rotation=0,
                output_path=None):
    if mode == 'data':
        cmap = datacmap
    elif mode == 'cnv':
        if vmax is None or vmax < 4:
            vmax = 4
        vmid = min(vmid, vmax - 1)
        vmax = int(vmax)
        vmid = int(vmid)
        cmap = get_cnv_cmap(vmax=vmax, vmid=vmid)
    else:
        raise AttributeError('mode argument must be one of \'data\' or \'cnv\'')
    cmap.set_bad(color='black') # for NaN

    data_ = np.array(data, copy=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if labels is not None:
        labels = np.array(labels).ravel()
        labels_ = np.array(labels, copy=True)

        if mode == 'cnv':
            data_, labels_ = cluster_clones(data_, labels_, within_clone=False)
        else:
            data_, labels_ = cluster_clones(data_, labels_, within_clone=cluster)

        ticks = dict()
        unique_labels = np.unique(labels_)
        n_unique_labels = len(unique_labels)
        for label in unique_labels: # sorted
            # Get last pos
            t = np.where(labels_ == label)[0][-1]

            ticks[label] = t + 1
        gs = GridSpec(1, 2, wspace=0.05, width_ratios=[1, 40])
        ax = fig.add_subplot(gs[0])
        bounds = [0] + list(ticks.values())
        subnorm = matplotlib.colors.BoundaryNorm(bounds, n_unique_labels)
        clone_colors = list(LABEL_COLORS_DICT.values())[:n_unique_labels]
        if '-' in unique_labels:
            clone_colors = ['black'] + clone_colors
        clonecmap = matplotlib.colors.ListedColormap(clone_colors)

        cb = matplotlib.colorbar.ColorbarBase(
            ax,
            cmap=clonecmap,
            norm=subnorm,
            boundaries=bounds,
            spacing="proportional",
            orientation="vertical",
        )
        cb.outline.set_visible(False)
        cb.ax.set_ylabel(labels_title, fontsize=textfontsize)
        ax.yaxis.set_label_position("left")
        for j, lab in enumerate(ticks.keys()):
            cb.ax.text(
                0.5,
                ((bounds[j + 1] + bounds[j]) / 2) / bounds[-1],
                lab,
                ha="center",
                va="center",
                rotation=90,
                color="w",
                fontsize=tickfontsize,
            )
        cb.set_ticks([])

        ax = fig.add_subplot(gs[1])
    else:
        ax = plt.gca()

    if labels is None and cluster is True:
        Z = ward(pdist(data_))
        hclust_index = leaves_list(Z)
        data_ = data_[hclust_index]
    im = plt.pcolormesh(data_, cmap=cmap, rasterized=True)
    ax = plt.gca()
    plt.ylabel('Cells', fontsize=textfontsize)
    plt.xlabel('Bins', fontsize=textfontsize)
    if bps is not None:
        ax.vlines(bps, *ax.get_ylim(), colors="k", linestyles="dashed", linewidth=2.)
    if chr_stops_dict is not None:
        chr_stops_chrs = list(chr_stops_dict.keys())
        chr_stops_bins = list(chr_stops_dict.values())
        chr_means = []
        chr_means.append(chr_stops_bins[0] / 2)
        for c in range(1, len(chr_stops_bins)):
            aux = (chr_stops_bins[c] + chr_stops_bins[c - 1]) / 2
            chr_means.append(aux)
        chrs_ = deepcopy(chr_stops_chrs)
        chrs_[12] = f'{chr_stops_chrs[12]} '
        chrs_[12]
        chrs_[20] = ""
        chrs_[21] = f'  {chr_stops_chrs[21]}'
        plt.xticks(chr_means, np.array(chrs_), rotation=0, fontsize=tickfontsize)
        ax.vlines(list(chr_stops_dict.values())[:-1], *ax.get_ylim(), ls='--', linewidth=2.5)
        plt.xlabel('Chromosomes', fontsize=textfontsize)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(tickfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(tickfontsize)
    if labels is not None:
        plt.yticks([])

    axins = inset_axes(
            ax,
            width="2%",  # width = 5% of parent_bbox width
            height="85%",
            loc="lower left",
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
    cb = plt.colorbar(im, cax=axins)
    if vmax is not None:
        im.set_clim(vmin=0, vmax=vmax)

    if mode == 'cnv':
        im.set_clim(vmin=0, vmax=vmax)
        tick_locs = (np.arange(vmax+1) + 0.5)*(vmax)/(vmax+1)
        cb.set_ticks(tick_locs)
        # cb.set_ticks([0.4, 1.2, 2, 2.8, 3.6])
        ticklabels = np.arange(0, vmax+1).astype(int).astype(str)
        ticklabels[-1] = f"{ticklabels[-1]}+"
        cb.set_ticklabels(ticklabels)
    elif mode == 'data':
        if vmax == 2:
            cb.set_ticks([0, 1, 2])
            cb.set_ticklabels(["0", "1", "2+"])

    cb.ax.tick_params(labelsize=tickfontsize)
    cb.outline.set_visible(False)
    cb.ax.set_title(cbar_title, y=1.01, fontsize=textfontsize)

    if output_path is not None:
        print("Creating {}...".format(output_path))
        plt.savefig(output_path, bbox_inches="tight", transparent=False)
        plt.close()
        print("Done.")
    else:
        plt.show()
