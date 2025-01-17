import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import matplotlib.colors as colors

# Example colors
c1 = (5/255,5/255,153/255)
c2 = (102/255,0/255,204/255)
c3 = (255/255,51/255,204/255)
c4 = (204/255,0/255,0/255)
c5 = (255/255,225/255,0/255)
default_colors = [c1, c2, c3, c4, c5]

def params(n=15):
    '''
    Function to set standard parameters for matplotlib.

    Parameters:
        n (int): Font size for matplotlib.

    '''
    plt.rcParams['font.size'] = n
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams['axes.linewidth'] = 1.9
    plt.rcParams['figure.figsize'] = (13, 7)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1.8
    plt.rcParams['ytick.major.width'] = 1.8   
    plt.rcParams['lines.markeredgewidth'] = 2
    pd.set_option('display.max_columns', None)

def color_gradient(x, input_colors=default_colors):
    '''
    Function to create a color gradient of 5 colors in this case
    
    Parameters:
        x (float): The value from 0 to 1 to assign a color
        input_colors (list): List of tuples or list of strings, optional. The list of colors to use for the gradient. Each color can be specified as a tuple of RGB values or as a string representing a named color. Default is default_colors.
            
    Returns:
        tuple: The RGB values for the color to assign
    
    Raises:
        ValueError: If the input x is not a float in the range [0, 1]
    '''
    size = len(input_colors)
    size_bins = size - 1 
    
    COLORS = []
    for i in range(size):
        
        if type(input_colors[i]) == str:
            c = colors.to_rgba(input_colors[i])
        else:
            c = input_colors[i]
        
        COLORS.append(c)
    
    try:
        x = float(x)
    except ValueError:
        raise ValueError(f'Input {x} should be a float in range [0 , 1]')
        
    if x > 1 or x < 0:
        raise ValueError(f'Input {x} should be in range [0 , 1]')
    
    for i in range(size_bins):
        if x >= i/size_bins and x <= (i+1)/size_bins:
            xeff = x - i/size_bins
            r = COLORS[i][0] * (1 - size_bins * xeff) + COLORS[i+1][0] * size_bins * xeff
            g = COLORS[i][1] * (1 - size_bins * xeff) + COLORS[i+1][1] * size_bins * xeff
            b = COLORS[i][2] * (1 - size_bins * xeff) + COLORS[i+1][2] * size_bins * xeff
            
    return (r, g, b)

def get_colors_multiplot(array, input_colors=default_colors, range=None, logscale=False):
    """
    Returns a list of colors corresponding to each element in the input array.
    
    Parameters:
    - array: list or array-like object containing the values for which colors are needed.
    - input_colors: list of colors to choose from. Default is default_colors.
    - ran: tuple (min, max) specifying the range of values. Default is None, in which case the 
            minimum and maximum values of the array are used.
    
    Returns:
    - output_colors: list of colors corresponding to each element in the input array.
    """
    if logscale:
        array = np.log10(array)
    
    # getting the color of each run
    output_colors = []
   
    if range != None:
        m, M = range[0], range[1]
    else:
        m, M = min(array), max(array)
    
    for i in np.arange(len(array)):
        
        if array[i] > M:
            output_colors.append(color_gradient(1, input_colors))
        elif array [i] < m:
            output_colors.append(color_gradient(0, input_colors))
        else:
            normalized_value = (array[i] - m) / (M - m)
            output_colors.append(color_gradient(normalized_value, input_colors))   
    
    return output_colors


def transparent_cmap(cmap, ranges=[0,1]):
    '''
    Returns a colormap object tuned to transparent.

    Parameters:
        cmap (str): The name of the base colormap.
        ranges (list, optional): The range of transparency values. Defaults to [0, 1].

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The transparent colormap object.
    '''

    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))
    color_array[:,-1] = np.linspace(*ranges, ncolors)

    # building the colormap
    return colors.LinearSegmentedColormap.from_list(name='cmap', colors=color_array)


def create_cmap_from_colors(input_colors):
    '''
    Create a colormap given an array of colors
    
    Parameters:
        colors (list): List of colors to create the colormap from
    
    Returns:
        matplotlib.colors.LinearSegmentedColormap: The created colormap
    '''    
    return colors.LinearSegmentedColormap.from_list('',  input_colors)


def plot_colorbar(fig, ax, array, cmap, label="", logscale=False):
    """
    Add a colorbar to a matplotlib figure.

    Parameters:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axes object where the colorbar will be added.
    - array: The array of values used to determine the color of each element in the colorbar.
    - cmap: The colormap used to map the values in the array to colors.
    - label: The label for the colorbar.

    Returns:
    None
    """
    if logscale == False:
        norm = mpl.colors.Normalize(vmin=min(array), vmax=max(array))
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=label)
    else:
        norm = mpl.colors.LogNorm(vmin=min(array), vmax=max(array))
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=label)

def get_cmap_colors(array, cmap):
    """
    Get normalized values and colors from an array using a specified colormap.

    Parameters:
    array (numpy.ndarray): The input array.
    cmap (matplotlib.colors.Colormap): The colormap to use.

    Returns:
    tuple: A tuple containing the normalization object and the array of colors.
    """

    norm   = mpl.colors.Normalize(vmin=np.min(array), vmax=np.max(array))
    output_colors = mpl.cm.ScalarMappable(norm, cmap).to_rgba(array)    
    
    return norm, output_colors




#  ------------------------------------------------------------------------
#  SOME FUNCTIONS TO DISPLAY SOME PLOTS EASILY
#  ------------------------------------------------------------------------
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
import astropy.units as u
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Wedge
from scipy.stats import norm
from astropy.io import fits

# Colors for the colormaps
cmap_cols = ["b", "darkviolet", "r"]
colormap = create_cmap_from_colors(cmap_cols)

def summary_obs_table(
    obs_table_init,
    hdu_table_init,
    source_coord,
    source_name,
    size_fov,
    coords_pointing,
    obs_ids,
    dir_dl3,
):
    
    display(obs_table_init)
    display(hdu_table_init)
    
    colors = get_colors_multiplot(obs_ids)
  
    table = hdu_table_init[hdu_table_init["OBS_ID"] == obs_ids[0]]
    hdul = fits.open(dir_dl3 + "/" + table["FILE_NAME"][0])
    energies = hdul["EVENTS"].data["ENERGY"]
    nskip = int(len(energies) / 100)
    
    
    nbins_run = 80
    energy_peaks = []
    energy_bins = np.logspace(-2, 2, 500)

    for obs_id in obs_ids:
        energy_peaks_run = []
        table = hdu_table_init[hdu_table_init["OBS_ID"] == obs_id]
        hdul = fits.open(dir_dl3 + "/" + table["FILE_NAME"][0])

        energy = hdul["EVENTS"].data["ENERGY"]
        len_en = int(len(energy)/nbins_run)

        for i in range(nbins_run-1):
            energy_binned = energy[i*len_en:(i+1)*len_en]
            counts = np.histogram(energy[i*len_en:(i+1)*len_en], energy_bins)[0]
            index_max = np.argmax(counts)
            
            if energy_bins[index_max] < 1.5:
                energy_peaks_run.append(energy_bins[index_max])
            else:
                energy_peaks_run.append(np.nan)
        
        energy_peaks.append(energy_peaks_run)
        
    energy_peaks = np.array(energy_peaks)


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))


    ax1.scatter([], [], marker="+", c="k", lw=2, s=100, label="Run pointing")
    ax2.plot(source_coord.ra.deg, source_coord.dec.deg, marker="x", ls="", color="r", label=source_name)


    for i, obs_id in enumerate(obs_ids):
        table = hdu_table_init[hdu_table_init["OBS_ID"] == obs_id]
        hdul = fits.open(dir_dl3 + "/" + table["FILE_NAME"][0])

        alt_pnt = obs_table_init[obs_table_init["OBS_ID"] == obs_id]["ALT_PNT"]
        az_pnt = obs_table_init[obs_table_init["OBS_ID"] == obs_id]["AZ_PNT"]

        ax1.plot(hdul["EVENTS"].data["ALT"][::nskip], hdul["EVENTS"].data["AZ"][::nskip], ".",
                color=colors[i], label=f"Run {obs_id}", alpha=0.5)
        ax1.scatter(alt_pnt, az_pnt, marker="+", c="k", lw=2, s=100, zorder=10)

    for row in obs_table_init:
        coord_pointing = SkyCoord(row["RA_PNT"], row["DEC_PNT"], unit=u.deg)
        ax2.plot(coord_pointing.ra.deg, coord_pointing.dec.deg, marker="+", ls="", color="k")
        RadialSize = SphericalCircle(coord_pointing, size_fov, ls="", ec="none", facecolor="b", alpha=0.2)
        ax2.add_patch(RadialSize)


    init_time = obs_table_init[0]["TSTART"]
    for i, row in enumerate(obs_table_init):
        timespan = np.linspace(row["TSTART"] - init_time, row["TSTOP"] - init_time, nbins_run-1) / 60
        ax3.plot(timespan, energy_peaks[i], label=f"Run {obs_ids[i]}", color=colors[i])     


    for coord in coords_pointing:
        ax2.plot(coord.ra.deg, coord.dec.deg, marker="+", ls="", color="b")

    RadialSize = SphericalCircle(coord_pointing, 0*u.deg, ls="", ec="none", facecolor="b", alpha=0.5,)
    ax2.add_patch(RadialSize)
    ax3.plot([], [], marker="x", color="r", ls="", label=source_name)
    ax3.plot([], [], marker="+", color="k", ls="", label="Run pointings (obs_table)")
    ax3.plot([], [], marker="+", color="b", ls="", label="Run pointings (event reco)")
    ax3.plot([], [], marker="s", color="b", ls="", label=f"{size_fov} FoV", alpha=0.5, ms=10)
    ax3.legend(loc=(1.03, 0), fontsize=7, frameon=False)

    ax1.grid(); ax1.set_xlabel("ALT [deg]"); ax1.set_ylabel("AZ [deg]")
    ax2.grid(); ax2.set_xlabel("RA [deg]"); ax2.set_ylabel("DEC [deg]")
    ax3.grid(); ax3.set_xlabel("Elapsed time [min]"); ax3.set_ylabel("Energy peak [TeV]")
    ax1.set_title("ALT-AZ pointing"); ax2.set_title("RA-DEC pointing"); ax3.set_title("Energy peak evolution")
    fig.tight_layout()
    
    plt.savefig(f"output/summary_obs_table.png", bbox_inches="tight", dpi=300)
    plt.show()


def summary_axes(
    size_fov,
    axis_acceptance_offset,
    axis_acceptance_energy,
    axis_energy,
    axis_energy_true,
):
    
    fig = plt.figure(figsize=(5, 2.3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(326)

    CircleFoV = plt.Circle((0, 0), size_fov.value, color="b", alpha=0.5, ls="", label=f"{size_fov} FoV")
    ax1.add_patch(CircleFoV)

    ax1.set_xlim(-size_fov.value * 1.1, size_fov.value * 1.1)
    ax1.set_ylim(-size_fov.value * 1.1, size_fov.value * 1.1)

    for radius in axis_acceptance_offset.as_plot_edges:
        Circle = plt.Circle((0, 0), radius.value, color="none", ec="k", ls="--")
        ax1.add_patch(Circle)

    edges_energy_bkg = axis_acceptance_energy.as_plot_edges
    ax2.axvspan(edges_energy_bkg[0].value, edges_energy_bkg[-1].value, alpha=0.5, color="c")
    for energy in edges_energy_bkg:
        ax2.axvline(energy.value, ls="--", color="k")

    edges_energy_true = axis_energy_true.as_plot_edges
    for energy in edges_energy_true:
        ax3.axvline(energy.value, ls="-", color="r")   
    edges_energy = axis_energy.as_plot_edges
    for energy in edges_energy:
        ax3.axvline(energy.value, ls="--", color="b")

    ax1.plot([], [], ls="--", color="k", label="Bin edges")
    ax1.plot([], [], ls="-", color="r", label="True energy")
    ax1.plot([], [], ls="--", color="b", label="Reco energy")

    ax1.legend(loc=(2.27, 0.6), frameon=False)
    for ax in [ax2, ax3]:
        ax.set_xscale("log")
        ax.set_xlabel("E [TeV]"); ax.set_yticks([]); 
    ax1.set_xlabel("Fov Lat. [deg]"); ax1.set_ylabel("FoV Lon. [deg]"); 

    ax1.set_title(f"BKG Offset bins ({len(axis_acceptance_offset.center)})")
    ax2.set_title(f"BKG Energy bins ({len(axis_acceptance_energy.center)})")
    ax3.set_title(f"Energy bins ({len(axis_energy.center)}&{len(axis_energy_true.center)})")

    plt.savefig(f"output/summary_axes.png", bbox_inches="tight", dpi=300)
    plt.show()

def summary_excluded_regions(
    obs_table_init,
    source_coord,
    source_name,
    size_fov,
    excluded_regions,
    coords_pointing,
):
    fig, ax = plt.subplots(figsize=(3,3))

    ax.plot(source_coord.ra.deg, source_coord.dec.deg, marker="x", ls="", color="r", label=source_name)

    for row in obs_table_init:
        coord_pointing = SkyCoord(row["RA_PNT"], row["DEC_PNT"], unit=u.deg)
        ax.plot(coord_pointing.ra.deg, coord_pointing.dec.deg, marker="+", ls="", color="k")
    for coord in coords_pointing:
        ax.plot(coord.ra.deg, coord.dec.deg, marker="+", ls="", color="b")
    for excluded_region in excluded_regions:
        ExclusionCircle = SphericalCircle(
            excluded_region.center, excluded_region.radius, ls="", ec="none", facecolor="k", alpha=0.5)
        ax.add_patch(ExclusionCircle)

    Circle = SphericalCircle(coord_pointing, 0*u.deg, ls="", ec="none", facecolor="k", alpha=0.5, 
                                 label=f"Masked region")
    ax.add_patch(Circle)

    ax.plot([], [], marker="+", color="k", ls="", label="Run pointings (obs_table)")
    ax.plot([], [], marker="+", color="b", ls="", label="Run pointings (event reco)")
    ax.legend(loc=(1.03, 0), frameon=False)
    ax.grid(); ax.set_xlabel("RA [deg]"); ax.set_ylabel("DEC [deg]")
    plt.savefig(f"output/summary_excluded_regions.png", bbox_inches="tight", dpi=300)
    plt.show()
    
def summary_acceptance_model(
    e_min,
    e_max,
    obs_ids,
    acceptance_model,
    axis_acceptance_offset,
    axis_acceptance_energy,
    dim_bkg_model,
):
    # Acceptance model can be plotted
    factor = 10**(np.log10(e_min.value) + (np.log10(e_max.value) - np.log10(e_min.value)) / 3) / e_min.value
    energy_plot = [f"{e_min:.3f}", f"{e_min*factor:.3f}", f"{e_min*(factor)**2:.3f}", f"{e_max:.3f}"]
    acceptance_model[obs_ids[0]].plot_at_energy(energy_plot, ncols=2, figsize=(5.5, 5))

    acceptance_data = acceptance_model[obs_ids[0]].data

    if dim_bkg_model == 2:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
        
        acceptance_model[obs_ids[0]].plot(ax=ax3)
        
        ax1.plot(axis_acceptance_offset.center, acceptance_data.sum(axis=0), "k-")
        ax2.plot(axis_acceptance_energy.center, acceptance_data.sum(axis=1), "k-", label="Integrated")

        colors_cmap = get_colors_multiplot(
            axis_acceptance_energy.center.value, input_colors=cmap_cols, logscale=True
        )
        for i in range(len(acceptance_data)):
            ax1.plot(axis_acceptance_offset.center, acceptance_data[i], "-", color=colors_cmap[i]) 

        plot_colorbar(
            fig, ax1, axis_acceptance_energy.center.value, cmap=colormap, label="Energy [TeV]", logscale=True
        )

        colors_cmap_offset = get_colors_multiplot(
            axis_acceptance_offset.center, input_colors=cmap_cols
        )  
        for i in range(len(acceptance_data.T)):
            ax2.plot(axis_acceptance_energy.center, acceptance_data.T[i], "-", c=colors_cmap_offset[i])
        plot_colorbar(
            fig, ax2, axis_acceptance_offset.center.value, cmap=colormap, label="Offset [deg]"
        )

        for ax in [ax1, ax2]:
            ax.set_ylabel("Bkg rate [MeV${}^{-1}$s${}^{-1}$sr${}^{-1}$]")

        ax1.set_xlabel("Energy [TeV]")
        ax2.set_xlabel("Offset [deg]")
        ax2.legend(frameon=False)
        ax1.set_yscale("log")
        ax2.loglog()
        fig.tight_layout()
        
        plt.savefig(f"output/summary_acceptance_model_dim{dim_bkg_model}.png", bbox_inches="tight", dpi=300)
        plt.show()        

    elif dim_bkg_model == 3:
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(4, 3))
        
        colors_cmap = get_colors_multiplot(
            axis_acceptance_energy.center.value, input_colors=cmap_cols, logscale=True
        )
        for i in range(len(acceptance_data)):       
            ax1.plot(axis_acceptance_energy.center, acceptance_data.T[i].sum(axis=0), "-", c=colors_cmap[i])

        plot_colorbar(
            fig, ax1, axis_acceptance_energy.center.value, cmap=colormap, label="Energy [TeV]", logscale=True
        )

        ax1.set_ylabel("Bkg rate [MeV${}^{-1}$s${}^{-1}$sr${}^{-1}$]")
        ax1.set_xlabel("Energy [TeV]")
        ax1.loglog()

        plt.savefig(f"output/summary_acceptance_model_dim{dim_bkg_model}.png", bbox_inches="tight", dpi=300)
        plt.show()
    
def summary_geometry(
    geom,
    exclusion_mask,
    obs_table,
    size_fov,
    coords_pointing,
):    
    ra_center, dec_center = geom.center_coord[0].value, geom.center_coord[1].value
    n_ra, n_dec = geom.data_shape[-1], geom.data_shape[-2]
    width_ra, width_dec = geom.width[0][0].value, geom.width[1][0].value
    edges_energy = [*geom.axes["energy"].edges_min.value, geom.axes["energy"].edges_max.value[-1]]
    
    cmap_transp = transparent_cmap("Greys_r", ranges=[1, 0])
    
    bin_edges_ra = np.linspace(ra_center - width_ra, ra_center + width_ra, n_ra)
    bin_edges_dec = np.linspace(dec_center - width_dec, dec_center + width_dec, n_dec)

    fig = plt.figure(figsize=(6,3))
    ax = plt.subplot(121)
    ax_e = plt.subplot(224)
    
    for ra in bin_edges_ra:
        ax.axvline(ra, lw=0.5, color="dimgray")
    for dec in bin_edges_dec:
        ax.axhline(dec, lw=0.5, color="dimgray")
    
    if exclusion_mask != False:
        ax.pcolormesh(bin_edges_ra, bin_edges_dec, np.flip(exclusion_mask.data[0], axis=1), cmap=cmap_transp, zorder=-10)
        ax.plot([], [], "s", color="k", ls="", label=f"Exclusion zone\n{np.sum(~exclusion_mask.data[0])} bins")
    else:    
        ax.plot([], [], "s", color="k", ls="", label=f"Exclusion zone\n0 bins")

    for energy in edges_energy:
        ax_e.axvline(energy, ls="--", color="k")    
    ax_e.axvspan(edges_energy[0], edges_energy[-1], color="b", alpha=0.4)

    for coord in coords_pointing:
        ax.plot(coord.ra.deg, coord.dec.deg, marker="+", ls="", color="k")
        RadialSize = SphericalCircle(coord, size_fov, ls="-", ec="k", facecolor="none")
        ax.add_patch(RadialSize)
        
    ax_e.set_yticks([])
    ax_e.set_xscale("log")
    ax_e.set_xlabel("E [TeV]")
    
    ax.plot(ra_center, dec_center, "xr", label="Center")
    ax.plot([], [], "-", color="dimgray", lw=0.5, label=f"RA-DEC binning")
    ax.plot([], [], "-", color="k", label=f"FoV {size_fov}")
    ax.plot([], [], marker="+", color="k", ls="", label="Run pointings")
    
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("DEC [deg]")    
    ax.set_xlim(ra_center - width_ra, ra_center + width_ra)
    ax.set_ylim(dec_center - width_dec, dec_center + width_dec)
    ax.legend(loc=(1.03, 0.55), frameon=False)
    ax.set_title(f"Spatial geometry ({n_ra}x{n_dec})")
    ax_e.set_title(f"Energy geometry ({len(edges_energy)-1} bins)")

    plt.savefig(f"output/summary_geometry.png", bbox_inches="tight", dpi=300)
    plt.show()
    

def summary_unstacked_datasets(
    geom,
    unstacked_datasets,
    source_coord,
    coords_pointing,
    source_name,
    obs_ids,
):
    ra_center, dec_center = geom.center_coord[0].value, geom.center_coord[1].value
    n_ra, n_dec = geom.data_shape[-1], geom.data_shape[-2]
    width_ra, width_dec = geom.width[0][0].value, geom.width[1][0].value

    bin_edges_ra = np.linspace(ra_center - width_ra, ra_center + width_ra, n_ra)
    bin_edges_dec = np.linspace(dec_center - width_dec, dec_center + width_dec, n_dec)

    nrows = len(unstacked_datasets) + 1
    ncols = len(unstacked_datasets[0].background.data)

    fig, ax = plt.subplots(nrows, ncols, figsize=(0.7*ncols, 0.7*nrows), sharex=True, sharey=True)
    fig.suptitle("Background maps")
    for i in range(len(ax)):
        for j in range(len(ax[i])):  
            ax[i,j].plot(source_coord.ra.deg, source_coord.dec.deg, marker="x", ms=1.5,
                         ls="", color="r", label=source_name)

            for k in range(len(coords_pointing)):
                coord_pointing = coords_pointing[k]
                if k == i:
                    ax[i,j].plot(coord_pointing.ra.deg, coord_pointing.dec.deg, marker="s", 
                                 ls="", color="b", ms=0.5, zorder=-10)
                else:
                    ax[i,j].plot(coord_pointing.ra.deg, coord_pointing.dec.deg, marker="s", 
                                 ls="", color="gray", ms=0.5, zorder=-11)

            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

            if i == 0:
                energy = geom.axes["energy"].center[j].value
                title = f"{energy:.2f} TeV" if energy < 1 else f"{energy:.1f} TeV"
                ax[i,j].set_title(title, fontsize=7)
            if i == len(ax)-1:
                integrated_bkg = sum([np.array(unstacked_datasets[ii].background.data[j]) 
                                      for ii in range(len(unstacked_datasets))])
                ax[i,j].pcolormesh(bin_edges_ra, bin_edges_dec, np.flip(integrated_bkg, axis=1), 
                                   cmap="cividis", zorder=-20)
            else:
                ax[i,0].set_ylabel(f"Run {obs_ids[i]}", fontsize=5)
                ax[i,j].pcolormesh(bin_edges_ra, bin_edges_dec, 
                                   np.flip(unstacked_datasets[i].background.data[j], axis=1), 
                                   cmap="magma", zorder=-20)
    ax[0,-1].plot([], [], marker="s", color="gray", ls="", label="Run pointings")
    ax[0,-1].plot([], [], marker="s", color="b", ls="", label="Current pointing")
    ax[-1,0].set_ylabel("Stacked", fontsize=5)
    ax[0,-1].legend(loc=(1.03, -0.3), frameon=False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"output/summary_unstacked_3d_bkg.png", bbox_inches="tight", dpi=300)
    plt.show()

    nrows = len(unstacked_datasets) + 1
    ncols = len(unstacked_datasets[0].counts.data)

    fig, ax = plt.subplots(nrows, ncols, figsize=(0.7*ncols, 0.7*nrows), sharex=True, sharey=True)
    fig.suptitle("Count maps")
    for i in range(len(ax)):
        for j in range(len(ax[i])):  
            ax[i,j].plot(source_coord.ra.deg, source_coord.dec.deg, marker="x", ms=1.5,
                         ls="", color="r", label=source_name)

            for k in range(len(coords_pointing)):
                coord_pointing = coords_pointing[k]
                if k == i:
                    ax[i,j].plot(coord_pointing.ra.deg, coord_pointing.dec.deg, marker="s", 
                                 ls="", color="b", ms=0.5, zorder=-10)
                else:
                    ax[i,j].plot(coord_pointing.ra.deg, coord_pointing.dec.deg, marker="s", 
                                 ls="", color="gray", ms=0.5, zorder=-11)

            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

            if i == 0:
                energy = geom.axes['energy'].center[j].value
                title = f"{energy:.2f} TeV" if energy < 1 else f"{energy:.1f} TeV"
                ax[i,j].set_title(title, fontsize=7)
            if i == len(ax)-1:
                integrated_bkg = sum([np.array(unstacked_datasets[ii].counts.data[j]) 
                                      for ii in range(len(unstacked_datasets))])
                ax[i,j].pcolormesh(bin_edges_ra, bin_edges_dec, np.flip(integrated_bkg, axis=1), 
                                   cmap="cividis", zorder=-20)
            else:
                ax[i,0].set_ylabel(f"Run {obs_ids[i]}", fontsize=5)
                ax[i,j].pcolormesh(bin_edges_ra, bin_edges_dec, 
                                   np.flip(unstacked_datasets[i].counts.data[j], axis=1), 
                                   cmap="magma", zorder=-20)
    ax[0,-1].plot([], [], marker="s", color="gray", ls="", label="Run pointings")
    ax[0,-1].plot([], [], marker="s", color="b", ls="", label="Current pointing")
    ax[-1,0].set_ylabel("Stacked", fontsize=5)
    ax[0,-1].legend(loc=(1.03, -0.3), frameon=False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"output/summary_unstacked_3d_counts.png", bbox_inches="tight", dpi=300)
    plt.show()
    
    ncols = len(unstacked_datasets)
    side = 1.8
    sigma = 3

    fig, ax = plt.subplots(4, ncols, figsize=(side * ncols, side * 3.5), sharey="row",
                          gridspec_kw={"height_ratios": [1,2,2,2]})

    for j in range(len(ax[0])):
        ax[0,j].set_title(f"Run {obs_ids[j]}")

        bkg = sum(unstacked_datasets[j].background.data)
        counts = sum(unstacked_datasets[j].counts.data)
        excess = gaussian_filter(sum(unstacked_datasets[j].excess.data), sigma=sigma)

        ax[0,j].plot(bin_edges_ra, np.flip(np.sum(bkg, axis=0)), c="b", label="RA integrated\nBackground")
        ax[0,j].plot(bin_edges_ra, np.flip(np.sum(counts, axis=0)), c="r", label="RA integrated\nCounts")

        ax[1,j].pcolormesh(bin_edges_ra, bin_edges_dec, np.flip(bkg, axis=1), cmap="magma", zorder=-20)
        ax[2,j].pcolormesh(bin_edges_ra, bin_edges_dec, np.flip(counts, axis=1), cmap="magma", zorder=-20)    

        max_excess = np.max(np.abs([np.max(excess), np.min(excess)]))
        ax[3,j].pcolormesh(bin_edges_ra, bin_edges_dec, np.flip(excess, axis=1), cmap="bwr", zorder=-20,
                          vmax=max_excess, vmin=-max_excess)    

        for i in range(len(ax)): 
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            if i == 0:
                ax[i,j].axis("off")
                #ax[i,j].set_yscale("log")
            else:
                ax[i,j].plot(ra_center, dec_center, "xr", label=source_name,)
                for coord in coords_pointing:
                    ax[i,j].plot(coord.ra.deg, coord.dec.deg, marker="+", ms=4, ls="", color="k")

    ax[0,-1].plot([], [], marker="x", color="r", ls="", label=source_name)
    ax[0,-1].plot([], [], marker="+", ms=4, color="k", ls="", label="Run pointings")
    ax[0,-1].legend(loc=(1.04, -1.7), frameon=False)

    ax[1,0].set_ylabel("Background")
    ax[2,0].set_ylabel("Counts")
    ax[3,0].set_ylabel("Excess")     

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"output/summary_unstacked_2d.png", bbox_inches="tight", dpi=300)
    plt.show()
    
    
def summary_ringbackground(
    geom,
    stacked_on_off,
    excess_map,
    significance_map,
    internal_ring_radius,
    width_ring,
    source_coord,
    coords_pointing,
    source_name,
):
    
    ra_center, dec_center = geom.center_coord[0].value, geom.center_coord[1].value
    n_ra, n_dec = geom.data_shape[-1], geom.data_shape[-2]
    width_ra, width_dec = geom.width[0][0].value, geom.width[1][0].value

    bin_edges_ra = np.linspace(ra_center - width_ra, ra_center + width_ra, n_ra)
    bin_edges_dec = np.linspace(dec_center - width_dec, dec_center + width_dec, n_dec)

    fig, ax = plt.subplots(2, 2, figsize=(7,5.3), subplot_kw={"projection": geom.wcs})

    ((ax1, ax2), (ax3, ax4)) = ax

    p1 = ax[0,0].pcolormesh(bin_edges_ra, bin_edges_dec, stacked_on_off.background.data[0], cmap="magma",
                           transform=ax[0,0].get_transform("icrs"))
    p2 = ax[0,1].pcolormesh(bin_edges_ra, bin_edges_dec, stacked_on_off.counts.data[0],     cmap="magma",
                           transform=ax[0,1].get_transform("icrs"))

    excess =  excess_map.data[0]
    max_excess = np.max(np.abs([np.max(excess), np.min(excess)]))
    p3 = ax[1,0].pcolormesh(bin_edges_ra, bin_edges_dec, excess, cmap="bwr", 
                            transform=ax[1,0].get_transform("icrs"), vmax=max_excess, vmin=-max_excess)

    sig = significance_map.data[0]
    max_sig = np.max(np.abs([np.max(sig), np.min(sig)]))
    p4 = ax[1,1].pcolormesh(bin_edges_ra, bin_edges_dec, sig, cmap="bwr", 
                            transform=ax[1,1].get_transform("icrs"), vmax=max_sig, vmin=-max_sig)

    fig.colorbar(p1, ax=ax[0,0])
    fig.colorbar(p2, ax=ax[0,1])
    fig.colorbar(p3, ax=ax[1,0])
    fig.colorbar(p4, ax=ax[1,1])

    center = (source_coord.ra.deg, source_coord.dec.deg)
    ring = Wedge(
        center, internal_ring_radius.value + width_ring.value, 
        0, 360, width=width_ring.value, color="b", alpha=0.4, ls="",
        label=f"Ring BKG Size\nInternal {internal_ring_radius}\nWidth {width_ring}",
        transform=ax[0,1].get_transform("icrs")
    )
    ax[0,1].add_patch(ring)

    for i in range(len(ax)):
        for j in range(len(ax[i])):

            ax[i,j].plot(ra_center, dec_center, "xr", label=source_name, transform=ax[i,j].get_transform("icrs"))
            for coord in coords_pointing:
                ax[i,j].plot(coord.ra.deg, coord.dec.deg, marker="+", ms=4, ls="", color="k",
                       transform=ax[i,j].get_transform("icrs"))

            ax[i,j].grid(lw=0.3, color="k")
            ra_ax = ax[i,j].coords["ra"]
            dec_ax = ax[i,j].coords["dec"]

            if i == 0:
                ra_ax.set_ticklabel_visible(False)
            else:
                ra_ax.set_axislabel("ra [h:m:s]")
            if j == 1:
                dec_ax.set_ticklabel_visible(False)
            else:
                dec_ax.set_axislabel("dec [deg:m:s]")

    ax[0,1].plot([], [], marker="+", ms=4, color="k", ls="", label="Run pointings")
    ax[0,1].legend(loc=(1.35, 0.5), frameon=False)

    ax[0,0].set_title("BKG map")
    ax[0,1].set_title("Counts map")
    ax[1,0].set_title("Excess map")
    ax[1,1].set_title("Significance map")
    plt.subplots_adjust(wspace=0.1, hspace=0.15)

    plt.savefig(f"output/summary_significance_map.png", bbox_inches="tight", dpi=300)
    plt.show()
    
    
def summary_significances(
    significance_all,
    significance_off,
    bkg_mu,
    bkg_std,
):

    nbins = 40

    fig, ax = plt.subplots(figsize=(4.5, 4))

    ax.hist(significance_all, density=True, alpha=0.5, color="r", label="ON+OFF bins", bins=nbins,)
    ax.hist(significance_off, density=True, alpha=0.5, color="b", label="OFF bins", bins=nbins,)

    _x_ = np.linspace(np.min(significance_all), np.max(significance_all), 100)
    _p1_ = norm.pdf(_x_, bkg_mu, bkg_std)
    _p2_ = norm.pdf(_x_, 0, 1)
    ax.plot(_x_, _p1_, lw=2, color="k", label="Normal fit")
    ax.plot(_x_, _p2_, lw=2, color="darkorange", label="Normal\n($\mu=0$, $\sigma=1$)")
    ax.set_title(f"BKG residuals\n$\mu$ = {bkg_mu:3.3f}  $\sigma$={bkg_std:3.3f}", fontsize=14)
    ax.legend(loc=(1.03, 0.7), frameon=False)

    ax.set_xlabel("Significance")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    plt.savefig(f"output/summary_residuals_significances.png", bbox_inches="tight", dpi=300)
    plt.show()
