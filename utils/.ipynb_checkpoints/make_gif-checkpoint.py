import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from tqdm import tqdm
from pathlib import Path

import imageio

STATION_COORDS = {
    "CI.WVP2.": (35.94939, -117.81769),
    "CI.WRV2.": (36.00774, -117.8904),
    "CI.WRC2.": (35.9479, -117.65038),
    "CI.WNM.": (35.8422, -117.90616),
    "CI.WMF.": (36.11758, -117.85486),
    "CI.WCS2.": (36.02521, -117.76526),
    "CI.WBM.": (35.60839, -117.89049),
    "CI.TOW2.": (35.80856, -117.76488),
    "CI.SRT.": (35.69235, -117.75051),
    "CI.SLA.": (35.89095, -117.28332),
    "CI.MPM.": (36.05799 ,-117.48901),
    "CI.LRL.": (35.47954, -117.68212),
    "CI.DTP.": (35.26742, -117.84581),
    "CI.CCC.": (35.52495, -117.36453),
    "CI.JRC2.": (35.98249, -117.80885)
}


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles. 
    Similar to plt.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection


    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_)
               for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def plot_earthquake_timeseries(actual_intensities, save_path, im_path, scale, predicted_intensities = None):
    """
    Plots intensities for each timestep of an earthquake
    
    Parameters
    ----------
    actual_intensities: numpy array, shape = (t, 15)
    Actual recorded intensities for at each timestep at each station
    
    save_path: str
    Directory under which generated plots should be stored
    
    im_path: str
    path to background image for gif
    
    scale: int
    Quantity by which to scale the earthquake magnitudes for visualization purposes
    
    predicted_intensities: numpy array, shape = (t, 15)
    Predicted intensities for each timestep at each station
    
    """
    
    lats = [STATION_COORDS[s][0] for s in STATION_COORDS.keys()]
    lngs = [STATION_COORDS[s][1] for s in STATION_COORDS.keys()]
    
    actual_intensities = actual_intensities / 10 * scale
    if predicted_intensities is not None:
        predicted_intensities = predicted_intensities / 10 * scale
        assert(actual_intensities.shape[0] == predicted_intensities.shape[0])
        
    for i in tqdm(range(actual_intensities.shape[0])):
        plt.figure(figsize=(15, 15))
        plt.gca().axis('off')
        img = plt.imread(im_path)
        width = img.shape[0]
        height = img.shape[1]
        latmin = 36.235
        lngmin = -118.046
        plt.imshow(img)

        # Change the x and y to adjust to coords
        y = [int((latmin - lat)*height) for lat in lats]
        x = [int((lng - lngmin)*width) for lng in lngs]
        
        actual_outs = circles(x, y, actual_intensities[i], c='b', alpha=0.5, edgecolor='none')
        if predicted_intensities is not None:
            predicted_outs = circles(x, y, predicted_intensities[i], c='r', alpha=0.5, edgecolor='none')

        plt.savefig(f"{save_path}/{i}.png")
        plt.close()


def generate_gifs(timeseries, save_path, gif_name, im_path, predicted_intensities = None, clear_tmp = True, scale = 1):
    """
    Generates gif of progress of given earthquake and predicted intensities
    
    Parameters
    ----------
    timeseries: numpy array, shape = (t, 15)
    Actual timeseries of the progress of the earthquake
    
    save_path: str
    Directory to save generated gif to
    
    gif_name: str
    Filename of the gif file
    
    im_path: str
    Path to background image for gif
    
    scale: int
    Quantity by which to scale the earthquake magnitudes for visualization purposes
    
    predicted_intensities: numpy array, shape = (t, 15)
    Predicted intensities for the given earthquake, default = None
    
    clear_tmp: bool
    Specifies whether to clear the temporary png files, default = True
    
    Returns
    -------
    gif_path: str
    Path to the generated gif
    """
    print("Generating plots for each timestep...")
    os.makedirs(f"{save_path}/tmp/", exist_ok = True)
    
    plot_earthquake_timeseries(timeseries, f"{save_path}/tmp", im_path, scale, predicted_intensities = predicted_intensities)
        
    images = []
    filenames = sorted(Path(f"{save_path}/tmp/").glob("*.png"), key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    print("Combining plots to generate gif...")
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
        
    gif_path = f'{save_path}/{gif_name}'
    print(gif_path)
    with imageio.get_writer(gif_path, mode='I') as writer:
        for image in images:
            writer.append_data(image)
            
    if clear_tmp:
        print("Clearing temp files...")
        shutil.rmtree(f'{save_path}/tmp')
    
    print(f"Gif stored at: {gif_path}")
    return gif_path
