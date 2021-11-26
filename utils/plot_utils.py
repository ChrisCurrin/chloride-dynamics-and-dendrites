""""""
import logging
from string import ascii_letters, ascii_uppercase
from typing import Iterable, Tuple, Union

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes, mlines
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig
from matplotlib.text import Annotation, Text
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset,
                                                   zoomed_inset_axes)
from mpl_toolkits.mplot3d.proj3d import proj_transform
from neuron import h
from shared import create_dir

from utils import settings
from utils.settings import GS_C, GS_R, PAGE_W_FULL, PAGE_H_half

logger = logging.getLogger(__name__)

num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]


def plot_input_events(ax=None, input_events=None, y_offset=0, marker="v"):
    if input_events is None:
        import pandas as pd

        input_events = pd.DataFrame()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    timing_colors = dict()
    # create a dict that maps offset time to color_cycle index
    for key in input_events:
        if "_t" in key:
            t_idx = key.find("_t")
            time_key = key[t_idx : key.find("_", t_idx + 1)]
            if time_key not in timing_colors:
                timing_colors[time_key] = len(timing_colors)
    logger.debug("number of offset time colors:{}".format(len(timing_colors)))
    for key, events in input_events.iteritems():
        if "_t" in key:
            t_idx = key.find("_t")
            time_key = key[t_idx : key.find("_", t_idx + 1)]
            base_color = color_cycle[timing_colors[time_key]]
        else:
            base_color = color_cycle[0]
        color = opacity(50, base_color)  # base color at 50% opacity
        ax.plot(
            events,
            np.ones(len(events)) * y_offset,
            color=color,
            linestyle="none",
            marker=marker,
            markersize=6,
            label=None,
        )


def plot_var(
    x,
    y,
    title=None,
    xlabel=settings.TIME + " " + settings.UNITS(settings.ms),
    ylabel=None,
    ax=None,
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, label=ylabel, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlabel == "time (ms)":
        ax.set_xlim([0, h.tstop])
    return ax


def plot_v(vec_hoc, title=None, ax=None, section=None, location=None, **kwargs):
    from shared import t_vec
    assert t_vec is not None, "t_vec not initialised"
    if section is None:
        section = "soma"
    if location is None:
        location = 0.5
    y = vec_hoc["v"][section][location]
    return plot_var(
        t_vec,
        y,
        title=title,
        ylabel=" ".join([settings.MEMBRANE_POTENTIAL, settings.UNITS(settings.mV)]),
        ax=ax, 
        **kwargs
    )


def plot_cli(vec_hoc, title=None, ax=None, section=None, location=None, **kwargs):
    from shared import t_vec
    assert t_vec is not None, "t_vec not initialised"
    if section is None:
        section = "soma"
    if location is None:
        location = 0.5
    y = vec_hoc["cli"][section][location]
    return plot_var(
        t_vec,
        y,
        title=title,
        ylabel=" ".join([settings.CLI, settings.UNITS(settings.mM)]),
        ax=ax, 
        **kwargs
    )


def fill_gaps(ax, **kwargs):
    logger.info("Filling gaps")
    handles, labels = ax.get_legend_handles_labels()
    ys = {}
    min_has_max = {}
    for line in handles * 2:
        x_values = line._x
        y_values = line._y
        min_x = min(x_values)
        max_x = max(x_values)
        ys[min_x] = y_values[x_values == min_x][0]
        ys[max_x] = y_values[x_values == max_x][0]
        if min_x in min_has_max and min_has_max[min_x] == -10:
            min_has_max[min_x] = True
        else:
            min_has_max[min_x] = -1
        if max_x in min_has_max and min_has_max[max_x] == -1:
            min_has_max[max_x] = True
        else:
            min_has_max[max_x] = -10

    # sort the keys and remove the minimum value
    sorted_keys = sorted(min_has_max)
    for i in range(len(sorted_keys)):
        x1 = sorted_keys[i - 1]
        x2 = sorted_keys[i]
        if (min_has_max[x1] == -10 or min_has_max[x1] is True) and min_has_max[
            x2
        ] == -1:
            y1 = ys[x1]
            y2 = ys[x2]
            ax.plot([x1, x2], [y1, y2], label="none", **kwargs)


def save_fig(plot_dict, cmd_args):
    logger.info("saving figures")
    dir_path = create_dir("output")
    for key, value in plot_dict.items():
        fig = value[0]  # value is tuple of (fig,axes,named_axes)
        # file name is args-[plot_instance]
        name = (
            "{}/{}-[{}]".format(dir_path, cmd_args, key)
            .replace("--", " ", 1)
            .replace(".", "pt")
            .replace(":", "=")
        )
        for file_format in ["pdf"]:
            try:
                fig.savefig("{}.{}".format(name, file_format), papertype="a4")
            except OSError:
                fig.savefig(
                    "{}.{}".format(name.replace(" ", ""), file_format), papertype="a4"
                )

def num2roman(num):
    """convert a number to roman numerals (uppercase)
    from: https://stackoverflow.com/questions/28777219/basic-program-to-convert-integer-to-roman-numerals#28777781"""
    roman = ''

    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i

    return roman


def letter_axes(*ax_list, start='A', subscript=None, repeat_subscript=False, xy=(0.0, 1.0), xycoords='axes fraction',
                **kwargs):
    """ Annotate a list of axes with uppercase letters for figure goodness.

    :param ax_list: Axes which to annotate.
    :type ax_list: Axes or List[Axes]
    :param start: The axis letter to start at. Can also be integer letter index, useful for subscripts: 3 for 'iii'
    :type start: str or int
    :param subscript: The letter to subscript (e.g. 'B') such that ax_list will be (i.e. Bi, Bii, ...)
    :type subscript: str
    :param repeat_subscript: Include the alphabet letter for every subscript. E.g. Ai, Aii, Aiii if `True` otherwise
    Ai, ii, iii if `False`
    :param xy: X and Y coordinate to place the letter. Relative to axes fraction, with bottom left
        (on data plot portion) being (0,0).
    :type xy: Tuple
    :param xycoords: The coordinate frame to use (`any valid value for Axes.annotate`)
    :param kwargs: passed to `Axes.annotate`
    """
    from matplotlib.cbook import flatten

    # get ax_list into a flatten List
    if type(ax_list[0]) is list:
        ax_list = ax_list[0]
    ax_list = list(flatten(ax_list))
    for i, ax in enumerate(ax_list):
        if type(ax) is str and start == 'A':
            # letter passed as arg
            ax_list = ax_list[:i]
            start = ax
            break

    # determine if xy is a list of placements (and should iterate xy along with ax_list, zip-like)
    iter_xy = np.iterable(xy) and np.iterable(xy[0]) and len(xy) == len(ax_list)
    if subscript is not None and start == 'A':
        start_idx = 1
    elif type(start) is int:
        start_idx = start
    else:
        start_idx = ascii_letters.find(start)

    if subscript is None:
        if (type(start) is str and len(start) != 1) or start_idx == -1:
            raise SyntaxError("'start' must be a single letter in the alphabet")
    else:
        if subscript not in ascii_letters or len(subscript) != 1:
            raise SyntaxError("'subscript' must be a single letter in the alphabet")

    for ax_n, ax in enumerate(ax_list):
        idx = ax_n + start_idx
        _letter = f"{num2roman(idx).lower()}" if subscript else ascii_letters[idx]
        if (subscript and repeat_subscript) or (subscript and idx == 1):
            _letter = f"{subscript}{_letter}"
        _xy = xy[ax_n] if iter_xy else xy
        ax.annotate(_letter, xy=_xy, xycoords=xycoords, fontsize='xx-large', **kwargs)


def plot_save(path: Union[str, Iterable], figs: list = None, close=True, tight=False, **kwargs) -> None:
    """Save figures to path (including filename and extension)

    If the extension is .pdf then all figures save to a single multipage PDF file.

    :param path: relative URI path including directory, filename, and extension
    :param figs: Optional list of figures to save (defaults to all of them)
    :param close: Close the figures after saving them (default: True)
    :param tight: call fig.tight_layout() before saving (default: False)
    :param kwargs: other keyword arguments passed to fig.savefig
    :raises IOError: if anything goes wrong (making directory, saving, closing, etc.)
    """
    logger = logging.getLogger("plot_save")
    if not isinstance(path, str) and np.iterable(path):
        if len(path) > 1:
            for i in range(len(path)):
                if i == 0:
                    # do tight_layout once
                    plot_save(path[i], figs, close=False, tight=tight)
                elif i == len(path) - 1:
                    # only close on last path
                    plot_save(path[i], figs, close=close, tight=False)
                else:
                    # neither close nor tight_layout otherwise
                    #   don't want to apply tight_layout multiple times
                    #   can't save a figure if it has already been closed
                    plot_save(path[i], figs, close=False, tight=False)
        else:
            plot_save(path[0], figs, close=close, tight=tight)
    else:
        try:
            import os

            from matplotlib.backends.backend_pdf import PdfPages
            directory = os.path.split(path)[0]
            if not os.path.exists(directory):
                os.makedirs(directory)
            i = 1
            tmp_path = path
            while os.path.exists(tmp_path) and os.path.isfile(tmp_path):
                tmp_path = path.replace(".", "_{}.".format(i))
                i += 1
            path = tmp_path
            if figs is None:
                figs = [plt.figure(n) for n in plt.get_fignums()]
            if path.endswith(".pdf"):
                pp = PdfPages(path)
            else:
                pp = path
            logger.info("saving to {}".format(path))
            from matplotlib.figure import Figure
            fig: Figure
            for f_i, fig in enumerate(figs):
                if tight:
                    fig.tight_layout()
                if path.endswith(".pdf"):
                    pp.savefig(fig, **kwargs)
                else:
                    dpi = kwargs.pop('dpi', 600)
                    if len(figs) > 1:
                        pp = path.replace(".", "_fignum{}.".format(f_i))
                    fig.savefig(pp, dpi=dpi, **kwargs)
            if path.endswith(".pdf"):
                pp.close()
            logger.info('Saved figures [{}]'.format(",".join([str(fig.number) for fig in figs])))
            if close:
                for fig in figs:
                    plt.close(fig)
        except IOError as save_err:
            logger.error('Cannot save figures. Error: {}'.format(save_err))
            raise save_err


def opacity(level, color):
    if level > 1:
        level /= 100
    _opacity = "%0.2X"%round((level*255))  # note: not sure if round or int works better here
    if len(color) == 9:
        # already has an opacity applied, so reset it by removing last 2
        color = color[:-2]
    return color + _opacity


def adjust_spines(ax, spines, position=0, sharedx=False, sharedy=False):
    """ Set custom visibility and positioning of of axes' spines.
    If part of a subplot with shared x or y axis, use `sharedx` or `sharedy` keywords, respectively.

    Noe: see seaborn's `despine`` method for a more modern and robust approach

    :param ax: Axes to adjust. Multidimensional lists and arrays may have unintended consequences.
    :type ax: Axes or List[Axes] or np.ndarray[Axes]
    :param spines: The list of spines to show and adjust by `position`.
    :type spines: List[str]
    :param position: Place the spine out from the data area by the specified number of points.
    :type position: float
    :param sharedx: True if part of a subplot with a shared x-axis. This keeps the x-axis visible, but will remove
        ticks for `ax` unless the correct spine is provided.
        If -1 is provided and ax is a List, then the last axis will have an x-axis.
    :type sharedx: bool
    :param sharedy: True if part of a subplot with a shared y-axis. This keeps the y-axis visible, but will remove
        ticks for `ax` unless the correct spine is provided.
        If -1 is provided and ax is a List, then the last axis will have a y-axis.
    :type sharedy: bool
    """
    if np.iterable(ax):
        ax = np.array(ax)
        for i, sub_axis in enumerate(ax):
            if sharedx or sharedy:
                from matplotlib.axes import SubplotBase
                if isinstance(sub_axis, SubplotBase):
                    sub_axis.label_outer()
                if sharedx >= 1 and i == ax.shape[0] - 1 and 'bottom' not in spines:
                    spines.append('bottom')
                if sharedy >= 1 and (
                        (len(ax.shape) == 1 and i == ax.shape[0] - 1) or (len(ax.shape) == 2 and i == ax.shape[1] - 1))\
                        and 'left' not in spines:
                    spines.append('left')
            adjust_spines(sub_axis, spines, position, sharedx=False, sharedy=False)
        return

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', position))
            spine.set_visible(True)
        else:
            spine.set_visible(False)

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_visible(True)
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left')
    elif 'right' in spines:
        ax.yaxis.set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    else:
        # no yaxis ticks
        # if sharedy <= 0:
        #     ax.yaxis.set_visible(False)
        for label in ax.get_yticklabels(which="both"):
            label.set_visible(False)
        ax.tick_params(axis='y', which='both', left=False, right=False)

    if 'bottom' in spines:
        ax.xaxis.set_visible(True)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
    elif 'top' in spines:
        ax.xaxis.set_visible(True)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    else:
        # no xaxis ticks
        # if sharedx <= 0:
        # ax.xaxis.set_visible(False)
        for label in ax.get_xticklabels(which="both"):
            label.set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)


def colorbar_adjacent(mappable, position="right", size="2%", pad=0.05, orientation='vertical', ax=None, **kwargs):
    """Create colorbar using axes toolkit, but means axes cannot be realigned properly"""
    ax = ax or mappable.axes
    fig: Figure = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size, pad)
    return fig.colorbar(mappable, cax=cax, orientation=orientation, **kwargs)


def colorbar_inset(mappable, position="outer right", size="2%", orientation='vertical', ax=None,
                   inset_axes_kwargs=None, **kwargs):
    """Create colorbar using axes toolkit by insetting the axis
    :param mappable:
    :type mappable: matplotlib.image.AxesImage
    :param position:
    :type position: str
    :param size:
    :type size: str
    :param orientation:
    :type orientation: str
    :param ax:
    :type ax: matplotlib.axes.Axes
    :param kwargs:
    :return: Color bar
    :rtype: matplotlib.colorbar.Colorbar
    """
    ax = ax or mappable.axes
    fig = ax.figure
    if inset_axes_kwargs is None:
        inset_axes_kwargs = {'borderpad': 0}
    if orientation == 'vertical':
        height = "100%"
        width = size
    else:
        height = size
        width = "100%"
    if 'outer' in position:
        # we use bbox to shift the colorbar across the entire image
        bbox = [0., 0., 1., 1.]
        if 'right' in position:
            loc = 'center left'
            bbox[0] = 1.
        elif 'left' in position:
            loc = 'center right'
            bbox[0] = 1.
        elif 'top' in position:
            loc = 'lower left'
            bbox[1] = 1.
            if orientation is None:
                orientation = 'horizontal'
        elif 'bottom' in position:
            loc = 'upper left'
            bbox[1] = 1.
            if orientation is None:
                orientation = 'horizontal'
        else:
            raise ValueError("unrecognised argument for 'position'. "
                             "Valid locations are 'right' (default),'left','top', 'bottom' "
                             "with each supporting 'inner' (default) and 'outer'")
        ax_cbar = inset_axes(ax, width=width, height=height, loc=loc,
                             bbox_to_anchor=bbox, bbox_transform=ax.transAxes, **inset_axes_kwargs)
    else:
        ax_cbar = inset_axes(ax, width=width, height=height, loc=position.replace('inner', '').strip(),
                             **inset_axes_kwargs)
    return fig.colorbar(mappable, cax=ax_cbar, orientation=orientation, **kwargs)


def align_axes(unaligned_axes):
    """
    Change position of unaligned_axes such that they have the same (minimum) width regardless of colorbars
    Especially useful when sharing the x axis
    see https://stackoverflow.com/questions/46694889/matplotlib-sharex-with-colorbar-not-working
    :param unaligned_axes: List of Axes objects to be aligned
    :type unaligned_axes: List[Axes] or ndarray
    """
    # get minimum width
    pos = unaligned_axes[0].get_position()
    for ax in unaligned_axes[1:]:
        pos_check = ax.get_position()
        if pos_check.width < pos.width:
            pos = pos_check
    # realign
    for ax in unaligned_axes:
        pos2 = ax.get_position()
        ax.set_position([pos.x0, pos2.y0, pos.width, pos2.height])


def create_zoom(ax_to_zoom, inset_size, lines=None, loc='lower left', loc1=1, loc2=2,
                xlim=None, ylim=None, xticks=2, yticks=2, ec='C7',
                inset_kwargs=None, box_kwargs=None, connector_kwargs=None, **kwargs):
    """Zoom into an axis with an inset plot

    The valid locations (for `loc`, `loc1`, and `loc2`) are: 'upper right' : 1, 'upper left' : 2, 'lower left' : 3,
    'lower right' : 4,
    'right' : 5, 'center left' : 6, 'center right' : 7, 'lower center' : 8, 'upper center' : 9, 'center' : 10

    :param ax_to_zoom: Source axis which data will be copied from and inset axis inserted.
    :type ax_to_zoom: Axes
    :param inset_size: Zoom factor for `zoomed_inset_axes` if argument is a float, else is width and height,
        respectively, for `inset_axes`.
    :type inset_size: float or tuple
    :param lines: Lines to plot in inset axis. If None, all lines are plotted.
    :type lines: List[Line2D]
    :param loc: Location to place the inset axes.
    :param loc1: Corner to use for connecting the inset axes and the area in the
        parent axes. Pass 'all' to connect all corners (overrides loc2).
    :param loc2: Corner to use for connecting the inset axes and the area in the
        parent axes. Pass 'all' to connect all corners (overrides loc1).
    :param xlim: Limits of x-axis. Also limits data **plotted** when copying to the inset axis.
    :type xlim: float or Tuple[float, float]
    :param ylim: Limits of y-axis.
    :type ylim: float or Tuple[float, float]
    :param xticks: Number of ticks (int) or location of ticks (list) or no x-axis (False).
    :type xticks: int or list
    :param yticks: Number of ticks (int) or location of ticks (list) or no y-axis (False).
    :type yticks: int or list
    :param ec: Edge color for border of zoom and connecting lines
    :param inset_kwargs: Keywords for `inset_axes` or `zoomed_inset_axes`.
            E.g. dict(bbox_to_anchor=(1,1), bbox_transform=ax_to_zoom.transAxes)
    :type inset_kwargs: dict
    :param box_kwargs: Keywords for `mpl_toolkits.axes_grid1.inset_locator.mark_inset`.
        To remove not mark the inset axis at all, set box_kwargs to `'None'`
    :type box_kwargs: dict or str
    :param connector_kwargs: Keywords for connecting lines between inset axis and source axis.
        To remove connecting lines set to '`None'`.
    :type connector_kwargs: dict or str
    :param kwargs: Additional keyword arguments for plotting. See `Axes` keywords.

    :return: inset axis
    :rtype: Axes
    """
    if inset_kwargs is None:
        inset_kwargs = dict(bbox_to_anchor=None, bbox_transform=None)
    elif 'bbox_to_anchor' in inset_kwargs:
        if inset_kwargs['bbox_to_anchor'] is None:
            inset_kwargs['bbox_transform'] = None
        elif 'bbox_transform' not in inset_kwargs or\
                ('bbox_transform' in inset_kwargs and inset_kwargs['bbox_transform'] is None):
            inset_kwargs['bbox_transform'] = ax_to_zoom.transAxes
    if box_kwargs is None:
        box_kwargs = dict()
    if connector_kwargs is None:
        connector_kwargs = dict()

    axes_kwargs = dict(facecolor="white")

    if type(inset_size) is tuple:
        ax_inset: Axes = inset_axes(ax_to_zoom, width=inset_size[0], height=inset_size[1],
                                    loc=loc, axes_kwargs=axes_kwargs, **inset_kwargs)
    else:
        ax_inset: Axes = zoomed_inset_axes(ax_to_zoom, zoom=inset_size,
                                           loc=loc, axes_kwargs=axes_kwargs, **inset_kwargs)
    src = ax_to_zoom if lines is None else lines
    copy_lines(src, ax_inset, xlim=xlim, **kwargs)

    ax_inset.set_xlim(xlim)
    ax_inset.set_ylim(ylim)
    for _ticks, _ticks_axis in zip([xticks, yticks], ['x', 'y']):
        get_axis = ax_inset.get_xaxis if _ticks_axis == 'x' else ax_inset.get_yaxis
        if _ticks:
            if type(_ticks) is int:
                from matplotlib.ticker import MaxNLocator
                get_axis().set_major_locator(MaxNLocator(_ticks))
            else:
                get_axis().set_ticks(_ticks)
        else:
            get_axis().set_visible(False)

    for spine in ax_inset.spines:
        ax_inset.spines[spine].set_visible(True)
        ax_inset.spines[spine].set_color(ec)
    ax_inset.tick_params(axis='both', which='both', color=ec, labelcolor=ec)
    if box_kwargs != 'None':
        box_connectors = []
        if loc1 == 'all' or loc2 == 'all':
            loc1, loc2 = 1, 2
            box_patch, p1, p2 = mark_inset(ax_to_zoom, ax_inset, loc1=loc1, loc2=loc2, ec=ec,
                                           **box_kwargs)
            box_connectors.extend([p1, p2])
            loc1, loc2 = 3, 4

        box_patch, p1, p2 = mark_inset(ax_to_zoom, ax_inset, loc1=loc1, loc2=loc2, ec=ec,
                                       **box_kwargs)
        box_connectors.extend([p1, p2])
        for loc, spine in ax_inset.spines.items():
            spine.set(**box_kwargs)  # consistency between inset border and marked box
        if type(connector_kwargs) is dict:
            for bc in box_connectors:
                # put connectors on original axis instead of new axis so that they go behind new axis
                bc.remove()
                ax_to_zoom.add_patch(bc)
                bc.set_zorder(4)
                bc.set(**connector_kwargs)
        elif connector_kwargs == 'None':
            for bc in box_connectors:
                bc.set(color='None')
    return ax_inset


# noinspection SpellCheckingInspection
line_props = {
    'agg_filter':         'get_agg_filter',
    'alpha':              'get_alpha',
    'antialiased':        'get_antialiased',
    'color':              'get_color',
    'dash_capstyle':      'get_dash_capstyle',
    'dash_joinstyle':     'get_dash_joinstyle',
    'drawstyle':          'get_drawstyle',
    'fillstyle':          'get_fillstyle',
    'label':              'get_label',
    'linestyle':          'get_linestyle',
    'linewidth':          'get_linewidth',
    'marker':             'get_marker',
    'markeredgecolor':    'get_markeredgecolor',
    'markeredgewidth':    'get_markeredgewidth',
    'markerfacecolor':    'get_markerfacecolor',
    'markerfacecoloralt': 'get_markerfacecoloralt',
    'markersize':         'get_markersize',
    'markevery':          'get_markevery',
    'rasterized':         'get_rasterized',
    'solid_capstyle':     'get_solid_capstyle',
    'solid_joinstyle':    'get_solid_joinstyle',
    'zorder':             'get_zorder',
    }


# noinspection PyProtectedMember
def copy_lines(src, ax_dest, xlim=None, xunit=None, rel_lw=2., cmap=None, **kwargs):
    """ Copies the lines from a source Axes `ax_src` to a destination Axes `ax_dest`.
    By default, linewidth is doubled. To disable, set rel_lw to 1 or provide 'lw' keyword.

    :param src: Source Axes or list of Line2D objects.
    :type src: Axes or list
    :param ax_dest: Destination Axes.
    :type ax_dest: Axes
    :param xlim: Set the domain of the axis. This will restrict what is plotted (hence should be accompanied by
        xunit), not just a call to `set_xlim`
    :type xlim: tuple or list or int
    :param xunit: The units used with xlim to restrict the domain.
    :type xunit: Quantity
    :param rel_lw: Relative linewidth change for each line in source. Default doubles the linewidth.
    :type rel_lw: float
    :param kwargs: Keywords to provide to `Axes.plot` that will overwrite source properties. The aliases to use are
        defined in `line_props`.
    """
    from matplotlib import cbook, colors
    kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D._alias_map)
    if isinstance(src, Axes):
        lines = src.get_lines()
    elif np.iterable(src):
        lines = src
    else:
        lines = [src]
    if xlim is not None and xunit is not None:
        if np.iterable(xlim):
            xlim = [int(xlim[0]/xunit), int(xlim[1]/xunit)]
        else:
            xlim = [int(xlim/xunit), -1]
    if cmap is not None and type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    for i, line in enumerate(lines):
        props = {}
        for prop, getter in line_props.items():
            props[prop] = getattr(line, getter)()
        if cmap is not None:
            props['color'] = colors.to_hex(cmap(i), keep_alpha=False)
        props = {**props, **kwargs}
        if 'linewidth' not in kwargs:
            # change relative thickness of line
            props['linewidth'] *= rel_lw
        x, y = line.get_data()
        if xlim is not None and xunit is not None:
            x = x[xlim[0]:xlim[1]]
            y = y[xlim[0]:xlim[1]]
        ax_dest.plot(x, y, **props)


class Annotation3D(Annotation):
    """Annotate the point xyz with text s"""

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    """add anotation text s to to Axes3d ax"""

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
    return tag


def shapeplot2d(h, ax, sections=None, order='pre', cvals=None,
                clim=None, cmap='viridis_r', **kwargs):
    """
    Plots a 2D shapeplot (altered and inherits from PyNeuronToolbox)

    Args:
        h = hocObject to interface with neuron
        ax = matplotlib axis for plotting
        sections = list of h.Section() objects to be plotted
        order = { None= use h.allsec() to get sections
                  'pre'= pre-order traversal of morphology }
        cvals = list/array with values mapped to color by cmap; useful
                for displaying voltage, calcium or some other state
                variable across the shapeplot.
        **kwargs passes on to matplotlib (e.g. color='r' for red lines)

    Returns:
        lines = list of line objects making up shapeplot
    """
    import numbers

    from PyNeuronToolbox.morphology import (allsec_preorder, get_section_path,
                                            interpolate_jagged)

    # Default is to plot all sections.
    if sections is None:
        if order == 'pre':
            sections = allsec_preorder(h)  # Get sections in "pre-order"
        else:
            sections = list(h.allsec())

    # Determine color limits
    if cvals is not None and clim is None:
        cn = [isinstance(cv, numbers.Number) for cv in cvals]
        if any(cn):
            clim = [np.min(cvals[cn]), np.max(cvals[cn])]

    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)

    # Plot each segement as a line
    lines = []
    i = 0
    cvals_idx = 0
    for (i, sec) in enumerate(sections):
        xyz = get_section_path(h, sec)
        seg_paths = interpolate_jagged(xyz, sec.nseg)
        seg_paths[0][0, 0] = 0
        seg_paths[-1] = seg_paths[-1][:-1]
        for (j, path) in enumerate(seg_paths):
            zorder = -j if j > len(seg_paths)/2 else 0
            zorder -= i*sec.nseg
            line, = ax.plot(path[:, 0], path[:, 1], '-k', zorder=zorder, **kwargs)
            if cvals is not None:
                if cvals_idx >= cvals.size:
                    cvals_idx = 0  # repeat from 0
                if isinstance(cvals[cvals_idx], numbers.Number):
                    # map number to colormap
                    col = cmap(int((cvals[cvals_idx] - clim[0])*255/(clim[1] - clim[0])))
                else:
                    # use input directly. E.g. if user specified color with a string.
                    col = cvals[cvals_idx]
                line.set_color(col)
            lines.append(line)
            i += 1
            cvals_idx += 1

    return lines


def recolor_shapeplot2d(ax, cmap, cvals=None, clim=None):
    import numbers

    import pandas as pd
    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    if type(cvals) is pd.Series or type(cvals) is pd.DataFrame:
        cvals = cvals.values
    if clim is None:
        clim = [np.min(cvals), np.max(cvals)]
    divisor = clim[1] - clim[0] if clim[1] != clim[0] else 1
    lines = ax.get_lines() if isinstance(ax, Axes) else ax
    cvals_idx = 0
    for i, line in enumerate(lines):
        if cvals is None:
            cvals = line.get_ydata()
            cvals_idx = 0  # always use 0
        if cvals_idx >= cvals.size:
            cvals_idx = 0  # repeat from 0

        if isinstance(cvals[cvals_idx], numbers.Number):
            # map number to colormap
            col = cmap(int((cvals[cvals_idx] - clim[0])*255/divisor))
        else:
            # use input directly. E.g. if user specified color with a string.
            col = cvals[cvals_idx]
        line.set_color(col)
        cvals_idx += 1


def mark_locations2d(h, section, locs, ax=None, markspec='or', **kwargs):
    """
    Marks one or more locations on along a section. Could be used to
    mark the location of a recording or electrical stimulation.

    Args:
        h = hocObject to interface with neuron
        section = reference to section
        locs = float between 0 and 1, or array of floats
        optional arguments specify details of marker

    Returns:
        line = reference to plotted markers
    """
    from PyNeuronToolbox.morphology import (find_coord, get_section_path,
                                            sequential_spherical)

    # get list of cartesian coordinates specifying section path
    xyz = get_section_path(h, section)
    (r, theta, phi) = sequential_spherical(xyz)
    rcum = np.append(0, np.cumsum(r))

    # convert locs into lengths from the beginning of the path
    if type(locs) is float or type(locs) is np.float64:
        locs = np.array([locs])
    if type(locs) is list:
        locs = np.array(locs)
    lengths = locs*rcum[-1]

    # find cartesian coordinates for markers
    xyz_marks = []
    for targ_length in lengths:
        xyz_marks.append(find_coord(targ_length, xyz, rcum, theta, phi))
    xyz_marks = np.array(xyz_marks)

    # plot markers
    if ax is None:
        ax = plt.gca()
    line, = ax.plot(xyz_marks[:, 0], xyz_marks[:, 1], markspec, **kwargs)
    return line


def adjust_ylabels(ax, x_offset=0):
    """
    Scan all ax list and identify the outmost y-axis position.
    Setting all the labels to that position + x_offset.
    """

    xc = np.inf
    for a in ax:
        xc = min(xc, (a.yaxis.get_label()).get_position()[0])

    for a in ax:
        label = a.yaxis.get_label()
        t = label.get_transform()
        a.yaxis.set_label_coords(xc + x_offset, label.get_position()[1], transform=t)
        label.set_va("top")


def annotate_cols_rows(axes, cols=None, rows=None, row_pad=5):
    if rows is None:
        rows = []
    if cols is None:
        cols = []
    annotate_cols(axes, cols)
    annotate_rows(axes, rows, pad=row_pad)


def annotate_cols(axes, labels):
    """SET COLUMN TITLE
    """
    if len(axes) > 0:
        for ax, col in zip(axes[0], labels):
            ax.set_title(col)


def annotate_rows(axes, labels, pad=5):
    """SET ROW TITLE
    """
    if len(axes) > 0:
        for ax, row in zip(axes[:, 0], labels):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')


def use_scalebar(ax, **kwargs):
    """

    :type ax: np.ndarray or plt.Axes


    """
    from utils.scalebars import add_scalebar
    if type(ax) == list or type(ax) == np.ndarray:
        for sub_ax in ax:
            use_scalebar(sub_ax, **kwargs)
    sb = add_scalebar(ax, **kwargs)
    return sb


def colorline(x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0), linewidth=1., ax=None, **kwargs):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    import matplotlib.collections as mcoll

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def new_gridspec(nrows=GS_R, ncols=GS_C,
                 figsize=(PAGE_W_FULL, PAGE_H_half), fig_kwargs=None,
                 grid_kwargs=None):
    """Figure setup"""
    from matplotlib.figure import Figure
    fig_kwargs = fig_kwargs or dict()
    grid_kwargs = grid_kwargs or dict()
    fig: Figure = plt.figure(figsize=figsize, **fig_kwargs)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, **grid_kwargs)
    return fig, gs


class CurvedText(Text):
    """
    A text object that follows an arbitrary curve.
    see https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib/44521963
    """

    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0], y[0], ' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = Text(0, 0, 'a')
                t.set_alpha(0.0)
            else:
                t = Text(0, 0, c, **kwargs)

            # resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder + 1)

            self.__Characters.append((c, t))
            axes.add_artist(t)

    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c, t in self.__Characters:
            t.set_zorder(self.__zorder + 1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self, renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        # preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW*w)/(figH*h))*(ylim[1] - ylim[0])/(xlim[1] - xlim[0])

        # points of the curve in figure coordinates:
        x_fig, y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i, j) for i, j in zip(self.__x, self.__y)
            ]))
            )

        # point distances in figure coordinates
        x_fig_dist = (x_fig[1:] - x_fig[:-1])
        y_fig_dist = (y_fig[1:] - y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist ** 2 + y_fig_dist ** 2)

        # arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist), 0, 0)

        # angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]), (x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)

        rel_pos = 10
        for c, t in self.__Characters:
            # finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1 = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            # ignore all letters that don't fit:
            if rel_pos + w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            # finding the two data points between which the horizontal
            # center point of the character will be situated
            # left and right indices:
            il = np.where(rel_pos + w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos + w/2 <= l_fig)[0][0]

            # if we exactly hit a data point:
            if ir == il:
                ir += 1

            # how much of the letter width was needed to find il:
            used = l_fig[il] - rel_pos
            rel_pos = l_fig[il]

            # relative distance between il and ir where the center
            # of the character will be
            fraction = (w/2 - used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il] + fraction*(self.__x[ir] - self.__x[il])
            y = self.__y[il] + fraction*(self.__y[ir] - self.__y[il])

            # getting the offset when setting correct vertical alignment
            # in data coordinates
            t.set_va(self.get_va())
            bbox2 = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0] - bbox1d[0])

            # the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [np.cos(rad), np.sin(rad)*aspect],
                [-np.sin(rad)/aspect, np.cos(rad)]
                ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr, rot_mat)

            # setting final position and rotation:
            t.set_position(np.array([x, y]) + drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            # updating rel_pos to right edge of character
            rel_pos += w - used
