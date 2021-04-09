import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import base
import os

plt.rcParams.update({'font.size': 8})

GRABTYPE_TO_COLOR_DICT = {
    base.GRABTYPES.SNATCHED: 'green',
    base.GRABTYPES.DROPPED: 'red',
    base.GRABTYPES.FAIL_WITH_PELLET: 'red',
    base.GRABTYPES.FAIL_POST_DROP: 'darkred',
    base.GRABTYPES.FAIL_POST_SNATCH: 'gray'
}


def save_fig(save_path, figname='', dpi=300, pdf=False, show=False):
    os.makedirs(save_path, exist_ok=True)
    figname = os.path.join(save_path, figname)
    plt.savefig(os.path.join(figname + '.jpg'), dpi=dpi)
    print('Figure saved at: ' + figname)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    if show:
        plt.show()


def plot_pellet_and_grab_locations(pellet_loc,
                                   grab_loc,
                                   grab_outcomes,
                                   xlim,
                                   grid=False,
                                   xlabel='X Coordinate',
                                   ylabel='Trials'):
    '''

    :param grab_colors:
    :param grab_legend:
    :param pellet_loc: list of coordinates per trial
    :param grab_loc: list of list of grabs per trial
    :return:
    '''

    n_trials = len(pellet_loc)
    offsets = np.arange(len(pellet_loc))
    color_pellet = 'black'
    w, h = 5, 10
    thickness = 2
    plt.figure(figsize=(w, h))
    plt.eventplot(pellet_loc.reshape(-1, 1),
                  lineoffsets=offsets,
                  linelengths=np.ones_like(offsets),
                  linewidths=thickness * np.ones_like(offsets),
                  colors=[color_pellet] * len(offsets),
                  orientation='horizontal')

    grab_offsets = []
    grab_xs = []
    # grab_colors = []
    # cmap = ['blue', 'cyan', 'green', 'lime', 'gold', 'orange']
    for i, gs in enumerate(grab_loc):
        for j, grab in enumerate(gs):
            grab_xs.append(grab)
            grab_offsets.append(i)
            # grab_colors.append(cmap[j])
    grab_xs = np.array(grab_xs)
    grab_legend = {k.value: c for k, c in GRABTYPE_TO_COLOR_DICT.items()}
    grab_colors = [GRABTYPE_TO_COLOR_DICT[x] for y in grab_outcomes for x in y]

    plt.eventplot(grab_xs.reshape(-1, 1),
                  lineoffsets=grab_offsets,
                  linelengths=np.ones_like(grab_offsets),
                  linewidths=thickness * np.ones_like(grab_offsets),
                  colors=grab_colors,
                  orientation='horizontal')

    # STYLING
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim([0, n_trials-0.5])
    ax = plt.gca()
    ax.invert_yaxis()
    sns.despine()
    major_ticks = np.arange(0, n_trials, 10)
    minor_ticks = np.arange(0, n_trials, 1)
    ax.set_yticks(major_ticks - 0.5)
    ax.set_yticks(minor_ticks - 0.5, minor=True)
    ax.set_yticklabels(major_ticks)

    if grid:
        xmajor_ticks = np.arange(xlim[0], xlim[1], 10)
        xminor_ticks = np.arange(xlim[0], xlim[1], 1)
        ax.set_xticks(xmajor_ticks)
        ax.set_xticks(xminor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.7)

    # custom_lines = [Line2D([0], [0], color=color_pellet, lw=4)] + \
    #                [Line2D([0], [0], color=x, lw=4) for x in cmap]
    # ax.legend(custom_lines,
    #           ['Pellet Location'] + [f'Grab {i+1}' for i in range(len(cmap))],
    #           frameon=False)
    custom_lines = [Line2D([0], [0], color=color_pellet, lw=4)] + \
                   [Line2D([0], [0], color=x, lw=4) for x in grab_legend.values()]
    ax.legend(custom_lines,
              ['Pellet Location'] + [f'{k}' for k in grab_legend.keys()],
              frameon=False,
              loc="upper right")


def black_theme(fig, ax):
    fig.set_facecolor('black')
    ax.set_facecolor('black')


def threedimstyle(fig, ax, elevation, azimuth):
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    sns.despine()
    ax.invert_zaxis()
    ax.view_init(elevation, azimuth)
    ax.invert_zaxis()

