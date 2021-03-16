import numpy as np
from glob import glob
import pandas as pd
import os.path
from collections import defaultdict
from matplotlib.pyplot import get_cmap
import matplotlib.pyplot as plt
from matplotlib import animation
import tools
from .common import get_video_params, natural_keys
from anipose_scripts import constants


def connect(points, bps, bp_dict, color, plot_args):
    ixs = [bp_dict[bp] for bp in bps]
    ax = plt.gca()

    return ax.plot(xs=points[ixs, 0],
                   ys=points[ixs, 1],
                   zs=points[ixs, 2],
                   color=color,
                   **plot_args
                   )

def connect_all(points, scheme, bp_dict, cmap, plot_args):
    lines = []
    for i, bps in enumerate(scheme):
        line, = connect(points, bps, bp_dict,
                        # color=cmap(i)[:3],
                        color='white',
                        plot_args=plot_args)
        lines.append(line)
    return lines

def update_line(line, points, bps, bp_dict):
    ixs = [bp_dict[bp] for bp in bps]
    new = np.vstack([points[ixs, 0], points[ixs, 1], points[ixs, 2]])
    line.set_data_3d(new)

def update_all_lines(lines, points, scheme, bp_dict):
    for line, bps in zip(lines, scheme):
        update_line(line, points, bps, bp_dict)



def visualize_labels(scheme,
                     labels_fname,
                     outname,
                     optim,
                     fps):
    data = pd.read_csv(labels_fname)
    cols = [x for x in data.columns if '_error' in x]

    if len(scheme) == 0:
        bodyparts = [c.replace('_error', '') for c in cols]
    else:
        bodyparts = sorted(set([x for dx in scheme for x in dx]))

    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
    bp_ix_dict = dict(zip(range(len(bodyparts)), bodyparts))

    all_points = np.array([np.array(data.loc[:, (bp+'_x', bp+'_y', bp+'_z')])
                           for bp in bodyparts], dtype='float64')

    all_errors = np.array([np.array(data.loc[:, bp+'_error'])
                           for bp in bodyparts], dtype='float64')

    all_scores = np.array([np.array(data.loc[:, bp+'_score'])
                           for bp in bodyparts], dtype='float64')

    if optim:
        all_errors[np.isnan(all_errors)] = 0
    else:
        all_errors[np.isnan(all_errors)] = 10000
    good1 = all_errors < 100
    all_scores[np.isnan(all_scores)] = 0
    good2 = np.zeros_like(all_scores)
    for i, scores in enumerate(all_scores):
        if 'pellet' in bodyparts[i]:
            threshold = 0.9
        else:
            threshold = 0.
        good2[i] = scores > threshold
    good = np.logical_and(good1, good2)
    all_points[~good] = np.nan

    all_points_flat = all_points.reshape(-1, 3)
    check = ~np.isnan(all_points_flat[:, 0])

    if np.sum(check) < 10:
        print('too few points to plot, skipping...')
        return
    
    nparts = len(bodyparts)
    framedict = dict(zip(data['fnum'], data.index))

    lines_cmap = get_cmap('tab10')
    points = np.copy(all_points[:, 0])
    points[np.isnan(points)] = 0
    # points = np.ma.masked_where(np.isnan(points), points)

    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Get rid of colored axes planes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    ax.grid(False)

    lim = 5
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)

    pts = ax.scatter(xs=points[:, 0],
                     ys=points[:, 1],
                     zs=points[:, 2],
                     c=[constants.colors[bp][:3] / 255. for bp in bodyparts],
                     s=[constants.sizes[bp] ** 2 for bp in bodyparts],
                     )

    text = fig.text(0, 1, "TEXT", va='top')
    lines = connect_all(points, scheme, bp_dict, lines_cmap,
                        plot_args={'linewidth':1})
    ax.view_init(elev=22, azim=77)
    ax.invert_xaxis()

    def animate(framenum):
        text.set_text(f'{framenum}')
        if framenum in framedict:
            points = all_points[:, framenum]
        else:
            points = np.zeros((nparts, 3))

        copy_points = np.copy(points)
        copy_points[np.isnan(copy_points)] = 1000
        pts._offsets3d = (copy_points[:, 0],
                          copy_points[:, 1],
                          copy_points[:, 2])
        update_all_lines(lines, points, scheme, bp_dict)
        return (pts, *lines)

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   blit=False,
                                   frames=all_points.shape[1])
    anim.save(outname,
              fps=fps,
              extra_args=['-vcodec', 'mjpeg',
                          '-qscale', '0',
                          '-pix_fmt', 'yuvj420p',
                          ])


def process_peter(scheme,
                  optim,
                  video_folder,
                  pose_3d_folder,
                  out_folder,
                  video_ext,
                  cam_regex):
    vid_fnames = glob(os.path.join(video_folder, "*."+video_ext))
    orig_fnames = defaultdict(list)
    for vid in vid_fnames:
        vidname = tools.get_video_name(cam_regex, vid)
        orig_fnames[vidname].append(vid)

    labels_fnames = glob(os.path.join(pose_3d_folder, '*.csv'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)

    if len(labels_fnames) > 0:
        os.makedirs(out_folder, exist_ok=True)

    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]
        out_fname = os.path.join(out_folder, basename + '.avi')

        print(out_fname)

        some_vid = orig_fnames[basename][0]
        params = get_video_params(some_vid)

        visualize_labels(scheme=scheme,
                         labels_fname=fname,
                         outname=out_fname,
                         optim=optim,
                         fps=params['fps'])
