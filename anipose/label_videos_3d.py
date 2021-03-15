import numpy as np
from glob import glob
import pandas as pd
import os.path
from collections import defaultdict
from matplotlib.pyplot import get_cmap
import matplotlib.pyplot as plt
from matplotlib import animation
from .common import make_process_fun, get_nframes, get_video_name, get_video_params, get_data_length, natural_keys


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
                        color=cmap(i)[:3],
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



def visualize_labels(config, labels_fname, outname, fps=300):
    try:
        scheme = config['labeling']['scheme']
    except KeyError:
        scheme = []

    data = pd.read_csv(labels_fname)
    cols = [x for x in data.columns if '_error' in x]

    if len(scheme) == 0:
        bodyparts = [c.replace('_error', '') for c in cols]
    else:
        bodyparts = sorted(set([x for dx in scheme for x in dx]))

    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))

    all_points = np.array([np.array(data.loc[:, (bp+'_x', bp+'_y', bp+'_z')])
                           for bp in bodyparts], dtype='float64')

    all_errors = np.array([np.array(data.loc[:, bp+'_error'])
                           for bp in bodyparts], dtype='float64')

    all_scores = np.array([np.array(data.loc[:, bp+'_score'])
                           for bp in bodyparts], dtype='float64')


    if config['triangulation']['optim']:
        all_errors[np.isnan(all_errors)] = 0
    else:
        all_errors[np.isnan(all_errors)] = 10000
    good = (all_errors < 100)
    all_points[~good] = np.nan

    all_points_flat = all_points.reshape(-1, 3)
    check = ~np.isnan(all_points_flat[:, 0])

    if np.sum(check) < 10:
        print('too few points to plot, skipping...')
        return
    
    low, high = np.percentile(all_points_flat[check], [10, 90], axis=0)

    nparts = len(bodyparts)
    framedict = dict(zip(data['fnum'], data.index))

    points_cmap = get_cmap('tab10')
    lines_cmap = get_cmap('tab10')
    points = np.copy(all_points[:, 0])
    points[0] = low
    points[1] = high

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

    # ax.set_xlim3d(-255, 255)
    # ax.set_ylim3d(-255, 255)
    # ax.set_zlim3d(-255, 255)

    pts = ax.scatter(xs=points[:, 0],
                     ys=points[:, 1],
                     zs=points[:, 2],
                     c=[points_cmap(i)[:3] for i in range(points.shape[0])]
                     )
    text = fig.text(0, 1, "TEXT", va='top')
    lines = connect_all(points, scheme, bp_dict, lines_cmap, plot_args={})
    ax.view_init(elev=22, azim=77)
    ax.invert_xaxis()

    def animate(framenum):
        text.set_text(f'{framenum}')
        if framenum in framedict:
            points = all_points[:, framenum]
        else:
            points = np.ones((nparts, 3))*np.nan
        pts._offsets3d = (points[:,0], points[:,1], points[:,2])
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
                          # '-format', 'auto'
                          ])



def process_session(config, session_path, filtered=False):
    pipeline_videos_raw = config['pipeline']['videos_raw']

    if filtered:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d_filter']
        pipeline_3d = config['pipeline']['pose_3d_filter']
    else:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d']
        pipeline_3d = config['pipeline']['pose_3d']

    video_ext = config['video_extension']

    vid_fnames = glob(os.path.join(session_path,
                                   pipeline_videos_raw, "*."+video_ext))
    orig_fnames = defaultdict(list)
    for vid in vid_fnames:
        vidname = get_video_name(config, vid)
        orig_fnames[vidname].append(vid)

    labels_fnames = glob(os.path.join(session_path,
                                      pipeline_3d, '*.csv'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)


    outdir = os.path.join(session_path, pipeline_videos_labeled_3d)

    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]

        out_fname = os.path.join(outdir, basename+'.avi')

        if os.path.exists(out_fname) and \
           abs(get_nframes(out_fname) - get_data_length(fname)) < 100:
            continue
        print(out_fname)

        some_vid = orig_fnames[basename][0]
        params = get_video_params(some_vid)

        visualize_labels(config, fname, out_fname, params['fps'])


label_videos_3d_all = make_process_fun(process_session, filtered=False)
label_videos_3d_filtered_all = make_process_fun(process_session, filtered=True)
