import numpy as np
import os
import base
import anz
import pandas as pd
import viz


BPS_TRACK_LOCATION = ['r2_in', 'r3_in']
BPS_TRACK_PAW = ['r1_in', 'r1_out',
                 'r2_in', 'r2_out',
                 'r3_in', 'r3_out',
                 'r4_in', 'r4_out']
BPS_TRACK_PELLET = ['pellet']

ANGLES_LOCATION = [
    ['r2_out', 'r2_in', 'r3_in'],
    ['r3_out', 'r3_in', 'r2_in'],
]
ANGLES_PAW = [
    ['r1_out', 'r1_in', 'r2_in'],
    ['r2_out', 'r2_in', 'r3_in'],
    ['r3_out', 'r3_in', 'r4_in'],
    ['r2_out', 'r2_in', 'r1_in'],
    ['r3_out', 'r3_in', 'r2_in'],
    ['r4_out', 'r4_in', 'r3_in']
]
BPS_ANGLES = {'location': ANGLES_LOCATION,
              'paw': ANGLES_PAW,
              'pellet': []}

# inputs
mouse = 'M9'
date = '2021.03.14'
analyzed_dir = os.path.join(r'C:\Users\Peter\Desktop\ANALYZED', mouse, date)
scheme = [['r1_in', 'r1_out'],
          ['r2_in', 'r2_out'],
          ['r3_in', 'r3_out'],
          ['r4_in', 'r4_out'],
          ['r1_in', 'r2_in', 'r3_in', 'r4_in'],
          ['pellet'],
          ['insured pellet']]
save_dir = os.path.join(os.getcwd(), '_FIGURES')

# load data
bps_to_include = np.unique([x for y in scheme for x in y])
bp_dict = {bp: i for i, bp in enumerate(bps_to_include)}
marker_xys_per_video = anz._get_markers(
    os.path.join(analyzed_dir, 'POSE_2D'),
    bps_to_include,
    base.CAMERA_NAMES)
label_regions_per_video, label_names = anz._get_labels(
    os.path.join(analyzed_dir, 'LABELS'))

# grab locations
x_grab_offset = 10
y_grab_offset = 0
grab_bps_ix = [bp_dict[bp] for bp in BPS_TRACK_LOCATION]
grabs = []
for marker_xys, label_regions in zip(marker_xys_per_video,
                                     label_regions_per_video):
    grabs_per_trial = []
    for r in label_regions['grab']:
        scores = marker_xys[1][grab_bps_ix, :, 2]
        xp = marker_xys[0][grab_bps_ix, :, 0] + x_grab_offset
        yp = marker_xys[1][grab_bps_ix, :, 0] + y_grab_offset
        grab_x = anz._get_grab_locations_atomic(xp, scores, r)
        grab_y = anz._get_grab_locations_atomic(yp, scores, r)
        grabs_per_trial.append([grab_x, grab_y])
    grabs.append(grabs_per_trial)

# pellet locations
# Y is defined as away from the mouse, towards the front camera
pellet_ix = bp_dict['pellet']
xs = [marker_xys[0][pellet_ix, :, 0] for marker_xys in marker_xys_per_video]
ys = [marker_xys[1][pellet_ix, :, 0] for marker_xys in marker_xys_per_video]
ss = [marker_xys[0][pellet_ix, :, 2] for marker_xys in marker_xys_per_video]
pellet_x = anz.get_pellet_location(xs, ss, plot=False)
pellet_y = anz.get_pellet_location(ys, ss, plot=False)

# outcomes
csv = os.path.join(analyzed_dir, 'notes.csv')
df = pd.read_csv(csv)
ground_truth = df['result'].to_numpy()
ground_truth = [base.OUTCOME_TO_KEY_DICT[x] if x in base.OUTCOME_TO_KEY_DICT.keys() else x for x in
                ground_truth]
mask = df['mask'].to_numpy() > 0

trial_outcomes = []
grab_outcomes = []
for i, label_regions in enumerate(label_regions_per_video):
    dropped_regions = label_regions['dropped']
    grabbed_regions = label_regions['grab']
    chew_regions = label_regions['chew']
    # chew region has to be greater than 50 consecutive
    chew_regions = [x for x in chew_regions if (x[1]-x[0]) > 50]
    # some insurance pellet drops at the start of trials, filter that out
    dropped_regions = [x for x in dropped_regions if x[0] > 30]
    outcome = anz.outcome_truth_table(dropped_regions,
                                      chew_regions,
                                      grabbed_regions)
    trial_outcomes.append(outcome)
    print(i)
    grab_outcome = anz.grab_truth_table(outcome,
                                        grabbed_regions,
                                        chew_regions,
                                        dropped_regions)
    grab_outcomes.append(grab_outcome)
anz.outcome_diagnostics(ground_truth, trial_outcomes, mask)


grabs_x = np.array([[x[0] for x in gs] for gs in grabs])
grabs_y = np.array([[x[1] for x in gs] for gs in grabs])

viz.plot_pellet_and_grab_locations(pellet_x[mask],
                                   np.array(grabs_x)[mask],
                                   np.array(grab_outcomes)[mask],
                                   xlim=[180, 260],
                                   grid=False,
                                   xlabel='X Coordinate',
                                   ylabel='Trials')
viz.save_fig(os.path.join(save_dir, 'pellet_and_grab_masked'),
             figname=f'{mouse}_{date}_masked')

viz.plot_pellet_and_grab_locations(pellet_x,
                                   np.array(grabs_x),
                                   np.array(grab_outcomes),
                                   xlim=[180, 260],
                                   grid=False,
                                   xlabel='X Coordinate',
                                   ylabel='Trials')
viz.save_fig(os.path.join(save_dir, 'pellet_and_grab_raw'),
             figname=f'{mouse}_{date}_raw')

viz.plot_pellet_and_grab_locations(pellet_y[mask],
                                   np.array(grabs_y)[mask],
                                   np.array(grab_outcomes)[mask],
                                   xlim=[180, 260],
                                   grid=False,
                                   xlabel='X Coordinate',
                                   ylabel='Trials')
viz.save_fig(os.path.join(save_dir, 'pellet_and_grab_y_masked'),
             figname=f'{mouse}_{date}_masked')

viz.plot_pellet_and_grab_locations(pellet_y,
                                   np.array(grabs_y),
                                   np.array(grab_outcomes),
                                   xlim=[180, 260],
                                   grid=False,
                                   xlabel='X Coordinate',
                                   ylabel='Trials')
viz.save_fig(os.path.join(save_dir, 'pellet_and_grab_y_raw'),
             figname=f'{mouse}_{date}_raw')


