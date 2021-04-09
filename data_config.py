class pilot:
    BPS_TRACK_LOCATION = ['r2_in', 'r3_in']
    BPS_TRACK_PAW = ['r1_in', 'r1_out',
                     'r2_in', 'r2_out',
                     'r3_in', 'r3_out',
                     'r4_in', 'r4_out']
    BPS_TRACK_PELLET = ['pellet']


    scheme = [['r1_in', 'r1_out'],
              ['r2_in', 'r2_out'],
              ['r3_in', 'r3_out'],
              ['r4_in', 'r4_out'],
              ['r1_in', 'r2_in', 'r3_in', 'r4_in'],
              ['pellet'],
              ['insured pellet']]

    mice_dict = {}
    mouse = 'M2'
    dates = ['2021_03_07',
             '2021_03_08',
             '2021_03_09',
             '2021_03_10',
             '2021_03_11',
             '2021_03_12']
    epochs = [
        ['2021_03_07', 0, '2021_03_10', 15],
        ['2021_03_10', 15, '2021_03_12', 20],
        ['2021_03_12', 20, '2021_03_12', 60],
    ]
    mice_dict[mouse] = [dates, epochs]

    mouse = 'M4'
    dates = ['2021_03_09',
             '2021_03_10',
             '2021_03_11',
             '2021_03_12']
    epochs = [
        ['2021_03_09', 0, '2021_03_10', 25],
        ['2021_03_10', 25, '2021_03_12', 40],
    ]
    mice_dict[mouse] = [dates, epochs]

    mouse = 'M5'
    dates = ['2021_03_11',
             '2021_03_12',
             '2021_03_14']
    epochs = [
        ['2021_03_11', 0, '2021_03_12', 10],
        ['2021_03_12', 10, '2021_03_14', 10],
        ['2021_03_14', 10, '2021_03_14', 30],
        ['2021_03_14', 30, '2021_03_14', 45],
    ]
    mice_dict[mouse] = [dates, epochs]

    mouse = 'M9'
    dates = ['2021.03.11',
             '2021.03.12',
             '2021.03.14']
    epochs = [
        ['2021.03.11', 0, '2021.03.12', 10],
        ['2021.03.12', 10, '2021.03.14', 15],
        ['2021.03.14', 15, '2021.03.14', 30],
        ['2021.03.14', 30, '2021.03.14', 50],
    ]
    mice_dict[mouse] = [dates, epochs]