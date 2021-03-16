import matplotlib.colors
import numpy as np

colors = {
    'r1_in': 'red',
    'r1_out': 'lightsalmon',
    'r2_in': 'orange',
    'r2_out': 'bisque',
    'r3_in': 'green',
    'r3_out': 'lightgreen',
    'r4_in': 'blue',
    'r4_out': 'lightblue',
    'pellet': 'aqua',
    'insured pellet': 'brown'
}
for k, v in colors.items():
    col = matplotlib.colors.to_rgba_array(v)[0] * 255
    colors[k] = np.array([int(c) for c in col])

sizes = {
    'r1_in': 1,
    'r1_out': 1,
    'r2_in': 1,
    'r2_out': 1,
    'r3_in': 1,
    'r3_out': 1,
    'r4_in': 1,
    'r4_out': 1,
    'pellet': 3,
    'insured pellet': 3
}