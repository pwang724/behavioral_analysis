import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# def update_lines(num):
#     dx, dy, dz = np.random.random((3,)) * 255 * 2 - 255  # replace this line with code to get data from serial line
#     text.set_text("{:d}: [{:.0f},{:.0f},{:.0f}]".format(num, dx, dy, dz))  # for debugging
#     x.append(dx)
#     y.append(dy)
#     z.append(dz)
#     graph._offsets3d = [x, y, z]
#     return graph,
#
#
# x = [0]
# y = [0]
# z = [0]
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# graph = ax.scatter(x, y, z, color='orange')
# text = fig.text(0, 1, "TEXT", va='top')  # for debugging
#
# ax.set_xlim3d(-255, 255)
# ax.set_ylim3d(-255, 255)
# ax.set_zlim3d(-255, 255)
#
# # Creating the Animation object
# ani = animation.FuncAnimation(fig, update_lines, frames=200, interval=50, blit=False)
# plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

x = [1]
y = [1]
z = [1]
pts = ax.scatter(x, y, z, color='orange')

# pts = ax.scatter(xs=points[:, 0],
#                  ys=points[:, 1],
#                  zs=points[:, 2],
#                  c=[cmap(i)[:3] for i in range(points.shape[0])]
#                  )
text = fig.text(0, 1, "TEXT", va='top')


# pts = connect_all(points, scheme, bp_dict, cmap,
#                   plot_args={'marker':'o',
#                              'markersize': 5,
#                              'markerfacecolor':'None',
#                              # 'markeredgecolor':'red',
#                              'linestyle': 'None'})
# lines = connect_all(points, scheme, bp_dict, cmap, plot_args={})
# ax.view_init(elev=22, azim=77)
# ax.invert_xaxis()

# def init():
#     pts = ax.scatter(xs=points[:, 0],
#                      ys=points[:, 1],
#                      zs=points[:, 2],
#                      c=[cmap(i)[:3] for i in range(points.shape[0])])
# update_all_lines(pts, points, scheme, bp_dict)
# update_all_lines(lines, points, scheme, bp_dict)
# return pts, lines[0]
# return pts

def animate(framenum):
    dx, dy, dz = np.random.random((
        3,)) * 255 * 2 - 255  # replace this line with code to get data from serial line
    text.set_text("{:d}: [{:.0f},{:.0f},{:.0f}]".format(framenum, dx, dy,
                                                        dz))  # for debugging
    x.append(dx)
    y.append(dy)
    z.append(dz)
    pts._offsets3d = [x, y, z]
    # if framenum in framedict:
    #     points = all_points[:, framenum]
    # else:
    #     points = np.ones((nparts, 3))*np.nan
    # # pts._offset3d = (points[:,0], points[:,1], points[:,2])
    # pts._offset3d = (framenum, framenum, framenum)

    # update_all_lines(pts, points, scheme, bp_dict)
    # update_all_lines(lines, points, scheme, bp_dict)
    # return pts, lines[0]
    return pts,


anim = animation.FuncAnimation(fig,
                               animate,
                               blit=False,
                               interval=50,
                               frames=50)
plt.show()