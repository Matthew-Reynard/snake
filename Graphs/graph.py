"""
Broken axis example, where the y-axis will have a portion cut out.
"""
import matplotlib.pyplot as plt
import numpy as np
import time
'''

# 30 points between [0, 0.2) originally made using np.random.rand(30)*.2
pts = np.array([
    0.015, 0.166, 0.133, 0.159, 0.041, 0.024, 0.195, 0.039, 0.161, 0.018,
    0.143, 0.056, 0.125, 0.096, 0.094, 0.051, 0.043, 0.021, 0.138, 0.075,
    0.109, 0.195, 0.050, 0.074, 0.079, 0.155, 0.020, 0.010, 0.061, 0.008])



# Now let's make two outlier points which are far away from everything.
pts[[3, 14]] += .8

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax.plot(pts)
ax2.plot(pts)

# zoom-in / limit the view to different portions of the data
ax.set_ylim(.78, 1.)  # outliers only
ax2.set_ylim(0, .22)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = .005  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

plt.show()
'''


xpts = np.array([0, 100000])
ypts = np.array([0, 1100])

xpts1M = np.array([0, 1000000])
ypts1M = np.array([0, 48])

xpts2 = np.array([0, 100000])
ypts2 = np.array([0, 24])

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax.plot(xpts, ypts)
ax2.plot(xpts, ypts)

plt.plot(xpts1M, ypts1M)

plt.plot(xpts2, ypts2)



# zoom-in / limit the view to different portions of the data
ax.set_ylim(1075, 1125)  # outliers only
ax2.set_ylim(0, 50) # most of the data

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# plt.plot([0, 100000], [0, 24])
# plt.plot([0, 100000], [0, 48])
# plt.plot([0, 1000000], [0, 24])

d = .02  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.show()





# def get_memory(t):
#     "Simulate a function that returns system memory"
#     return 100 * (0.5 + 0.5 * np.sin(0.5 * np.pi * t))


# def get_cpu(t):
#     "Simulate a function that returns cpu usage"
#     return 100 * (0.5 + 0.5 * np.sin(0.2 * np.pi * (t - 0.25)))


# def get_net(t):
#     "Simulate a function that returns network bandwidth"
#     return 100 * (0.5 + 0.5 * np.sin(0.7 * np.pi * (t - 0.1)))


# def get_stats(t):
#     return get_memory(t), get_cpu(t), get_net(t)

# fig, ax = plt.subplots()
# ind = np.arange(1, 4)

# # show the figure, but do not block
# plt.show(block=False)


# pm, pc, pn = plt.bar(ind, get_stats(0))
# pm.set_facecolor('r')
# pc.set_facecolor('g')
# pn.set_facecolor('b')
# ax.set_xticks(ind)
# ax.set_xticklabels(['Memory', 'CPU', 'Bandwidth'])
# ax.set_ylim([0, 100])
# ax.set_ylabel('Percent usage')
# ax.set_title('System Monitor')

# start = time.time()
# for i in range(200):  # run for a little while
#     m, c, n = get_stats(i / 10.0)

#     # update the animated artists
#     pm.set_height(m)
#     pc.set_height(c)
#     pn.set_height(n)

#     # ask the canvas to re-draw itself the next time it
#     # has a chance.
#     # For most of the GUI backends this adds an event to the queue
#     # of the GUI frameworks event loop.
#     fig.canvas.draw_idle()
#     try:
#         # make sure that the GUI framework has a chance to run its event loop
#         # and clear any GUI events.  This needs to be in a try/except block
#         # because the default implementation of this method is to raise
#         # NotImplementedError
#         fig.canvas.flush_events()
#     except NotImplementedError:
#         pass

# stop = time.time()
# print("{fps:.1f} frames per second".format(fps=200 / (stop - start)))