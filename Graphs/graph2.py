import matplotlib.pyplot as plt
import numpy as np
import time

x1 = "Python Q-Learn \nw/o tail"
y1 = 2

x2 = "Python CNN \nw/ tail"
y2 = 24

x3 = "Socket Q-Learn \nw/o tail"
y3 = 48

x4 = "Socket CNN \nw/ tail \n(expected)"
y4 = 576

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# ax.set_title("Tensorflow model training time \nfor different versions of Snake")

plt.ylabel("Training time [hours]", labelpad=5)
# plt.xlabel("Versions", labelpad=5)

ax.bar(x1, y1)
ax2.bar(x1, y1)

ax.bar(x2, y2)
ax2.bar(x2, y2)

ax.bar(x3, y3)
ax2.bar(x3, y3)

ax.bar(x4, y4)
ax2.bar(x4, y4)

# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(rotation=0)


# zoom-in / limit the view to different portions of the data
ax.set_ylim(550, 600)  # outliers only
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

f.savefig("graph.pdf", bbox_inches='tight')