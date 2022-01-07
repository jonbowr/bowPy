from matplotlib import pyplot as plt
from matplotlib import container


def legend_loc(fig,ax,label = '',location = 'rightside_outside',size = '10%',pad = .05,loc = 'upper left'):

    leg = ax.legend()

    w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).width/fig.get_window_extent().width
    leg.remove()

    fig.subplots_adjust(right=1-w)
    cax = fig.add_axes([1-w, 0, 1, .98])
    cax.set_axis_off()

    # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

    cax.legend(loc = loc,handles = handles,labels = labels,title = label)
    # fig.add_subplot(cax)
