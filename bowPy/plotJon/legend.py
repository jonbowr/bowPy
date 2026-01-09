from matplotlib import pyplot as plt
from matplotlib import container


def legend_loc(fig,ax,label = '',location = 'right',size = '10%',pad = .1):

    # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

    leg = ax.legend(handles = handles,labels = labels,title = label)

    figsize = fig.get_size_inches()

    if location == 'right':
        w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).width/fig.get_window_extent().width*(1+pad)
        leg.remove()
        fig.set_figwidth(figsize[0]*(1+w))

        w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).width/fig.get_window_extent().width*(1+pad)
        fig.subplots_adjust(right=1-w)
        cax = fig.add_axes([1-w, 0, 1, .98])
        loc = 'upper left'
    elif location =='below':
        w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).height/fig.get_window_extent().height*(1+pad)
        fig.set_figheight(figsize[1]*(1+w))

        w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).height/fig.get_window_extent().height*(1+pad)
        fig.subplots_adjust(bottom=w+fig.subplotpars.hspace)
        cax = fig.add_axes([0, 0, 1, w])
        loc = 'lower left'
    elif location =='above':
        w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).height/fig.get_window_extent().height*(1+pad)
        fig.set_figheight(figsize[1]*(1+w))

        w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).height/fig.get_window_extent().height*(1+pad)
        fig.subplots_adjust(top=1-w)
        cax = fig.add_axes([0, 1-w, 1, 1])
        loc = 'lower left'
    leg.remove()

    cax.set_axis_off()

    cax.legend(loc = loc,handles = handles,labels = labels,title = label)
