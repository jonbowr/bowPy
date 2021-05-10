from matplotlib import pyplot as plt


def legend_loc(fig,ax,label = '',location = 'rightside_outside',size = '10%',pad = .05):

    leg = ax.legend()

    w = leg.get_window_extent(renderer = fig.canvas.get_renderer()).width/fig.get_window_extent().width
    leg.remove()

    fig.subplots_adjust(right=1-w)
    cax = fig.add_axes([1-w, 0, 1, .98])
    cax.set_axis_off()
    cax.legend(loc = 'upper left',handles = ax.get_lines(),title = label)
    # fig.add_subplot(cax)
