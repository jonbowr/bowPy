from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm

def hist2d(x,y, bins = [], weights = None, density = False, 
                    plt_mesh_args = {}, fig = None,ax = None,
                                        log = True,
                                        cmap = cm.jet,
                                        # cont_cmap = cm.gist_rainbow
                                        vmin = None,vmax = None,
                                        thresh = .5, levels = 10,
                                        imtype = 'both',colorbar_name = '',
                                        v_func = lambda x: x,
                                        cont_func = lambda x: x,show_cbar = True,
                                        int_x = False,int_y = False):

    if not ax:
        fig,ax = plt.subplots()

    import matplotlib.colors as colors

    cnts,xbins,ybins = np.histogram2d(x,y,bins = bins,weights = weights,density = density)
    cnts[cnts==0]= np.nan
    xb,yb = (n[:-1]+np.diff(n)/2 for n in [xbins,ybins])
    cnts = cnts*(len(x) if density else 1)
    cnts = v_func(cnts)
    mino = (np.nanmin(cnts) if not vmin else vmin)
    maxo = (np.nanmax(cnts) if not vmax else vmax)

    thresh = np.nanmin(cnts[cnts>0])

    if imtype =='vmap' or imtype == 'both':
        im = ax.pcolormesh(xb,yb,
                      cnts.T,cmap = cmap,
                      norm=(colors.SymLogNorm(linthresh=thresh,vmin=mino, vmax=maxo) if log == True else None),
                      **plt_mesh_args)

    cont = []
    if imtype == 'contour' or imtype == 'both':
        from ..numJon.numJon import sym_logspace
        if type(levels)==int and log == True:
            plt_levels = np.geomspace((mino if mino>thresh else thresh),maxo,levels)
        elif type (levels) == int:
            plt_levels = np.linspace(mino,maxo,levels)
        else: plt_levels = levels
        cont = plt.contour(xb,yb,cont_func(cnts.T),
                           norm=(colors.SymLogNorm(linthresh=thresh,vmin=mino, vmax=maxo) if log == True else None),
                         levels = plt_levels,cmap = cmap.reversed())
    if im or cont:
        if show_cbar:
            from matplotlib import ticker
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            # fig.colorbar(cax = cax)
            if cont:
                plt.clabel(cont, inline=True, fontsize=12,fmt = '%1.2e')
            #     cbar = plt.colorbar(cont,ax = ax,label = colorbar_name,cax = cax)
            if im:
                cbar = plt.colorbar(im,ax = ax,label = colorbar_name,cax = cax)

    

    fig.tight_layout()
    return(fig,ax,im)



def density_scatter( x , y, ax = None, sort = True, bins = 20,weights = None, **kwargs ):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins,weights = weights)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , 
                np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    return ax