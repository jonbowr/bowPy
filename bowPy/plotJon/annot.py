from matplotlib import pyplot as plt


def vline(x,txt = '',txt_props = {},
                        lineprops = {},ax = plt,
                        text_loc = 'top',
                        col = None,rot = 45,sep = .05):
    
    bbox=dict(boxstyle="round",
                alpha = .2,
                fc = col,
                # ha= 'center',
                # va = 'center'
                   # ec=(1., 0.5, 0.5),
                   # fc=(1., 0.8, 0.8),
                   )
    # sep = .05
    ax.text(x,
            (ax.get_ylim()[1]*(1+sep) if 'top' in text_loc else ax.get_ylim()[0]-ax.get_ylim()[1]*sep),
            txt,
            rotation = (rot if 'top' in text_loc else -rot),
            rotation_mode = 'anchor',
            bbox = bbox,
            # transform=ax.transAxes,
            **txt_props)
    ax.axvline(x,color = 'k',linewidth = 2)
    ax.axvline(x,color = col,**lineprops)
