import numpy as np


def sym_logspace(start,stop,num,thresh=1):
    rng = abs(stop-start)

    sfrac = int(np.log(abs(start))/np.log(rng)*num)
    stfrac = int(np.log(abs(stop))/np.log(rng)*num)
    if start <0 and stop >0:
        return(np.sort(np.concatenate([
                        start/abs(start)*np.geomspace(thresh,abs(start),sfrac),
                        np.insert(stop/abs(stop)*np.geomspace(thresh,abs(stop),stfrac),0,0)],
                        )))


def gauss_filt_nan(U,sigma,truncate = 4):
    from scipy.ndimage import gaussian_filter
    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma,truncate=truncate)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=sigma,truncate=truncate)

    return(VV/WW)


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx-1
    else:
        return idx

def std_w(x,w):
    m = np.average(x,weights=w)
    return(np.sqrt(np.average((x-m)**2,weights = w)))