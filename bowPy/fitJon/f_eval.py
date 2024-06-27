import numpy as np
from ..numJon.numJon import find_nearest

def fwhm(fit,xx,param=[],return_locs = False):
    ex = xx.copy()[~np.isnan(xx)]
    try:
      y = fit(ex,*param)
      yna =  np.isnan(y)
      ex = ex[~yna]
      y = y[ ~yna]
      peak = np.nanargmax(y)

      lo_ex=ex[ex<ex[peak]]
      lo_fit=y[ex<ex[peak]]      
      haf_down=lo_ex[find_nearest(lo_fit,y[peak]/2)]


      hi_ex=ex[ex>ex[peak]]
      hi_fit=y[ex>ex[peak]]      
      haf_up=hi_ex[find_nearest(-hi_fit,-y[peak]/2)]
      
      w = haf_up-haf_down
      xp = ex[peak]

    except:
      print('Couldnt find the FWHM')
      w = np.nan
      xp = np.NaN
      haf_up = np.nan
      haf_down = np.NaN

    if return_locs:
      return haf_up - haf_down,[haf_down,haf_up,xp]
    else:
      return(haf_up-haf_down)


def peak(fit,ex,param = [],peaknness = True):
      from scipy.signal import find_peaks
      y = fit(ex,*param)
      peaks_ind = find_peaks(y)[0]
      m_peak_ind = peaks_ind[np.argmax(y[peaks_ind])]
      return(ex[m_peak_ind])


evals = {
  'fwhm':fwhm,
  'peak':peak,
  'de_e':lambda fit,ex:fwhm(fit,ex)/peak(fi,ex),
}