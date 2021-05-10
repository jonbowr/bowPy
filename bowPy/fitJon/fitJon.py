
import numpy as np
from scipy.stats import rv_continuous
from scipy import stats
from scipy.optimize import curve_fit as cf
from . import f_eval as fe
from .funcs import func

def bin_fit(data,weights = None,fittype = 'gauss',bino= 50,
                    norm_data = None,norm_weights = None,
                    cnt_density = False,
                    use_err = True,log_bins = False,normx = 1,norm_max = False,
                    fit_input= {}):

    def gauss(x, a=1, x0=1, sigma=.5):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def log_gauss(x,a=1,sigma=1,mu=0):
        return(a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))

    def tanh(xl,x0 =1,sigma =10):
        # x = (xl/np.nanmax(xl)- x0/np.nanmax(xl))*2*sigma
        x = (xl- x0)*2*sigma/np.nanmax(xl)
        y = (np.tanh(x)+1)/2
        return(y)

    def log_gauss_flip(xi,a=1,sigma=1,mu=0,x0 = 2):
        x = abs(1-xi/x0)
        y = a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2))*np.heaviside(x0-xi,.5)
        # y[xi>x0]= 0
        return(y)    
    
    def tanh_trunk_gauss(xi,a=1,sigma=1,mu=0,x0 = 2,x02= .5,zig = 10):
        return(log_gauss_flip(xi,a,sigma,mu,x0)*tanh(xi,x02,zig))

    def skew_gauss(x, a=1, loc=.25, scale = .3):
        return(stats.skewnorm.pdf(x,a, loc, scale))

    def skew_trunk_gauss(xi,a=1,loc = .25,scale= .3,x02= .5,zig = 10):
        return(skew_gauss(xi,a,loc,scale)*tanh(xi,x02,zig))

    def exp_gauss(x,k=1,loc=.25,scale=.3):
        return(stats.exponnorm.pdf(x,k,loc,scale))

    def kappa(x,h = .1,k = 0,l = 0,scale =1,hh = 1,b=0):
        # h = 0
        v = stats.kappa4.pdf(x,h,k,l,scale)
        return(hh*v)

    def kappa3(x,h = .1,k = 0,l = 0,scale =1,hh = 1):
        v = stats.kappa4.pdf(x,h,k,l,scale)
        return(hh*v)

    e_funcs = {
              'gauss': gauss,
              'log_gauss':log_gauss,
              'tanh':tanh,
              'log_gauss_flip': log_gauss_flip,
              'log_trunk_gauss':tanh_trunk_gauss,
              'skew_gauss':skew_gauss,
              'skew_trunk_gauss':skew_trunk_gauss,
              'kappa':kappa
              # 'exp_gauss':exp_gauss
              }


    # rng = max(data)
    mean = np.average(data,weights = weights)
    std = np.std(data)
    mino = np.min(data)-std
    maxo = np.max(data)+std

    y,bin_edges=np.histogram(data,bins=(np.geomspace(mino,maxo,bino) if log_bins else np.linspace(mino,maxo,bino)),
                                                density=False,weights = weights)

    # if type(norm_data)==np.array:
    if norm_data:
      y = np.nan_to_num(y/np.histogram(norm_data,bins=bin_edges,
                                                  density=False,weights = norm_weights)[0])
    elif cnt_density:
      y = y/(np.diff(bin_edges)/normx)
    
    if norm_max:
      y = y/np.max(y)

    x=(bin_edges[1:]+bin_edges[:-1])/2
    if use_err:
      cnts = np.histogram(data,bins=bin_edges)[0]
      err = 1/np.sqrt(cnts)
      y_err = y*err
    else:
      err = None
      y_err = None
    
    if fittype is not 'spline':

        from scipy.stats import lognorm,skewnorm
        e_guess = {
              'gauss': [np.max(y),mean,std],
              'log_gauss':lognorm.fit(data),
              'tanh':None,
              'log_gauss_flip': [np.max(y),mean,mino,maxo],
              'log_trunk_gauss':[np.max(y),mean,mino,maxo,mean-2*std,std],
              'skew_gauss':skewnorm.fit(data),
              'skew_trunk_gauss':list(skewnorm.fit(data))+[mean-2*std,std],
              'kappa':[.1,-.1,mean,1,np.max(y),0]
              # 'exp_gauss':exponnorm.fit(data)
              }  

        try:
          parms=cf(e_funcs[fittype],x,y,e_guess[fittype],sigma = err,**fit_input)
          
          # def out_func(x):
          #     return(e_funcs[fittype](x,*parms[0]))
          # return(lambda x: e_funcs[fittype](x,*parms[0]),[x,y,y_err])
          f = func(e_funcs[fittype],parms[0],parms[1])
          f.hist = {'x':x,'y':y,'err':y_err}
          # return(f)
          # return(func(e_funcs[fittype],parms[0],parms[1]),[x,y,y_err])
        except RuntimeError:
          # print('bad_stuffs')
          # return(None,[x,y,y*err])
          f = func(e_funcs[fittype],np.ones(len(e_guess[fittype]))*np.nan,np.ones([len(e_guess[fittype])]*2)*np.nan)
          f.hist = {'x':x,'y':y,'err':y_err}
        return(f)
          # return(func(e_funcs[fittype],np.ones(len(e_guess[fittype]))*np.nan,np.ones([len(e_guess[fittype])]*2)*np.nan),[x,y,y_err])
    else:
        from scipy.interpolate import interp1d as up
        from scipy.ndimage import gaussian_filter1d as gf
        return(func(up(x,gf(gf(y,1),1),kind = 'cubic'),[]),[x,y,y_err])

# class func:

#     def __init__(self,func,params,cov = None):
#         self.f = func
#         self.params = params
#         self.cov = cov

#     def __call__(self,x):
#         if type(self.params) == dict:
#             return(self.f(x,**self.params))
#         else:
#             return(self.f(x,*self.params))
            


from scipy.stats import rv_continuous
class rv_func:


    def __init__(self,func,
                 data,
                 p0 = None,
                 bino = None,
                 weights = None):
        
        def gauss(x, a=1, x0=1, sigma=.5):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        def log_gauss(x,a=1,sigma=1,mu=0):
            return(a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))

        def tanh(xl,x0 =1,sigma =10):
            # x = (xl/np.nanmax(xl)- x0/np.nanmax(xl))*2*sigma
            x = (xl- x0)*2*sigma/np.nanmax(xl)
            y = (np.tanh(x)+1)/2
            return(y)

        def log_gauss_flip(xi,a=1,sigma=1,mu=0,x0 = 2):
            x = abs(1-xi/x0)
            y = a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2))*np.heaviside(x0-xi,.5)
            # y[xi>x0]= 0
            return(y)    
        
        def tanh_trunk_gauss(xi,a=1,sigma=1,mu=0,x0 = 2,x02= .5,zig = 10):
            return(log_gauss_flip(xi,a,sigma,mu,x0)*tanh(xi,x02,zig))

        def skew_gauss(x, a=1, loc=.25, scale = .3):
            return(stats.skewnorm.pdf(x,a, loc, scale))

        def skew_trunk_gauss(xi,a=1,loc = .25,scale= .3,x02= .5,zig = 10):
            return(skew_gauss(xi,a,loc,scale)*tanh(xi,x02,zig))

        def exp_gauss(x,k=1,loc=.25,scale=.3):
            return(stats.exponnorm.pdf(x,k,loc,scale))

        def poisson(x1,b=1,c=0,k=1):
            a=1
            x = (x1-c)/b
            norm = k**k*np.exp(-k)/np.math.factorial(k)
            return(a*(x*k**2)**k*np.exp(-x*k**2)/np.math.factorial(k)/norm)

        def kappa(x,h = .1,k = 0,l = 0,scale =1,hh = 1):
            return(hh*stats.kappa4.pdf(x,h,k,l,scale))

        def kappa3(x,h = .1,k = 0,l = 0,scale =1,hh = 1):
            v = stats.kappa3.pdf(x,h,k,l,scale)
            return(hh*v)

        e_funcs = {
              'gauss': gauss,
              'log_gauss':log_gauss,
              'tanh':tanh,
              'log_gauss_flip': log_gauss_flip,
              'log_trunk_gauss':tanh_trunk_gauss,
              'skew_gauss':skew_gauss,
              'skew_trunk_gauss':skew_trunk_gauss,
              'kappa':kappa,
              'kappa3':kappa3
              # 'exp_gauss':exp_gauss
              }
        self.data = data
        if func in e_funcs:
            self.pdf = rv_continuous()
            self.pdf._pdf = e_funcs[func]
        else: 
            self.pdf = func
        self.p0 = (p0 if p0 else self.pdf.fit(data))


            # rng = max(data)
        if bino:
            mean = np.average(data,weights = weights)
            std = np.std(data)

            mino = np.min(data)-std
            maxo = np.max(data)+std
            y,bin_edges=np.histogram(data,bins=np.linspace(mino,maxo,bino),
                                                        density=False,weights = weights)

            x=(bin_edges[1:]+bin_edges[:-1])/2
            cnts = np.histogram(data,bins=np.linspace(min(data),max(data),bino))[0]
            err = 1/np.sqrt(cnts)
            self.h_data = {'x':x,
                            'y':y,
                            'y_err':err*y}
        # self.pdf.fit(data)
        # self.f = e_funcs[func_nam]

    #     from scipy.stats import lognorm,skewnorm
    #     e_guess = {
    #       'gauss': [np.max(y),mean,std],
    #       'log_gauss':lognorm.fit(data),
    #       'tanh':None,
    #       'log_gauss_flip': [np.max(y),mean,mino,maxo],
    #       'log_trunk_gauss':[np.max(y),mean,mino,maxo,mean*.1,mean*.1],
    #       'skew_gauss':skewnorm.fit(data),
    #       'skew_trunk_gauss':list(skewnorm.fit(data))+[mean*.1]*2
    #       # 'exp_gauss':exponnorm.fit(data)
    #       }  
    #     self.f = func
    #     self.params = params
    #     self.cov = cov

    def __call__(self,x):
        return(self.pdf._pdf(x,*self.p0))


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx-1
    else:
        return idx

def fwhm(ex,fit,param=[]):
    try:
      y = fit(ex,*param)
      peak = np.argmax(y)

      lo_ex=ex[ex<ex[peak]]
      lo_fit=y[ex<ex[peak]]      
      haf_down=lo_ex[find_nearest(lo_fit,y[peak]/2)]


      hi_ex=ex[ex>ex[peak]]
      hi_fit=y[ex>ex[peak]]      
      haf_up=hi_ex[find_nearest(-hi_fit,-y[peak]/2)]

      return haf_up - haf_down,[haf_down,haf_up,ex[peak]]
    except:
      print('Couldnt find the FWHM')
      return(np.nan,[np.nan]*3)