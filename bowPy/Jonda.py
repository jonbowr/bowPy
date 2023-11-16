import numpy as np
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
from .numJon.numJon import gauss_filt_nan as gf
from .fitJon import funcs as fc

class Jonda:

    def __init__(self,
                     data = None,
                     xy_data = None,
                     err = None,
                     weights = None,
                     func = None,
                     p0 = None,
                     bins = None,
                     covs = None):

        self.data = data
        self.xy = xy_data
        self.err = err
        self.weights = weights
        self.bins = bins
        self.p0 = p0
        self.covs = covs
        self.f = None
        self.cnt = None


        if type(func) == str: 
            self.func = fc.funcs[func]['f']
            self.p_i = fc.p0_xy[func]
            self.p0_xy = fc.p0_xy[func]
        else:
            self.func = func

    def __call__(self,x,p0 = None):
        if p0 == None: 
            return(self.f(x,*self.p0))
        else: 
            return(self.f(x,*p0))
    def __str__(self):
        return('bowPy.Jonda:\n\tfit:%s,p0:%s'%(str(self.f),str(self.p0)))

    def __repr__(self):
        return(str(self))

    def bin_data(self,bins= None,inplace = True,
                            params = {},
                                norm_binwidth = True,
                                max_norm = True,
                                    cnt_err = True):
        if bins != None:
            self.bins = bins
        h,xb = np.histogram(self.data,bins = self.bins,
                            weights = self.weights,**params)
        
        if norm_binwidth:
            h = h/np.diff(xb)
        if max_norm:
            h = h/np.nanmax(h)

        if cnt_err:
            cnt,xb = np.histogram(self.data,bins = self.bins,density = False)
            with np.errstate(divide='ignore'):
                err = 1/np.sqrt(cnt)
            if inplace:
                self.cnt = cnt
                self.err = h*err

        xy = np.stack([xb[:-1]+np.diff(xb)/2,h])
        if inplace:
            self.xy = xy
            self.bins = xb
            return(self)
        else:
            return(xy,err,xb)


    def fit_xy(self,p_i = None,use_err = True,args = {},fy = lambda x: x):
        if p_i == None:
            # p_i = self.p_i(*self.xy)
            try:
                params,covs = cf(self.func,self.xy[0],fy(self.xy[1]),**args)
                self.p0 = params
                self.covs = covs
                self.f = self.func
            except:
                print('Curve Fit Failed')
                params = [np.nan]*len(self.p0)
                covs = None
                # self.f = lambda x: np.nan
                self.f = self.func
        if use_err:
            nano = ~np.sum(np.isnan(np.concatenate([self.xy,self.err.reshape(1,-1)])),axis = 0).flatten().astype(bool)
            params,covs = cf(self.func,*self.xy[:,nano],p0 = p_i,sigma = self.err[nano],**args)
            self.p0 = params
            self.covs = covs
            self.f = self.func

    def interp_xy(self, kind = 'linear',sigma = 1):
        self.f = interp1d(self.xy[0,:],
                          gf(self.xy[1,:],sigma = sigma),
                          kind = kind,bounds_error = False)
        self.p0 = []
        return(self)

    def find_xy(self, find = 'fwhm',ex = None):
        from .fitJon.f_eval import evals
        if ex == None: 
            ex = np.linspace(np.nanmin(self.xy[0,:]),np.nanmax(self.xy[0,:]),len(self.xy[0,:])*100)
        try:
            return(evals[find](self.f,ex,self.p0))
        except:
            print('eval failed')
            return(np.nan)
    def get_fxy(self,ex = None):
        if ex == None: 
            ex = np.linspace(np.nanmin(self.xy[0,:]),np.nanmax(self.xy[0,:]),len(self.xy[0,:])*100)
        return(ex,self(ex))        

    def show(self,fig = None,ax = None,label = None):
        from matplotlib import pyplot as plt
        if ax == None:
            fig,ax = plt.subplots()
        lin = ax.plot(*self.xy,label = label)[0]
        if self.f != None:
            ax.plot(*self.get_fxy(),color = lin.get_color())

