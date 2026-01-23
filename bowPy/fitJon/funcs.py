import numpy as np
from scipy.stats import skewnorm,kappa4,kappa3,lognorm
from ..numJon.numJon import std_w

def fx(x):
    return(x)

def gauss(x, a=1, x0=1, sigma=.5):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def log_gauss(x,a=1,sigma=1,mu=0):
    return(a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))

def tanh(xl,a = 1,x0 =1,sigma =10):
    # x = (xl/np.nanmax(xl)- x0/np.nanmax(xl))*2*sigma
    x = (xl- x0)*2*sigma#/np.nanmax(xl)
    y = a*(np.tanh(x)+1)/2
    return(y)

def log_gauss_flip(xi,a=1,sigma=1,mu=0,x0 = 2):
    x = abs(1-xi/x0)
    y = a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2))*np.heaviside(x0-xi,.5)
    # y[xi>x0]= 0
    return(y)    

def tanh_trunk_gauss(xi,a=1,sigma=1,mu=0,x0 = 2,x02= .5,zig = 10):
    return(log_gauss_flip(xi,a,sigma,mu,x0)*tanh(xi,x02,zig))

def skew_gauss(x, a= 1,s=1, loc=.25, scale = .3):
    return(a*skewnorm._pdf(x,s, loc, scale))

def skew_trunk_gauss(xi,a = 1,s=1,loc = .25,scale= .3,x02= .5,zig = 10):
    return(a*skewnorm._pdf(xi,s,loc,scale)*tanh(xi,x02,zig))

# def exp_gauss(x,a = 1,k=1,loc=.25,scale=.3):
#     return(a*exponnorm._pdf(x,k,loc,scale))

def poisson(x1,a = 1,b=1,c=0,k=1):
    # a=1
    x = (x1-c)/b
    # norm = k**k*np.exp(-k)/np.math.factorial(k)
    return(a*(x*k**2)**k*np.exp(-x*k**2)/np.math.factorial(k))

def kappa_4(x,h = .1,k = 0,l = 0,scale =1,a = 1,y0 = 0):
    return(a*kappa4.pdf((x-l)/scale,h,k)+y0)

def kappa_3(x,h = .1,k = 0,l = 0,scale =1,a = 1):
    v = kappa3._pdf(x,h,k,l,scale)
    return(a*v)

def nat_log(x,a = 1,b = 0,x0 = 0):
    return(a * np.log(x+x0) + b)

def power_law(x,a = 1,k = -1,y0 = 0,x0 = 0):
    return(a*(x-x0)**k+y0)

def linear(x,m=1,b=1):
    return(m*x+b)

def gauss_flip_flop(x,a=1,x0=0,sigma1=1,sigma2=1):
        return(a*np.exp(-(x-x0)**2/sigma1**2/2)*np.heaviside(x0-x,.5)+
                 a*np.exp(-(x0-x)**2/sigma2**2/2)*np.heaviside(x-x0,.5))

def gauss_asym(E,a=1,Ec=0,En=1,Ep=1):
    delta1 = 2*(1-En/Ec)
    delta2 = 2*(1-Ec/Ep)
    return(a*np.exp(-4*np.log(2)*(E/Ec-1)**2/delta1**2)*np.heaviside(Ec-E,.5)+
                 a*np.exp(-4*np.log(2)*(Ec/E-1)**2/delta2**2)*np.heaviside(E-Ec,.5))

# funcs = {
#           'gauss':gauss,
#           'log_gauss':log_gauss,
#           'tanh':tanh,
#           'log_gauss_flip': log_gauss_flip,
#           'log_trunk_gauss':tanh_trunk_gauss,
#           'skew_gauss':skew_gauss,
#           'skew_trunk_gauss':skew_trunk_gauss,
#           'kappa4':kappa_4,
#           'kappa3':kappa_3,
#           'power_law':power_law,
#          }


funcs = {
            'fx':
               {
                    'f':fx,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'gauss':
                {
                    'f':gauss,
                    'name':'Gaussian Distribution',
                    'latex':r'$a*e^{\dfrac{-(x-x0)^2}{2*\sigma^2}',
                    'params':['a','x0','sigma'],
                    'reference':'',
                },    
           'log_gauss': 
               {
                    'f':log_gauss,
                    'name':'',
                    'latex':r'a*(x*\sigma*\sqrt{2*\pi})^{-1}\exp{-(\log(x) - \mu)^2/(2\sigma^2)}',
                    'params':[],
                    'reference':'',
                },
            'tanh': 
               {
                    'f':tanh,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'log_gauss_flip': 
               {
                    'f':log_gauss_flip,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'log_trunk_gauss': 
               {
                    'f':tanh_trunk_gauss,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'skew_gauss': 
               {
                    'f':skew_gauss,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'skew_trunk_gauss':
               {
                    'f':skew_trunk_gauss,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'kappa4':
               {
                    'f':kappa_4,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'kappa3':
               {
                    'f':kappa_3,
                    'name':'',
                    'latex':r'',
                    'params':[],
                    'reference':'',
                },
            'power_law':
               {
                    'f':power_law,
                    'name':'Power Law',
                    'latex':r'$y = a*(x-x_0)^k+y_0$',
                    'params':['a','k','y0','x0'],
                    'reference':'',
                },
            'nat_log':
               {
                    'f':nat_log,
                    'name':'Natural Log',
                    'latex':r'',
                    'params':['a','b','x0'],
                    'reference':'',
                },
            'linear':
               {
                    'f':linear,
                    'name':'Linear',
                    'latex':r'$y = m*x+b$',
                    'params':['m','b'],
                    'reference':'',
                },
            'gauss_flip_flop':
               {
                    'f':gauss_flip_flop,
                    'name':'Assymetric Gaussian',
                    'latex':r'$$',
                    'params':['a','x0','sigma1','sigma2'],
                    'reference':'',
                },
            'gauss_asym':
               {
                    'f':gauss_asym,
                    'name':'Assymetric Gaussian',
                    'latex':r'$$',
                    'params':['a','Ec','En','Ep'],
                    'reference':'',
                },
        }

# p0 = {
#       'gauss': lambda x: [np.max(y),mean,std],
#       'log_gauss':lognorm.fit(data),
#       'tanh':None,
#       'log_gauss_flip': [np.max(y),mean,mino,maxo],
#       'log_trunk_gauss':[np.max(y),mean,mino,maxo,mean-2*std,std],
#       'skew_gauss':skewnorm.fit(data),
#       'skew_trunk_gauss':list(skewnorm.fit(data))+[mean-2*std,std],
#       'kappa':[.1,-.1,mean,1,np.max(y),0]
#       # 'exp_gauss':exponnorm.fit(data)
#       }  


# integrate with above db
p0_xy = {
        'fx':lambda x,y: [],
      'gauss': lambda x,y: [np.max(y),np.average(x,weights=y),np.average(x,weights=y)],
      'log_gauss':lambda x,y: [np.max(y),np.average(x,weights=y),np.average(x,weights=y)],
      'tanh':None,
      'log_gauss_flip': lambda x,y: [np.max(y),np.average(x,weights=y),np.nanmin(x),np.nanmax(x)],
      'log_trunk_gauss': lambda x,y: [np.max(y),
                            np.average(x,weights=y),
                            np.nanmin(x),np.nanmax(x),
                            np.average(x,weights=y)-2*std_w(x,y),
                                std_w(x,y)],
      'skew_gauss':lambda x,y: None,
      'skew_trunk_gauss':lambda x,y: None,
      'kappa4':lambda x,y: [.1,-.1,np.average(x,weights=y),1,np.max(y),0],
      'kappa3':lambda x,y: [-.1,np.average(x,weights=y),1,np.max(y),0],
      'power_law':lambda x,y: [1]*4,
      'nat_log':lambda x,y: [1,1,0],
      'linear':lambda x,y: [1,0],
      'gauss_flip_flop': lambda x,y: [np.max(y),np.average(x,weights=y),np.average(x,weights=y),np.average(x,weights=y)],
      'gauss_asym': lambda x,y: [np.max(y),np.average(x,weights=y),np.average(x,weights=y),np.average(x,weights=y)],
      }


class func:

    def __init__(self,func='fx',params=[],cov = None):
        # self.f = func
        self.p = params
        self.p0 = params
        self.cov = cov
        if type(func) == str: 
            self.info = funcs[func]
            self.f = funcs[func]['f']
            self.f_name = func
            # self.p0 = 
        else:
            self.f = func
            self.info = funcs['fx']
            self.f_name = None


    def __call__(self,x,p0 = None):
        if p0 == None: 
            if type(self.p) == dict:
                return(self.f(x,**self.p))
            else:
                return(self.f(x,*self.p))
        else: 
            return(self.f(x,*p0))

    def f_txt(self):
        return(self.info['latex'])

    def p_txt(self,style = '%2.2e'):
        return('\n'.join([('%s='+style)%(n,p) for n,p in zip(self.info['params'],self.p)]))


    def pretty_txt(self,p_style = '%2.2e'):
        return('%s\n%s'%(self.f_txt(),self.p_txt(p_style)))


    def guess_pi(self,x,y):
        return(p0_xy[self.f_name](x,y))


