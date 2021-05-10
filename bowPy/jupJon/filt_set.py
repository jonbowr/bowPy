import numpy as np

FILTERZ = {
           'Equal To':lambda x,y: np.equal(x,y),
           'Not Equal':lambda x,y:np.not_equal(x,y),
           'Greater Than': lambda x,y:np.greater_equal(x,y),
            'Less Than':lambda x,y:np.less_equal(x,y),
           'Contains (str)':lambda x,y: x.str.contains(y,na = False)
          }