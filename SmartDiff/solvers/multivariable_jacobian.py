from SmartDiff.solvers.integrator import AutoDiff as AD
import numpy as np
import SmartDiff.solvers.element_op as el

'''
def get_hess(func, vals, ders = None, J = False):
    """
    Parameters
    ----------
    func: n variable, m function
    vals: scalar or list of scalar ([x1,x2,x3,...,xn])
    ders: the derivatives of input variable
    J: Check if the user want Jacobian matrix in their output. The default is False

    Returns
    -------

    """
    if len(func(*vals)) != 1:
        raise ValueError("No hessian for vector-valued function!")
    if J == True:
        values, jaco = get_jaco(func, vals, ders = ders)

    hess = np.zeros([m, m])
    values = np.zeros(n)
'''

# def univariate(func, vals, ders): # YW: also we don't use this function anymore if all input is formatted as list
#     vals = np.array(vals) # YW got rid of the []
#     m = func(vals).shape[0]
#     n = 1
#     ders = np.array(ders) # YW got rid of the []
#     if ders.shape[0] != 1:
#         raise ValueError("Number of rows in ders and variables should be the same")
#     AD_val = AD(vals, ders)
#     if m == 1:
#         AD_func = func(AD_val)
#     if m != 1:
#         AD_func = func(*AD_val)
#     values = AD_func.val
#     jaco = AD_func.der
#     return values, jaco

def multivariable(func, vals, ders):
    n = len(vals)
    m = len(func(*vals))
    if len(ders) != n:
        raise ValueError("Number of rows in ders and number of variables should be the same")
    ders = np.array(ders)
    AD_val = [AD(vals[i], ders[i,:]) for i in range(n)]
    AD_func = func(*AD_val)
    
    values = [AD_func[i].val for i in range(m)]
    jaco = [AD_func[i].der for i in range(m)]
    return np.array(values), np.array(jaco)


def get_jaco(func, vals, ders = None):
    """
    func: m function, n variable
    vals: list of scalar, len = n
    ders: list of derivatives of the input variable(s), n x a number (input dimension of the inner function, default = n)
    """
    # if type(vals) == int or type(vals) == float:
    #     # if no ders given, set it default into 1
    #     if not ders: # YW updated this line to avoid ambiguous comparison of the np array with None
    #         ders = 1
    #     return scalar(func, vals, ders)

    # User gives vals as array(x1,x2,...,xn)
    # if type(vals) == np.ndarray:
    #     # if no ders given, set it default into a n*n identity matrix
    #     if type(ders) != np.ndarray: # YW updated this line to avoid ambiguous comparison of the np array with None
    #         n = vals.shape[0]
    #         ders = np.identity(n)
    #     return multivariable(func, vals, ders)
    # else:
    #     raise AttributeError("Input ders must be a numpy ndarray")

    # if no ders given, set it as a n*n identity matrix by default
    if ders == None:
        n = len(vals)
        ders = np.identity(n)
    return multivariable(func, vals, ders)


output_value1, jacobian1 = get_jaco(func = (lambda x,y,z : [x*z,y*z]), vals = [2,3,4], ders = [[3,4],[3,5],[4,5]])
print('value1', output_value1)
print('jacobian1',jacobian1)

print('----------')
output_value2, jacobian2 = get_jaco(func = (lambda x,y,z : [el.cos(el.exp(x))*z+y*z]), vals = [1,2,3])
print('value2',output_value2)
print('jacobian2',jacobian2)

print('----------')
output_value3, jacobian3 = get_jaco(func = (lambda x: [x**4+10]), vals = [2], ders = [[2]])
print('value3',output_value3)
print('jacobian3',jacobian3)

print('----Now works----')
output_value4, jacobian4 = get_jaco(func = (lambda x: [x**4+10, x**6]), vals = [2])
print('value4',output_value4)
print('jacobian4',jacobian4)

# print('----Trial----')
# output_value5, jacobian5 = get_jaco(func = (lambda x: np.array([x**4+10, x**6])), vals = 2)
# print('value5',output_value5)
# print('jacobian5',jacobian5)

