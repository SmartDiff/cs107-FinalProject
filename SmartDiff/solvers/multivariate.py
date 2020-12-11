import numpy as np
from sympy import symbols
from SmartDiff.solvers.element_op import AutoDiff as AD
from SmartDiff.solvers.element_op import *


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

def jacob_hessian(func, vals, N=1):
    '''
    :param func: m function, n variable
    :param vals: list of scalar, len = n
    :param N: specify order of derivatives, default 1, max 2
    :return:
    '''
    n = len(vals)
    m = len(func(*vals))
    if N == 2:
        if m > 1:
            print("Warning: Only scalar valued function is supported for Hessian. Only calculating Jacobian now")
            N = 1
    elif N > 2:
        raise ValueError("Only support 1st and 2nd order derivatives for multivariable cases")
    xs = symbols('x:%d' % n)
    print(xs)
    AD_val = [AD(vals[i], N=N, var=str(xs[i])) for i in range(n)]
    AD_func = func(*AD_val)
    values = np.array([AD_func[i].val for i in range(m)])
    jacobian = np.array([AD_func[i].der[0] for i in range(m)])
    hessian = np.array([AD_func[i].der[1] for i in range(m)]) if N == 2 else np.empty((n, n))
    return values, jacobian, hessian


# output_value1, jacobian1 = jacob_hessian(func = (lambda x,y,z : [x*z+y,y*z+x]), vals = [2,3,4])
# print('value1', output_value1)
# print('jacobian1',jacobian1)
output_value1, jacobian1, hess1 = jacob_hessian(lambda x,y : [x**2*exp(x)+y, y], [2,3], 2)
print('value1', output_value1)
print('jacobian1',jacobian1)
print('hessian1',hess1)

# output_value1, jacobian1, hess1 = jacob_hessian(lambda x,y : [x**2*exp(x),x*y], [2,3], 2)
# print('value2', output_value1)
# print('jacobian2',jacobian1)
# print('hessian2',hess1)

# print('----------')
# output_value2, jacobian2 = get_jaco(func = (lambda x,y,z : [el.cos(el.exp(x))*z+y*z]), vals = [1,2,3])
# print('value2',output_value2)
# print('jacobian2',jacobian2)

# print('----------')
# output_value3, jacobian3 = get_jaco(func = (lambda x: [x**4+10]), vals = [2], ders = [[2]])
# print('value3',output_value3)
# print('jacobian3',jacobian3)

# print('----Now works----')
# output_value4, jacobian4 = get_jaco(func = (lambda x: [x**4+10, x**6]), vals = [2])
# print('value4',output_value4)
# print('jacobian4',jacobian4)

# print('----Trial----')
# output_value5, jacobian5 = get_jaco(func = (lambda x: np.array([x**4+10, x**6])), vals = 2)
# print('value5',output_value5)
# print('jacobian5',jacobian5)

