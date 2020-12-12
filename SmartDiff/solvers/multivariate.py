import numpy as np
from sympy import symbols, ordered, Matrix, hessian
from sympy.core.sympify import sympify

from SmartDiff.globals import FUNC_MAP, MATH_FUNC_MAP
from SmartDiff.solvers.element_op import AutoDiff as AD
from SmartDiff.solvers.element_op import *


def get_ord2_der(func_str, all_vals, dvar_idx, func_map):
    """
    Returns the 2nd order derivative of dvar_idx^th variable
    :param func_str: A string of input math function
    :param all_vals: A list of real scalar values (in the same order as the variable names)
    :param dvar_idx: Index of the variable to take derivative
    :param func_map: A mapping of math expression to python's math function
    :return:
    """
    var_map = {"x%d" % dvar_idx: AutoDiff(value=all_vals[dvar_idx], N=2)}
    var_map.update({"x%d" % idx: val for idx, val in enumerate(all_vals) if idx != dvar_idx})
    var_map.update(func_map)
    AD_out = eval(func_str, var_map)
    return AD_out.der[-1]


def get_hessian(func_str, all_vals, eval_func_map=FUNC_MAP, math_func_map=MATH_FUNC_MAP):
    """
    Returns the hessian matrix of the input function over the input variables
    :param func_str: A string of input math function
    :param all_vals: A list of real scalar values
    :param eval_func_map: A mapping of math expression to python's math function
    :param math_func_map: A mapping of math expression to python's math function
    :return:
    """
    # Assume func is a valid expression
    D = len(all_vals)
    assert D > 0, "There should be at least 1 variable!"

    H = np.zeros((D, D))
    if D == 1:
        H[0][0] = get_ord2_der(func_str, all_vals, 0, eval_func_map)
    else:
        var_map = {"x%d" % i: val for i, val in enumerate(all_vals)}
        var_map.update(math_func_map)
        f = sympify(func_str)
        vs = f.free_symbols
        hess = hessian(f, list(ordered(vs)))
        # print(hess)
        for i in range(D):
            for j in range(D):
                didj_func = hess[i * D + j]
                H[i][j] = eval(str(didj_func), var_map)
    return H


def get_val_jacobian(func_list, vals, eval_func_map=FUNC_MAP):
    '''
    Calulates the function values and Jacobian for multivariate input
    :param func_list: a list of m function strs
    :param vals: list of scalar, len = n
    :param eval_func_map: A mapping of math expression to python's math function
    :return: value, Jacobian
    '''
    n = len(vals)
    m = len(func_list)
    jacobian = np.zeros((m, n))
    values = np.zeros(m)
    xs = symbols('x:%d' % n)
    for i, f in enumerate(func_list):
        var_map = {str(xs[i]): val for i, val in enumerate(vals)}
        var_map.update(eval_func_map)
        get_val = False
        for j, x in enumerate(vals):
            AD_map = var_map.copy()
            AD_map.update({str(xs[j]): AutoDiff(value=vals[j])})
            AD_out = eval(f, AD_map)
            if not get_val:
                try:
                    values[i] = AD_out.val
                    get_val = True
                except AttributeError:
                    pass
            try:
                jacobian[i, j] = AD_out.der[0]
            except AttributeError:
                pass

    return values, jacobian


# if __name__ == "__main__":
#     print("\nTest Hessian (single variable):")
#     func = "x0 + 3"
#     vals = [5]
#     Hmat = get_hessian(func_str=func, all_vals=vals)
#     print("Hessian 1:", Hmat, "; Expected = [[0]]")
#
#     func = "x0**2"
#     vals = [5]
#     Hmat = get_hessian(func_str=func, all_vals=vals)
#     print("Hessian 2:", Hmat, "; Expected = [[2]]")
#
#     func = "x0**3 + log(4, 2)"
#     vals = [5]
#     Hmat = get_hessian(func_str=func, all_vals=vals)
#     print("Hessian 3:", Hmat, "; Expected = [30]]")
#
#     print("\n\nTest Hessian (multivariate):")
#     func = "x0 + x1"
#     vals = [3, 5]
#     Hmat = get_hessian(func_str=func, all_vals=vals)
#     print("Hessian 1:", Hmat)
#     print("Expected = [[0, 0], [0, 0]]\n")
#
#     func = "x0 * x1"
#     vals = [3, 5]
#     Hmat = get_hessian(func_str=func, all_vals=vals)
#     print("Hessian 2:", Hmat)
#     print("Expected = [[0, 1], [1, 0]]\n")
#
#     func = "x0 * x1**3"
#     vals = [3, 5]
#     Hmat = get_hessian(func_str=func, all_vals=vals)
#     print("Hessian 3:", Hmat)
#     print("Expected = [[0, 75], [75, 90]]\n")
#
#     func = "x0 * x1**x0 * x1"
#     vals = [3, 5]
#     Hmat = get_hessian(func_str=func, all_vals=vals)
#     print("Hessian 4:", Hmat)
#     print("Expected = [[0, 375], [375, 900]]\n")
#
#     print("Expected = " + str([[0, np.log(5)+1],[np.log(5)+1, 3/5]]) + "\n")
#
#     func_list = ['x0*x2+x1', 'x1']
#     val_list = [2,3,4]
#
#     values, jacobian = get_val_jacobian(func_list, val_list)
#     print("values:")
#     print(values)
#     print("Jacobian:")
#     print(jacobian)
#
#     func_list = ["x0*exp(x0+9)"]
#     val_list = [2]
#     values, jacobian = get_val_jacobian(func_list, val_list)
#     print("values:")
#     print(values)
#     print("Jacobian:")
#     print(jacobian)
    # output_value1, jacobian1, hess1 = jacob_hessian(lambda x,y,z : [x*z+y, y*z+x, z, z+2], [2,3,4])
    # print('value1', output_value1)
    # print('jacobian1',jacobian1)
    # output_value1, jacobian1, hess1 = jacob_hessian(lambda x,y : [x**2*exp(x)+y, y], [2,3], 1)
    # print('value1', output_value1)
    # print('jacobian1',jacobian1)
    # print('hessian1',hess1)
    #
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