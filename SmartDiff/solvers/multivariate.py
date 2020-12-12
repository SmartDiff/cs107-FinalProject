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
        print(hess)
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