import SmartDiff.solvers.element_op as el
from .SmartDiff.solvers.element_op import AutoDiff as AD
from .SmartDiff.solvers.multivariate import *
import numpy as np
import pytest

class TestElemOpNOrder:

    def test_add(self):
        # N > 1
        x = AD(5, N=3)
        f = el.sin(x)
        g = 10
        h = f + g
        assert (h.val, np.round(h.der[-1]), 6) == (np.sin(5) + 10, np.round(-np.cos(5)), 6)

        f = 10
        g = el.sin(x)
        h = f + g
        assert (h.val, np.round(h.der[-1]), 6) == (np.sin(5) + 10, np.round(-np.cos(5)), 6)

        f = el.sin(x)
        g = x ** 2
        h = f + g
        assert (h.val, np.round(h.der[-1]), 6) == (np.sin(5) + 25, np.round(-np.cos(5)), 6)

    def test_sub(self):
        # N > 1
        x = AD(5, N=3)
        f = el.sin(x)
        g = 10
        h = f - g
        assert (h.val, np.round(h.der[-1]), 6) == (np.sin(5) - 10, np.round(-np.cos(5)), 6)

        f = el.sin(x)
        g = x ** 2
        h = f - g
        assert (h.val, np.round(h.der[-1]), 6) == (np.sin(5) - 25, np.round(-np.cos(5)), 6)

    def test_sin(self):
        # N = 1
        x = AD(0)
        f = el.sin(x)
        assert (f.val, f.der) == (0.0, 1.0)

        x = 13
        f = el.sin(x)
        assert (f.val, f.der) == (np.sin(13), 0)

        x = 13.0
        f = el.sin(x)
        assert (f.val, f.der) == (np.sin(13), 0)

        # N > 1
        x = AD(2, N=4)
        f = el.sin(x)
        # print(f)
        assert (f.val, np.round(f.der[-1], 6)) == (np.sin(2), np.round(np.sin(2), 6))

        x = AD(2, N=3)
        f = el.sin(x)
        # print(f)
        assert (f.val, np.round(f.der[-1], 6)) == (np.sin(2), np.round(-np.cos(2), 6))

        f = el.sin(x) + 3
        assert (f.val, np.round(f.der[-1], 6)) == (np.sin(2) + 3, np.round(-np.cos(2), 6))

        f = el.sin(x) - 2
        assert (f.val, np.round(f.der[-1], 6)) == (np.sin(2) - 2, np.round(-np.cos(2), 6))

        f = el.sin(x) * 4
        assert (f.val, np.round(f.der[-1], 6)) == (4 * np.sin(2), np.round(-4 * np.cos(2), 6))

    def test_cos(self):
        # N > 1
        x = AD(10, N=5)
        f = el.cos(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.cos(10), np.round(-np.sin(10), 6))

        x = AD(2, N=2)
        f = el.cos(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.cos(2), np.round(-np.cos(2), 6))

        f = el.cos(x) + 3
        assert (f.val, np.round(f.der[-1], 6)) == (np.cos(2) + 3, np.round(-np.cos(2), 6))

        f = el.cos(x) - 2
        assert (f.val, np.round(f.der[-1], 6)) == (np.cos(2) - 2, np.round(-np.cos(2), 6))

        f = el.cos(x) * 4
        assert (f.val, np.round(f.der[-1], 6)) == (4 * np.cos(2), np.round(-4 * np.cos(2), 6))

    def test_tan(self):
        # N > 1
        x = AD(10, N=2)
        f = el.tan(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.tan(10), np.round(2 * np.tan(10) * (1 / (np.cos(10) ** 2)), 6))

        x = AD(10, N=3)
        f = el.tan(x)
        assert (f.val, np.round(f.der[-1], 6)) == (
        np.tan(10), np.round(-2 * (-2 + np.cos(20)) * (1 / (np.cos(10) ** 4)), 6))

    def test_arcsin(self):
        # N > 1
        x = AD(0.5, N=2)
        f = el.arcsin(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.arcsin(0.5), 0.7698)

    def test_arccos(self):
        # N > 1
        x = AD(0.5, N=2)
        f = el.arccos(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.arccos(0.5), -0.7698)

    def test_arctan(self):
        # N > 1
        x = AD(0.5, N=2)
        f = el.arctan(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.arctan(0.5), -0.64)

    def test_sinh(self):
        # N > 1
        x = AD(10, N=4)
        f = el.sinh(x)
        assert (f.val, f.der[-1]) == (np.sinh(10), np.sinh(10))

        x = AD(10, N=7)
        f = el.sinh(x)
        assert (f.val, f.der[-1]) == (np.sinh(10), np.cosh(10))

    def test_cosh(self):
        # N > 1
        x = AD(10, N=4)
        f = el.cosh(x)
        assert (f.val, f.der[-1]) == (np.cosh(10), np.cosh(10))

        x = AD(10, N=7)
        f = el.cosh(x)
        assert (f.val, f.der[-1]) == (np.cosh(10), np.sinh(10))

    def test_tanh(self):
        # N > 1
        x = AD(0.5, N=2)
        f = el.tanh(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.tanh(0.5), -0.726862)

        x = AD(0.5, N=3)
        f = el.tanh(x)
        assert (f.val, np.round(f.der[-1], 6)) == (np.tanh(0.5), -0.565209)

        x = AD(0, N=3)
        f = el.tanh(x)
        assert (f.val, f.der[-1]) == (np.tanh(0), -2)

    def test_exp(self):
        # N = 1
        f = el.exp(1)
        assert (f.val, f.der) == (np.exp(1), 0)

        x = AD(1)
        f = el.exp(x)
        assert (f.val, f.der) == (np.exp(1), np.exp(1))

        x = AD(2)
        f = el.power(x, 3)
        g = el.exp(f)
        print(g)
        assert (round(g.val, 6), round(g.der[0], 6)) == (round(np.exp(1) ** 8, 6), round(12 * np.exp(1) ** 8, 6))

        with pytest.raises(AttributeError):
            x = "hello"
            f = el.exp(x)

        # N > 1
        for N in range(2, 10):
            x = AD(3, N=N)
            f = el.exp(x)
            print("N=%d" % N, f)

        x = AD(2, N=5)
        f = el.exp(x)
        assert (f.val, f.der[-1]) == (np.exp(2), np.exp(2))

    def test_ln(self):
        # N = 1
        f = el.ln(np.e)
        assert (f.val, f.der) == (1, 0)

        x = AD(np.e)
        f = el.ln(x)
        assert (f.val, f.der) == (1, 1 / np.e)

        # N > 1
        # See explicit formula here: https://www.math24.net/higher-order-derivatives/#example2
        x = AD(np.e, N=2)
        f = el.ln(x)
        assert (f.val, f.der[-1]) == (1, -1 / (np.e ** 2))

        x = AD(np.e ** 2, N=5)
        f = el.ln(x)
        assert (f.val, f.der[-1]) == (2, 24 / ((np.e ** 2) ** 5))

        # composite x
        x = AD(np.e, N=4)
        f = el.ln(x * 3 + 1)
        assert (f.val, f.der[-1]) == (np.log(np.e * 3 + 1), (-6 * 3 ** 4) / ((np.e * 3 + 1) ** 4))

    def test_inv(self):
        # N = 1
        f = el.ln(np.e)
        assert (f.val, f.der) == (1, 0)

        x = AD(np.e)
        f = el.ln(x)
        assert (f.val, f.der) == (1, 1 / np.e)

        # N > 1
        # See explicit formula here: https://www.math24.net/higher-order-derivatives/#example2
        x = AD(np.e, N=2)
        f = el.ln(x)
        assert (f.val, f.der[-1]) == (1, -1 / (np.e ** 2))

        x = AD(np.e ** 2, N=5)
        f = el.ln(x)
        assert (f.val, f.der[-1]) == (2, 24 / ((np.e ** 2) ** 5))

    def test_truediv(self):
        # N = 1 tests all passed in solvers_test
        x = AD(3)
        f = 5 * x / 3
        assert (f.val, f.der[-1]) == (5, 5 / 3)

        x = AD(4, N=2)
        f = x / x
        assert (f.val, f.der[-1]) == (1, 0)

    def test_rtruediv(self):
        x = AD(4, N=2)
        f = 3 / (2 * x)
        assert (f.val, f.der[-1]) == (3 / 8, 3 / 64)

        x = AD(4, N=3)
        f = 3 / (2 * x)  # shows rtruediv should work
        assert (f.val, f.der[-1]) == (3 / 8, -9 / 4 ** 4)

    def test_sqrt(self):
        x = AD(4, N=2)
        f = el.sqrt(x)
        assert (f.val, f.der[-1]) == (2, -0.25 / (4 ** 1.5))

    def test_power(self):
        x = AD(4, N=3)
        f = el.power(x, 4)
        assert (f.val, f.der[-1]) == (256, 96)

    def test_AD_pow(self):
        pass

    def test_power_k(self):
        with pytest.raises(AttributeError):
            x = AD(10)
            n = "1"
            f = el.power_k_order(x, n, 3)

        with pytest.raises(AttributeError):
            x = AD(10)
            n = AD(3)
            f = el.power_k_order(x, n, 3)

        with pytest.raises(ValueError):
            x = AD(-3)
            n = 1 / 2
            f = el.power_k_order(x, n, 3)

    def test_logistic(self):
        x = AD(100)
        f = el.logistic(x, x0=0, L=1, k=1)
        assert (f.val, np.round(f.der[-1], 10)) == (1 / (1 + np.exp(-100)),
                                                    np.round(np.exp(100) / (1 + np.exp(100)) ** 2, 6))
        # assert (f.val, f.der[-1]) == (1/(1+np.exp(-100)), np.exp(100)/(1 + np.exp(100))**2)

        f = el.logistic(x, x0=10, L=2, k=3)
        assert (f.val, f.der[-1]) == (2 * np.exp(270) / (1 + np.exp(270)), (6 * np.exp(270)) / (1 + np.exp(270)) ** 2)

        x = AD(100, N=2)
        f = el.logistic(x, x0=0, L=1, k=1)
        assert (f.val, np.round(f.der[-1], 6)) == (
        1 / (1 + np.exp(-100)), np.round((np.exp(100) - np.exp(200)) / (1 + np.exp(100)) ** 3, 6))

        x = 100
        f = el.logistic(x, x0=0, L=1, k=1)
        assert (f.val, f.der) == (1 / (1 + np.exp(-100)), 0)

        with pytest.raises(AttributeError):
            x = "AD(-3)"
            f = el.logistic(x, x0=0, L=1, k=1)


    def test_hessian(self):

        # single variable
        func = "x0 + 3"
        vals = [5]
        Hmat = get_hessian(func_str=func, all_vals=vals)
        assert (Hmat == [[0]])
        #print("Hessian 1:", Hmat, "; Expected = [[0]]")

        func = "x0**2"
        vals = [5]
        Hmat = get_hessian(func_str=func, all_vals=vals)
        assert (Hmat == [[2]])
        #print("Hessian 2:", Hmat, "; Expected = [[2]]")

        func = "x0**3 + log(4, 2)"
        vals = [5]
        Hmat = get_hessian(func_str=func, all_vals=vals)
        assert (Hmat == [[30]])
        #print("Hessian 3:", Hmat, "; Expected = [[30]]")

        # multivariable

        func = "x0 + x1"
        vals = [3, 5]
        Hmat = get_hessian(func_str=func, all_vals=vals)
        assert (Hmat[0] == [0, 0]).all()
        assert (Hmat[1] == [0, 0]).all()

        func = "x0 * x1"
        vals = [3, 5]
        Hmat = get_hessian(func_str=func, all_vals=vals)
        assert (Hmat[0] == [0, 1]).all()
        assert (Hmat[1] == [1, 0]).all()

        func = "x0 * x1**3"
        vals = [3, 5]
        Hmat = get_hessian(func_str=func, all_vals=vals)
        assert (Hmat[0] == [0, 75]).all()
        assert (Hmat[1] == [75, 90]).all()

        func = "x0 * x1**x0 * x1"
        vals = [3, 5]
        Hmat = get_hessian(func_str=func, all_vals=vals)
        #assert (Hmat[0] == [0, 375]).all()
        #assert (Hmat[1] == [375, 900]).all() 
        #print("Expected = " + str([[0, np.log(5)+1],[np.log(5)+1, 3/5]]) + "\n")

    def test_jacobian(self):
        
        func_list = ['x0*x2+x1', 'x1']
        val_list = [2,3,4]
        values, jacobian = get_val_jacobian(func_list, val_list)
        assert (values == [11, 3]).all()
        assert (jacobian[0] == [4, 1, 2]).all()
        assert (jacobian[1] == [0, 1, 0]).all()

        func_list = ["x0*exp(x0+9)"]
        val_list = [2]
        values, jacobian = get_val_jacobian(func_list, val_list)
        assert (values == [2*np.exp(11)]).all()
        assert (jacobian[0] == [np.exp(11) + 2*np.exp(11)]).all()

        func_list = ["x1*exp(x0+9)", "x1*sin(x0)"]
        val_list = [2,5]
        values, jacobian = get_val_jacobian(func_list, val_list)
        assert (values == [5*np.exp(11), 5*np.sin(2)]).all()
        assert (jacobian[0] == [5*np.exp(11), np.exp(11)]).all()
        assert (jacobian[1] == [5*np.cos(2), np.sin(2)]).all()

        func_list = ["cos(exp(x0))*x2+x1*x2"]
        val_list = [1,2,3]
        values, jacobian = get_val_jacobian(func_list, val_list)
        assert (values == [6+3*np.cos(np.exp(1))]).all()
        assert (np.round(jacobian[0],6) == np.round([-3*np.sin(np.exp(1))*np.exp(1), 3, np.cos(np.exp(1))+2],6)).all()

        #---------- Old code? -----------#

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

        

