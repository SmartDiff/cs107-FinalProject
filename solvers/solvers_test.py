# usage: pytest -q solvers_test.py 

from integrator import AutoDiffToy as AD
import element_op as el
import numpy as np

class TestElemOp:

    def test_pow(self):
        x = AD(5)
        f = el.power(x,2)
        assert (f.val,f.der) == (25,10)

    def test_log(self):
        x = AD(12)
        f = el.log(x,10)
        assert (f.val,f.der) == (np.log(12) / np.log(10), 1/(12 * np.log(10)))

    def test_exp(self):
        x = AD(2)
        f = el.exp(x,3)
        assert (f.val,f.der) == (3**(2), 3**(2) * np.log(3))

    def test_sqrt(self):
        x = AD(4)
        f = el.sqrt(x)
        assert (f.val,f.der) == (2, 0.5 * 4 ** (-0.5))

    def test_sin(self):
        x = AD(0)
        f = el.sin(x)
        assert (f.val,f.der) == (0.0,1.0)

    def test_cos(self):
        x = AD(90)
        f = el.cos(x)
        assert (f.val,f.der) == (np.cos(90),-np.sin(90))

    def test_tan(self):
        x = AD(90)
        f = el.tan(x)
        assert (f.val,f.der) == (np.tan(90), 1/(np.cos(90))**2)

    def test_arcsin(self):
        x = AD(0.5)
        f = el.arcsin(x)
        assert (f.val,f.der) == (np.arcsin(0.5), 1/np.sqrt(1 - 0.5**2) * 0.5)

    def test_arccos(self):
        x = AD(0.5)
        f = el.arccos(x)
        assert (f.val,f.der) == (np.arccos(0.5), -1/np.sqrt(1 - 0.5**2) * 0.5)

    def test_arctan(self):
        x = AD(0.5)
        f = el.arctan(x)
        assert (f.val,f.der) == (np.arctan(0.5), 1/(1 + 0.5**2))

    def test_sinh(self):
        x = AD(0.5)
        f = el.sinh(x)
        assert (f.val,f.der) == (np.sinh(0.5),np.cosh(0.5))

    def test_cosh(self):
        x = AD(0.5)
        f = el.cosh(x)
        assert (f.val,f.der) == (np.cosh(0.5),np.sinh(0.5))

    def test_tanh(self):
        x = AD(0.5)
        f = el.tanh(x)
        assert (f.val,f.der) == (np.tanh(0.5),1 - np.tanh(0.5)**2)

    def test_sum(self):
        x = AD(5)
        f = el.power(x,2) + 5
        assert (f.val,f.der) == (30,10)

        f = 5 + el.power(x,2)
        assert (f.val,f.der) == (30,10)

        f = el.power(x,2) + 5*x
        assert (f.val,f.der) == (50,15)

        f = x*5 + el.power(x,2)
        assert (f.val,f.der) == (50,15)

    def test_sub(self):
        x = AD(5)
        f = el.power(x,2) + -5*x
        assert (f.val,f.der) == (0,5)

        f = -x*5 + el.power(x,2)
        assert (f.val,f.der) == (0,5)

        f = el.power(x,2) - 5*x
        assert (f.val,f.der) == (0,5)

        f = x*5 - el.power(x,2)
        assert (f.val,f.der) == (0,-5)

    def test_mul(self):
        x = AD(4)
        f = el.log(x,2) * el.exp(x,3)
        assert (f.val,f.der) == (162, 81/(4*np.log(2)) + 162*np.log(3))

        f = el.exp(x,3) * el.log(x,2)  
        assert (f.val,f.der) == (162, 81/(4*np.log(2)) + 162*np.log(3))

    def test_truediv(self):
        x = AD(4)
        f = el.log(x,2) / el.exp(x,3)
        assert (f.val,f.der) == (2/81, (81/(4*np.log(2)) - 162*np.log(3))/3**8)

        f = el.exp(x,3) / el.log(x,2)  
        assert (f.val,f.der) == (81/2, (162*np.log(3) - 81/(4*np.log(2)))/(np.log(4)/np.log(2))**2)

        





        
