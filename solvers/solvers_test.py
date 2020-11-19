# usage: pytest -q solvers_test.py 

from integrator import AutoDiffToy as AD
import element_op as el
import numpy as np
import pytest

class TestElemOp:

    def test_power(self):
        x = AD(5)
        f = el.power(x,2)
        assert (f.val,f.der) == (25,10)

        x = 10
        f = el.power(x,3)
        assert (f.val,f.der) == (1000,0)

        x = 10.0
        f = el.power(x,3)
        assert (f.val,f.der) == (1000,0)


    def test_log(self):
        x = AD(12)
        f = el.log(x,10)
        assert (f.val,f.der) == (np.log(12) / np.log(10), 1/(12 * np.log(10)))

        x = 8
        f = el.log(x,2)
        assert (f.val,f.der) == (3, 0)

        x = 9.0
        f = el.log(x,3)
        assert (f.val,f.der) == (2, 0)

        x = AD(0)
        with pytest.raises(ValueError):
            f = el.log(x,2)
        
        x = 0
        with pytest.raises(ValueError):
            f = el.log(x,2)


    def test_exp(self):
        f = el.exp(1)
        assert (f.val,f.der) == (np.exp(1), 0)

        x = AD(1)
        f = el.exp(x)
        assert (f.val,f.der) == (np.exp(1), np.exp(1))

        x = AD(2)
        f = el.power(x,3)
        g = el.exp(f)
        assert (round(g.val, 6), round(g.der,6)) == (round(np.exp(1)**8,6), round(12*np.exp(1)**8,6))

    def test_sqrt(self):
        x = AD(4)
        f = el.sqrt(x)
        assert (f.val,f.der) == (2, 0.5 * 4 ** (-0.5))

        x = 4
        f = el.sqrt(x)
        assert (f.val,f.der) == (2, 0)

        x = 16.0
        f = el.sqrt(x)
        assert (f.val,f.der) == (4, 0)

        x = AD(-5)
        with pytest.raises(ValueError):
            f = el.sqrt(x)

        x = -5
        with pytest.raises(ValueError):
            f = el.sqrt(x)

    def test_sin(self):
        x = AD(0)
        f = el.sin(x)
        assert (f.val,f.der) == (0.0,1.0)

        x = 13
        f = el.sin(x)
        assert (f.val,f.der) == (np.sin(13),0)

        x = 13.0
        f = el.sin(x)
        assert (f.val,f.der) == (np.sin(13),0)

    def test_cos(self):
        x = AD(90)
        f = el.cos(x)
        assert (f.val,f.der) == (np.cos(90),-np.sin(90))

        x = 13
        f = el.cos(x)
        assert (f.val,f.der) == (np.cos(13),0)

        x = 13.0
        f = el.cos(x)
        assert (f.val,f.der) == (np.cos(13),0)

    def test_tan(self):
        x = AD(90)
        f = el.tan(x)
        assert (f.val,f.der) == (np.tan(90), 1/(np.cos(90))**2)

        x = 13
        f = el.tan(x)
        assert (f.val,f.der) == (np.tan(13),0)

        x = 13.0
        f = el.tan(x)
        assert (f.val,f.der) == (np.tan(13),0)

    def test_arcsin(self):
        x = AD(0.5)
        f = el.arcsin(x)
        assert (f.val,f.der) == (np.arcsin(0.5), 1/np.sqrt(1 - 0.5**2) * 0.5)

        x = 0
        f = el.arcsin(x)
        assert (f.val,f.der) == (np.arcsin(0),0)

        x = 1.0
        f = el.arcsin(x)
        assert (f.val,f.der) == (np.arcsin(1),0)

        x = -5
        with pytest.raises(ValueError):
            f = el.arcsin(x)

        x = 10.0
        with pytest.raises(ValueError):
            f = el.arcsin(x)

        x = AD(-5.0)
        with pytest.raises(ValueError):
            f = el.arcsin(x)

        x = AD(10)
        with pytest.raises(ValueError):
            f = el.arcsin(x)

    def test_arccos(self):
        x = AD(0.5)
        f = el.arccos(x)
        assert (f.val,f.der) == (np.arccos(0.5), -1/np.sqrt(1 - 0.5**2) * 0.5)

        x = -5
        with pytest.raises(ValueError):
            f = el.arccos(x)

        x = 10.0
        with pytest.raises(ValueError):
            f = el.arccos(x)

        x = AD(-5.0)
        with pytest.raises(ValueError):
            f = el.arccos(x)

        x = AD(10)
        with pytest.raises(ValueError):
            f = el.arccos(x)

    def test_arctan(self):
        x = AD(0.5)
        f = el.arctan(x)
        assert (f.val,f.der) == (np.arctan(0.5), 1/(1 + 0.5**2))

        x = 0.5
        f = el.arctan(x)
        assert (f.val,f.der) == (np.arctan(0.5), 0)

    def test_sinh(self):
        x = AD(0.5)
        f = el.sinh(x)
        assert (f.val,f.der) == (np.sinh(0.5),np.cosh(0.5))

        x = 0.5
        f = el.sinh(x)
        assert (f.val,f.der) == (np.sinh(0.5), 0)

    def test_cosh(self):
        x = AD(0.5)
        f = el.cosh(x)
        assert (f.val,f.der) == (np.cosh(0.5),np.sinh(0.5))

        x = 0.5
        f = el.cosh(x)
        assert (f.val,f.der) == (np.cosh(0.5), 0)

    def test_tanh(self):
        x = AD(0.5)
        f = el.tanh(x)
        assert (f.val,f.der) == (np.tanh(0.5),1 - np.tanh(0.5)**2)

        x = 0.5
        f = el.tanh(x)
        assert (f.val,f.der) == (np.tanh(0.5), 0)

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
        f = el.log(x,2) * 3**x
        assert (f.val,f.der) == (162, 81/(4*np.log(2)) + 162*np.log(3))

        f = 3**x * el.log(x,2)  
        assert (f.val,f.der) == (162, 81/(4*np.log(2)) + 162*np.log(3))

    def test_truediv(self):
        x = AD(4)
        f = el.log(x,2) / 3**x
        assert (f.val,f.der) == (2/81, (81/(4*np.log(2)) - 162*np.log(3))/3**8)

        f = 3**x / el.log(x,2)  
        assert (f.val,f.der) == (81/2, (162*np.log(3) - 81/(4*np.log(2)))/(np.log(4)/np.log(2))**2)

    def test_pow(self):
        x = AD(2)
        f = (el.power(x,2))**x
        assert (f.val,f.der) == (16, 16*np.log(4) + 32)

        f = (2**x)**x
        assert (f.val,f.der) == (16, 16*np.log(16))

        f = x**(2**x)
        assert (f.val,f.der) == (16, 32 + 64*(np.log(2)**2))

