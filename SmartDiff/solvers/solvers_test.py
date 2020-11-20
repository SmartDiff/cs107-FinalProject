# usage: pytest -q solvers_test.py 

from SmartDiff.solvers.integrator import AutoDiff as AD
import SmartDiff.solvers.element_op as el
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

      with pytest.raises(AttributeError):
        x = "s"
        f = el.power(x,3)

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

      with pytest.raises(AttributeError):
        x = "s"
        f = el.log(x,3)

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

      with pytest.raises(AttributeError):
        x = "hello"
        f = el.exp(x)

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

      with pytest.raises(AttributeError):
        x = "World"
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

      with pytest.raises(AttributeError):
        x = "!"
        f = el.sin(x)

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

      with pytest.raises(AttributeError):
        x = "ABC"
        f = el.cos(x)

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

      with pytest.raises(AttributeError):
        x = "xyz"
        f = el.tan(x)

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

      with pytest.raises(AttributeError):
        x = "."
        f = el.arcsin(x)

    def test_arccos(self):
      x = AD(0.5)
      f = el.arccos(x)
      assert (f.val,f.der) == (np.arccos(0.5), -1/np.sqrt(1 - 0.5**2) * 0.5)

      x = 0.5
      f = el.arccos(x)
      assert (f.val,f.der) == (np.arccos(0.5), 0)

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

      with pytest.raises(AttributeError):
        x = "--"
        f = el.arccos(x)

    def test_arctan(self):
      x = AD(0.5)
      f = el.arctan(x)
      assert (f.val,f.der) == (np.arctan(0.5), 1/(1 + 0.5**2))

      x = 0.5
      f = el.arctan(x)
      assert (f.val,f.der) == (np.arctan(0.5), 0)

      with pytest.raises(AttributeError):
        x = "AD(0.5)"
        f = el.arctan(x)

    def test_sinh(self):
      x = AD(0.5)
      f = el.sinh(x)
      assert (f.val,f.der) == (np.sinh(0.5),np.cosh(0.5))

      x = 0.5
      f = el.sinh(x)
      assert (f.val,f.der) == (np.sinh(0.5), 0)

      with pytest.raises(AttributeError):
        x = "0.5"
        f = el.sinh(x)

    def test_cosh(self):
      x = AD(0.5)
      f = el.cosh(x)
      assert (f.val,f.der) == (np.cosh(0.5),np.sinh(0.5))

      x = 0.5
      f = el.cosh(x)
      assert (f.val,f.der) == (np.cosh(0.5), 0)

      with pytest.raises(AttributeError):
        x = "0.5"
        f = el.cosh(x)

    def test_tanh(self):
      x = AD(0.5)
      f = el.tanh(x)
      assert (f.val,f.der) == (np.tanh(0.5),1 - np.tanh(0.5)**2)

      x = 0.5
      f = el.tanh(x)
      assert (f.val,f.der) == (np.tanh(0.5), 0)

      with pytest.raises(AttributeError):
        x = "0.5"
        f = el.tanh(x)

    def test_add(self):
      x = AD(5)
      f = el.power(x,2) + 5
      assert (f.val,f.der) == (30,10)

      f = 5 + el.power(x,2)
      assert (f.val,f.der) == (30,10)

      f = el.power(x,2) + 5*x
      assert (f.val,f.der) == (50,15)

      f = x*5 + el.power(x,2)
      assert (f.val,f.der) == (50,15)

      with pytest.raises(AttributeError):
        f = el.power(x,2) + "5"

    def test_sub(self):
      x = AD(5)
      f = el.power(x,2) + -5*x
      assert (f.val,f.der) == (0,5)

      f = el.power(x,2) - 50
      assert (f.val,f.der) == (-25,10)

      f = - 50 + el.power(x,2) 
      assert (f.val,f.der) == (-25,10)

      f = 50 - el.power(x,2) 
      assert (f.val,f.der) == (25,-10)

      f = -5*x + el.power(x,2)
      assert (f.val,f.der) == (0,5)

      f = -x*5 + el.power(x,2)
      assert (f.val,f.der) == (0,5)

      f = el.power(x,2) - 5*x
      assert (f.val,f.der) == (0,5)

      f = x*5 - el.power(x,2)
      assert (f.val,f.der) == (0,-5)

      with pytest.raises(AttributeError):
        f = el.sin(x) - "5"

    def test_mul(self):
      x = AD(4)
      f = el.log(x,2) * 3**x
      assert (f.val,f.der) == (162, 81/(4*np.log(2)) + 162*np.log(3))

      f = 3**x * el.log(x,2)  
      assert (f.val,f.der) == (162, 81/(4*np.log(2)) + 162*np.log(3))

      with pytest.raises(AttributeError):
          f = x * "5"

    def test_truediv(self):
      x = AD(4)
      f = el.log(x,2) / 3**x
      assert (f.val,f.der) == (2/81, (81/(4*np.log(2)) - 162*np.log(3))/3**8)

      f = el.sin(x) / 4
      assert (f.val,f.der) == ((np.sin(4))/4, (np.cos(4))/4)

      with pytest.raises(ZeroDivisionError):
        f = el.cos(x) / el.sin(0)

      with pytest.raises(ZeroDivisionError):
        f = el.cos(x) / 0

      f = 3**x / el.log(x,2)  
      assert (f.val,f.der) == (81/2, (162*np.log(3) - 81/(4*np.log(2)))/(np.log(4)/np.log(2))**2)

      with pytest.raises(AttributeError):
        f = el.cos(x) / "el.sin(0)"

    def test_pow(self):
      x = AD(2)
      f = (el.power(x,2))**x
      assert (f.val,f.der) == (16, 16*np.log(4) + 32)

      f = (el.power(x,2))**3
      assert (f.val,f.der) == (64, 192)

      with pytest.raises(AttributeError):
        f = (el.power(x,2))**"3"

      f = (2**x)**x
      assert (f.val,f.der) == (16, 16*np.log(16))

      f = x**(2**x)
      assert (f.val,f.der) == (16, 32 + 64*(np.log(2)**2))

    def test_pos(self):
      x = AD(100)
      assert (x.val,x.der) == (100, 1)

      f = el.log(x,10) + x 
      assert (f.val,f.der) == (102, 1/(100*np.log(10))+1)

