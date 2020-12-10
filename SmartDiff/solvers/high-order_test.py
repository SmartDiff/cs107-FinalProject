
import SmartDiff.solvers.element_op as el
from SmartDiff.solvers.element_op import AutoDiff as AD
import numpy as np
import pytest


class NOrderTestElemOp:

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
    x = AD(0, N=4)
    f = el.sin(x)
    print(f)

    f = el.sin(x) + 3
    print(f)

    f = el.sin(x) - 2
    print(f)

    f = el.sin(x) * 4
    print(f)


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
    assert (round(g.val, 6), round(g.der[0], 6)) == (round(np.exp(1) ** 8, 6), round(12 * np.exp(1) ** 8, 6))

    with pytest.raises(AttributeError):
      x = "hello"
      f = el.exp(x)

    # N > 1
    for N in range(2, 10):
      x = AD(3, N=N)
      f = el.exp(x)
      print("N=%d" % N, f)

  def test_ln(self):
    # N = 1
    f = el.ln(np.e)
    assert (f.val, f.der) == (1, 0)

    x = AD(np.e)
    f = el.ln(x)
    assert (f.val, f.der) == (1, 1 / np.e)

    # N > 1
    # See explicit formula here: https://www.math24.net/higher-order-derivatives/#example2
    x = AD(np.e, N = 2)
    f = el.ln(x)
    assert (f.val, f.der[-1]) == (1, -1 / (np.e**2))

    x = AD(np.e**2, N=5)
    f = el.ln(x)
    assert (f.val, f.der[-1]) == (2, 24 / ((np.e**2) ** 5))

    # composite x
    x = AD(np.e, N=4)
    f = el.ln(x * 3 + 1)
    assert (f.val, f.der[-1]) == (np.log(np.e * 3 + 1), (-6 * 3**4) / ((np.e * 3 + 1) ** 4))

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
    f = 5*x / 3
    assert (f.val, f.der[-1]) == (5, 5/3)

    x = AD(4, N=2)
    f = x / x
    assert (f.val, f.der[-1]) == (1, 0)

  def test_rtruediv(self):
    x = AD(4, N=2)
    f = 3 / (2*x)
    assert (f.val, f.der[-1]) == (3/8, 3/64)

    x = AD(4, N=3)
    f = 3 / (2 * x)  # shows rtruediv should work
    assert (f.val, f.der[-1]) == (3 / 8, -9 / 4**4)


  def test_power(self):

    pass

  def test_AD_pow(self):
    pass





if __name__ == "__main__":
  test = NOrderTestElemOp()
  #test.test_sin()
  #test.test_exp()
  #test.test_ln()
  test.test_truediv()
  test.test_rtruediv()