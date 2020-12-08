from SmartDiff.solvers.integrator import AutoDiff as AD
import SmartDiff.solvers.element_op as el
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
    print(f.val, f.der)

    f = el.sin(x) + 3
    print(f.val, f.der)

    f = el.sin(x) - 2
    print(f.val, f.der)

    f = el.sin(x) * 4
    print(f.val, f.der)


  def test_log(self):
    pass



if __name__ == "__main__":
  test = NOrderTestElemOp()
  test.test_sin()