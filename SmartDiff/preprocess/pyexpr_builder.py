import sympy

from SmartDiff.solvers.integrator import AutoDiffToy as AD
from SmartDiff.solvers.element_op import *

# The evaluator takes as input a string representation of a math expression
# and parses the string by iteratively building the AD object applied with
# the corresponding operations starting from the variable


import math
code = compile("math.log(3)","<string>", "eval")
print(eval(code))
print(eval("math.log(4)"))


s = "el.power(x, 2) * 5"
x = AD(3)
print(eval(s))


