import math
from SmartDiff.solvers.element_op import *
import sympy as sp



FUNC_MAP = {"pi": math.pi, "e": math.e, "power": power, "log": log, "exp": exp, "sqrt": sqrt, "sin": sin,
            "cos": cos, "tan": tan, "arcsin": arcsin, "arccos": arccos, "arctan": arctan, "sinh": sinh,
            "cosh": cosh, "tanh": tanh}

MATH_FUNC_MAP = {"pi": math.pi, "e": math.e, "power": math.pow, "log": math.log, "exp": math.exp, "sqrt": math.sqrt,
                 "sin": math.sin, "cos": math.cos, "tan": math.tan, "arcsin": sp.asin, "arccos": sp.acos,
                 "arctan": sp.atan, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh}
