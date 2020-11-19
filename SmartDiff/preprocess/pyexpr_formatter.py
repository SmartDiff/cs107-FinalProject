import math

from SmartDiff.solvers.element_op import *

# The evaluator takes as input a string representation of a math expression
# and parses the string by iteratively building the AD object applied with
# the corresponding operations starting from the variable

class PyExpression_Formatter(object):

  def __init__(self):
    pass

  def format_to_pyexpr(self, input_str):
    """
    This function formats the input string of math function into python code

    :param input_str: a math function
    :return: a string of compilable python code
    """
    # Check input validity
    if not self.valid_parenthesis(input_str):
      raise SyntaxError("Input function has unmatched parenthesis!")
    # e --> math.e
    i = 0
    pyexpr = []
    while i < len(input_str) - 1:
      c = input_str[i]
      if c == "e":
        next_c = input_str[i+1]
        if next_c in {"x", "r"}:
          pyexpr.append(c)
          pyexpr.append(next_c)
          i += 2
          continue
        else:
          pyexpr.append("math.e")
      else:
        pyexpr.append(c)
      i += 1
    if input_str[-1] == "e":
      pyexpr.append("math.e")
    else:
      pyexpr.append(input_str[-1])
    return "".join(pyexpr)

  def valid_parenthesis(self, input_str):
    """
    This function returns true if the input string has valid parenthesis.

    :param input_str: a string of math function
    :return: True if the input has valid parenthesis
    """
    suffix_demand = 0
    for c in input_str:
      if c == "(":
        suffix_demand += 1
      elif c == ")":
        suffix_demand -= 1
      else:
        continue
      if suffix_demand == -1:
        return False
    return suffix_demand == 0



if __name__ == "__main__":
  formatter = PyExpression_Formatter()
  input1 = "(2 + 4) * 9"
  input2 = "e**(x)"
  input3 = "sin(x)+5.88"
  print(formatter.format_to_pyexpr(input1))
  print(formatter.format_to_pyexpr(input2))
  print(formatter.format_to_pyexpr(input3))

  #
  # code = compile("math.log(3)", "<string>", "eval")
  # print(eval(code))
  # print(eval("math.log(4)"))
  #
  # s = "pow(x, 2) * 5"
  # x = AD(3)
  # print(eval(s))
