import math

from SmartDiff.solvers.element_op import *

# The evaluator takes as input a string representation of a math expression
# and parses the string by iteratively building the AD object applied with
# the corresponding operations starting from the variable

class PyExpression_Formatter(object):

  def __init__(self, valid_special_chars=None):
    if valid_special_chars:
      self.special_char_set = valid_special_chars
    else:
      self.special_char_set = {"+", "-", "*", "/", ".", "(", ")", ",", "_", " "}

  def format_to_pyexpr(self, input_str):
    """
    This function formats the input string of math function into python code

    :param input_str: a math function
    :return: a string of compilable python code
    """
    # Check input validity
    valid_out = self.is_valid_input(input_str)
    if valid_out == 1:
      raise SyntaxError("Input function has unmatched parenthesis!")
    elif valid_out == 2:
      raise SyntaxError("Input function contains invalid character!")
    # e --> math.e
    i = 0
    pyexpr = []
    while i < len(input_str) - 1:
      c = input_str[i]
      if c.isspace():
        i += 1
      elif c == "e":
        next_c = input_str[i+1]
        if next_c in {"x", "r"}:
          pyexpr.append(c)
          pyexpr.append(next_c)
          i += 2
        else:
          pyexpr.append("math.e")
          i += 1
      elif c == "p" and input_str[i+1] == "i":
          pyexpr.append("math.pi")
          i += 2
      else:
        pyexpr.append(c)
        i += 1

    if i < len(input_str):
      if input_str[-1] == "e":
        pyexpr.append("math.e")
      elif not input_str[-1].isspace():
        pyexpr.append(input_str[-1])
    return "".join(pyexpr)

  def is_valid_input(self, input_str):
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
        # check if a non alphanumeral char is in our special character set
        if not c.isalnum() and c not in self.special_char_set:
          return 2
        continue
      if suffix_demand == -1:
        return 1
    return 0 if suffix_demand == 0 else 1


  # code = compile("math.log(3)", "<string>", "eval")
  # print(eval(code))
  # print(eval("math.log(4)"))
  #
  # s = "pow(x, 2) * 5"
  # x = AD(3)
  # print(eval(s))