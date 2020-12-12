import sys, os
sys.path.append('../')
from SmartDiff.preprocess.pyexpr_formatter import PyExpression_Formatter

class TestPyExpFormatter:

  def test_init_default(self):
    pef = PyExpression_Formatter()
    assert pef.special_char_set == {"+", "-", "*", "/", ".", "(", ")", ",", "_", " "}

  def test_init_custom(self):
    char_set = {"+", "-", "*", "/", ".", "(", ")", ",", "{", "}", "[", "]", "_"}
    pef = PyExpression_Formatter(char_set)
    assert pef.special_char_set == char_set

  def test_is_valid_input_out0(self):
    pef = PyExpression_Formatter()
    s0 = "3 * (x + 8) - 3.4 / 1.750"
    s1 = "sqrt(48) * 3.9 + x"
    s2 = "log(x+4) / 20.4"
    s3 = "sin(x-2) ** 3"
    s4 = "arccos(cos(x))"
    s5 = "tanh(0.2*x_0)"
    s6 = "arcsin(sin(x_1/3)) + 0.8"
    s7 = "arctan(tan(x_2 + x_3))"
    s8 = "power(x_0123, 3 / 9)"
    s9 = "2.8 * exp(x + 4)"

    test_inputs = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
    for i in range(len(test_inputs)):
      s = test_inputs[i]
      assert pef.is_valid_input(s) == 0, "Test %d failed!" % i

  def test_is_valid_input_out0_with_e(self):
    pef = PyExpression_Formatter()
    s0 = "e**e"
    s1 = "e*2"
    s2 = "e / 3.89"
    s3 = "(e+8.4) - x * 10"
    s4 = "log(exp(e))"
    s5 = "arcsin(x) + e"
    s6 = "sin(e / 100.001) + sqrt(e)"
    s7 = "power(x, 3) + e / 2 "

    test_e_inputs = [s0, s1, s2, s3, s4, s5, s6, s7]
    for i in range(len(test_e_inputs)):
      s = test_e_inputs[i]
      assert pef.is_valid_input(s) == 0, "Test %d failed!" % i

  def test_is_valid_input_out0_with_pi(self):
    pef = PyExpression_Formatter()
    s0 = "pi**(2*pi)"
    s1 = "pi + 3 - 4*pi"
    s2 = "pi / 3.89"
    s3 = "(pi+8.4) - x * 10 - pi"
    s4 = "log(exp(pi)) + 8 "
    s5 = "arcsin(4.9*pi) + e"
    s6 = "cos( pi / 100.001) + sqrt( pi )"
    s7 = "power(pi, 3) + e / 2 "

    test_inputs = [s0, s1, s2, s3, s4, s5, s6, s7]
    for i in range(len(test_inputs)):
      s = test_inputs[i]
      assert pef.is_valid_input(s) == 0, "Test %d failed!" % i

  def test_is_valid_input_out1(self):
    # This set of inputs contain unmatched parenthesis
    pef = PyExpression_Formatter()
    s0 = "log(9) + ((x + 3)*8 + 9"
    s1 = "sinh(y) + )65 * (9 + 8)"
    s2 = "1 * (x + (e + 5 * (pi + 99)) / (x_2 + 3)"
    s3 = "exp((x + 9)"

    test_inputs = [s0, s1, s2, s3]
    for i in range(len(test_inputs)):
      s = test_inputs[i]
      assert pef.is_valid_input(s) == 1, "Test %d failed!" % i

  def test_is_valid_input_out2(self):
    pef = PyExpression_Formatter()
    s0 = "log(9) + [(x + 3)*8 + 9]"
    s1 = "sinh(y) + {65 * (9 + 8)}"
    s2 = "1 * (x + (e + 5 * (pi + 99))) / (x_2 + 3);"
    s3 = "3^4 + x"

    test_inputs = [s0, s1, s2, s3]
    for i in range(len(test_inputs)):
      s = test_inputs[i]
      assert pef.is_valid_input(s) == 2, "Test %d failed!" % i

  def test_format_to_pyexpr_no_e_no_pi(self):
    pef = PyExpression_Formatter()
    s0 = "3 * (x + 8) - 3.4 / 1.750 "
    s1 = "sqrt(48) * 3.9 + x"
    s2 = "log(x+4) / 20.4"
    s3 = "sin(x-2) ** 3"
    s4 = "arccos(cos(x))"
    s5 = "tanh(0.2*x_0)"
    s6 = "arcsin(sin(x_1/3)) + 0.8 "
    s7 = "arctan(tan(x_2 + x_3))"
    s8 = "power(x_0123, 3 / 9) "
    s9 = "2.8 * exp(x + 4)"
    test_inputs = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
    expected_outs = [s.replace(" ", "") for s in test_inputs]

    for i in range(len(test_inputs)):
      s = test_inputs[i]
      assert pef.format_to_pyexpr(s) == expected_outs[i], "Test %d failed!" % i

  def test_format_to_pyexpr_with_e(self):
    pef = PyExpression_Formatter()
    s0 = "2.5 * e**e"
    s1 = "e / 3.89"
    s2 = "(e+8.4) - x * 10"
    s3 = "log(exp(e))"
    s4 = "arcsin(x) + e"
    s5 = "sin(e / 100.001) + sqrt(e)"
    s6 = "power(x, 3) + e / 2 "
    test_inputs = [s0, s1, s2, s3, s4, s5, s6]

    a0 = "2.5*math.e**math.e"
    a1 = "math.e/3.89"
    a2 = "(math.e+8.4)-x*10"
    a3 = "log(exp(math.e))"
    a4 = "arcsin(x)+math.e"
    a5 = "sin(math.e/100.001)+sqrt(math.e)"
    a6 = "power(x,3)+math.e/2"
    expected_outs = [a0, a1, a2, a3, a4, a5, a6]

    for i in range(len(test_inputs)):
      s = test_inputs[i]
      assert pef.format_to_pyexpr(s) == expected_outs[i], "Test %d failed!" % i

  def test_format_to_pyexpr_with_pi(self):
    pef = PyExpression_Formatter()
    s0 = "pi**(2*pi)"
    s1 = "pi + 3 - 4*pi"
    s2 = "pi / 3.89"
    s3 = "(pi+8.4) - x * 10 - pi"
    s4 = "log(exp(pi)) + 8 "
    s5 = "arcsin(4.9*pi) + e"
    s6 = "cos( pi / 100.001) + sqrt( pi )"
    s7 = "power(pi, 3) + e / 2 "
    test_inputs = [s0, s1, s2, s3, s4, s5, s6, s7]

    a0 = "math.pi**(2*math.pi)"
    a1 = "math.pi+3-4*math.pi"
    a2 = "math.pi/3.89"
    a3 = "(math.pi+8.4)-x*10-math.pi"
    a4 = "log(exp(math.pi))+8"
    a5 = "arcsin(4.9*math.pi)+math.e"
    a6 = "cos(math.pi/100.001)+sqrt(math.pi)"
    a7 = "power(math.pi,3)+math.e/2"
    test_outputs = [a0, a1, a2, a3, a4, a5, a6, a7]

    for i in range(len(test_inputs)):
      s = test_inputs[i]
      assert pef.format_to_pyexpr(s) == test_outputs[i], "Test %d failed!\n" % i
