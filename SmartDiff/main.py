import sys
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import math
from sympy import symbols
from SmartDiff.preprocess.pyexpr_formatter import PyExpression_Formatter
from SmartDiff.solvers.element_op import *
from SmartDiff.solvers.multivariate import *

# global Ui_MainWindow, Ui_SecondDiag
Ui_MainWindow, QtBaseClass = uic.loadUiType('GUI/step1.ui')  # .ui drawn in Qt Designer
Ui_FourthDiag, QtBaseClass4 = uic.loadUiType('GUI/step4.ui')  # .ui drawn in Qt Designer
Ui_FifthDiag, QtBaseClass5 = uic.loadUiType('GUI/step5.ui')  # .ui drawn in Qt Designer

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setWindowTitle("SmartDiff")
        self.setupUi(self)
        self.Instructions.setWordWrap(True)

        self.FuncDim = 1  # default
        self.InputDim = 1  # default

        self.order = self.setupOrder()
        self.val = np.zeros(1)
        self.func = None

        # once OK button is pressed, go to step 2 and 3 and 4
        self.OKButton.clicked.connect(self.onClickOK)

    def setupOrder(self):
        '''
        :param: None
        :return:
        int, user input: the order of derivative to calculate
        '''
        order, okPressed = QtWidgets.QInputDialog.getInt(self, "Step 0: Input the order of derivative", "N = ",
                                                         value=1, min=1)
        if okPressed:
            return order

    def SetDimInput(self):
        '''
        Set up the function and input dimensions
        :return:
        user input of function dimension and input dimension
        '''
        self.FuncDim = int(self.FuncDimBox.currentText())
        self.InputDim = int(self.InputDimBox.currentText())

    def DisplayOrderWarning(self):
        '''
        Display warnings if the order is too high
        Reset the self.FuncDim and self.InputDim based on self.order
        :return:
        None
        '''
        warning_msg = ""
        if self.order > 10:
            self.FuncDim = 1
            self.InputDim = 1
            warning_msg += "Warning: the order of derivative is high, so the computation may take a long time. "
            if self.FuncDim > 1:
                self.FuncDim = 1
                warning_msg += "Reset function dimension to 1. "
            if self.InputDim > 1:
                self.InputDim = 1
                warning_msg += "Reset input dimension to 1. "
        elif self.order > 2:
            if self.FuncDim > 1:
                self.FuncDim = 1
                warning_msg += "Warning: Reset function dimension to 1. "
            if self.InputDim > 1:
                self.InputDim = 1
                warning_msg += "Warning: Reset input dimension to 1. "
        elif self.order == 2:
            if self.FuncDim > 1:
                self.FuncDim = 1
                warning_msg += "Warning: Reset function dimension to 1. Hessian is not supported for vector-valued " \
                               "functions."
        self.ErrMsg.setText(warning_msg)
        self.ErrMsg.setWordWrap(True)

    def PointEval(self):
        '''
        get user input values for each variable
        :return:
        np array of the user input (size = self.InputDim from step 1)
        '''
        xs = symbols('x:%d' % self.InputDim)
        x_list = []
        for i in range(self.InputDim):
            x_list.append(self._PointEval(str(xs[i])))
        return x_list

    def _PointEval(self, string):
        '''
        :param string: x0, x1, x2, ... to put in the QInputDialog box title
        :return:
        double, user input, default 0, min -100, max 100, up to 4 decimals
        '''
        # Need to make the dialog window larger to show the title
        num, okPressed = QtWidgets.QInputDialog.getDouble(self, "Step 2: Input the evaluating point", string+" = ",
                                                          0, -100, 100, 4)
        if okPressed and num != '':
            return num

    def FuncEval(self):
        '''
        get user input function to differentiate
        :return:
        a list of user input function (each element is a component of the function)
        '''
        fs = symbols('f:%d' % self.FuncDim)
        f_list = []
        for i, f in enumerate(fs):
            f_list.append(self._FuncEval(str(f)))
        return f_list

    def _FuncEval(self, string):
        '''
        :param string: f0, f1, f2,... to put in the QInputDialog box title
        :return:
        str, user input
        '''
        # Need to make the dialog window larger to show the title
        func, okPressed = QtWidgets.QInputDialog.getText(self, "Step 3: Input the function", string+" = ",
                                                         QtWidgets.QLineEdit.Normal, "")
        if okPressed and func != '':
            return func

    def onClickOK(self):
        '''
        Trigger step 2: User puts in values of the variables to evaluate
        Trigger step 3: User puts in the function to evaluate and differentiate (working on this now)
        Trigger step 4: User confirms the input is correct and chooses whether to show the function value
        :return: None
        '''
        self.SetDimInput()
        self.DisplayOrderWarning()
        # step 2
        self.val = self.PointEval()
        # step 3
        self.func = self.FuncEval()
        # step 4
        dlg4 = FourthDiag(self.InputDim, self.FuncDim, self.val, self.func, self.order)
        dlg4.exec_()


class FourthDiag(QtWidgets.QDialog, Ui_FourthDiag):

    def __init__(self, InputDim, FuncDim, val, func, order):
        QtWidgets.QDialog.__init__(self)
        # load a dialogue based on user input from step one
        self.InputDim = InputDim
        self.FuncDim = FuncDim
        self.val = val
        self.func = func
        self.order = order

        self.setupUi(self)
        Ui_FourthDiag, QtBaseClass4 = uic.loadUiType('GUI/step4.ui')
        Ui_FourthDiag.__init__(self)
        self.DisVal = False
        # populate the boxes based on user input in step 2 and 3
        self.SetupValFunc()
        # checkBox to select whether the user wants to show the function values
        self.checkBox.clicked.connect(self.setDisVal)
        # click OK button to start the computation
        self.OKButton.clicked.connect(self.onClickOK)

    def SetupValFunc(self):
        '''
        Display the function expressions, variable value input based on user input
        :return: None
        '''
        xs = symbols('x:%d' % self.InputDim)
        fs = symbols('f:%d' % self.FuncDim)
        var_msg = ""
        func_msg = ""
        for i, x in enumerate(self.val):
            var_msg += f"{str(xs[i])}={self.val[i]}\n"
        for i, f in enumerate(self.func):
            func_msg += f"{str(fs[i])}={self.func[i]}\n"
        self.ValInput.setPlainText(var_msg)
        self.ValInput.setReadOnly(True)
        self.ValInput.setStyleSheet("background: white; color: black")
        self.FuncInput.setPlainText(func_msg)
        self.FuncInput.setReadOnly(True)
        self.FuncInput.setStyleSheet("background: white; color: black")

    def setDisVal(self):
        '''
        Set whether the GUI displays the function value in addition to the derivative
        :return:
        None
        '''
        self.DisVal = not self.DisVal

    def onClickOK(self):
        '''
        sets up step 5
        :return:
        The function value and derivative at the specific point from user input
        '''
        # compute function value and derivative and get error messsage

        val, der, msg = self.compValDer()
        # step 5
        dlg5 = FifthDiag(val, der, msg, self.order, self.DisVal)
        dlg5.exec_()

    def compValDer(self):
        '''
        Format the user input in step 2 and 3 into strings and call AD modules in solvers
        :return:
        np.array, np.array, str: function value, function derivative, error message
        '''
        # instantiate a formatter object
        formatter = PyExpression_Formatter()
        # xs = symbols('x:%d' % self.InputDim)
        # var_map = {str(xs[i]): self.val[i] for i in range(self.InputDim)}
        # var_map.update({"x": self.val[0]})
        # if self.InputDim == 2:
        #     var_map.update({"y": self.val[1]})
        # if self.InputDim == 3:
        #     var_map.update({"y": self.val[1], "z": self.val[2]})
        func_map = {"pi": math.pi, "e": math.e, "power": power, "log": log, "exp": exp, "sqrt": sqrt, "sin": sin,
                    "cos": cos, "tan": tan, "arcsin": arcsin, "arccos": arccos, "arctan": arctan, "sinh": sinh,
                    "cosh": cosh, "tanh": tanh}
        # var_map.update(func_map)

        err_msg = ""
        # Get user input and check if it's valid
        for i, f in enumerate(self.func):
            is_valid = formatter.is_valid_input(f)
            if is_valid == 1:
                return np.zeros(1), np.zeros(1), f"Input function has unmatched parenthesis in f{i}!"
            elif is_valid != 0:
                return np.zeros(1), np.zeros(1), f"Input function contains invalid character in f{i}!"

        # proceed if all the functions pass user input checking
        try:
            if self.FuncDim == 1:
                if self.order == 1:
                    val, der = get_val_jacobian(self.func, self.val, eval_func_map=func_map)
                    return val, der, err_msg
                elif self.order == 2:
                    val, _ = get_val_jacobian(self.func, self.val, eval_func_map=func_map)
                    der = get_hessian(self.func[0], self.val, eval_func_map=func_map)
                    return val, der, err_msg
                else:  # self.order > 2 and self.FuncDim == 1 and self.InputDim == 1
                    var_map = {'x': AutoDiff(self.val[0], N=self.order), 'x1': AutoDiff(self.val[0], N=self.order), 'x0': AutoDiff(self.val[0], N=self.order)}
                    var_map.update(func_map)
                    AD_out = eval(self.func[0], var_map)
                    try:
                        val = AD_out.val
                        der = AD_out.der[-1]
                    except AttributeError:
                        val = AD_out
                        der = 1
                    return val, der, err_msg
            else:
                # self.FuncDim > 1 and self.order == 1
                val, der = get_val_jacobian(self.func, self.val, eval_func_map=func_map)
                return val, der, err_msg
        except ValueError as e:
            return np.zeros(1), np.zeros(1), str(e)+" "
        except AttributeError as e:
            return np.zeros(1), np.zeros(1), str(e)+" "
        except ZeroDivisionError as e:
            return np.zeros(1), np.zeros(1), str(e)+" "
        except TypeError as e:
            return np.zeros(1), np.zeros(1), str(e)+" "


class FifthDiag(QtWidgets.QDialog, Ui_FifthDiag):

    def __init__(self, Val, Der, Msg, Order, DisVal):
        QtWidgets.QDialog.__init__(self)
        # load a dialogue based on user input from step one
        self.val = Val
        self.der = Der
        self.msg = Msg
        self.order = Order
        self.DisVal = DisVal
        Ui_FifthDiag, QtBaseClass5 = uic.loadUiType('GUI/step5.ui')
        Ui_FifthDiag.__init__(self)
        self.setupUi(self)

        # populate the boxes based on user input in step 2 and 3
        self.ResultDisplay()
        # stop the program when user clicks quit
        self.quitButton.clicked.connect(self.onClickQuit)

    def ResultDisplay(self):
        '''
        Display the results of the SmartDiff, based on user input
        :return:
        None
        '''
        self.ErrMsg.setWordWrap(True)
        self.FuncVal.setReadOnly(True)
        self.FuncVal.setStyleSheet("background: white; color: black")
        self.DerVal.setReadOnly(True)
        self.DerVal.setStyleSheet("background: white; color: black")

        if self.msg == "":
            self.ErrMsg.setText("Success! See results below")
            if self.DisVal:
                self.FuncVal.setPlainText(str(self.val))
            else:
                self.FuncVal.setPlainText("N/A")
            der_msg = f"{self.order}-order derivative:\n"
            self.DerVal.setPlainText(der_msg + str(self.der))
        else:
            self.FuncVal.setPlainText("N/A")
            self.DerVal.setPlainText("N/A")
            self.ErrMsg.setText("Failure: " + self.msg +
                                "Close windows of step 4 and 5 and start again from step 1.")

    def onClickQuit(self):
        '''
        Stop the program when user clicks quit
        :return:
        None
        '''
        sys.exit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
