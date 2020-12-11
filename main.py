import sys
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import math
from sympy import symbols
from SmartDiff.preprocess.pyexpr_formatter import PyExpression_Formatter
from SmartDiff.solvers.element_op import *
# from SmartDiff.multivariate import *

# global Ui_MainWindow, Ui_SecondDiag
Ui_MainWindow, QtBaseClass = uic.loadUiType('SmartDiff/GUI/step1.ui')  # .ui drawn in Qt Designer
Ui_FourthDiag, QtBaseClass4 = uic.loadUiType('SmartDiff/GUI/step4.ui')  # .ui drawn in Qt Designer
Ui_FifthDiag, QtBaseClass5 = uic.loadUiType('SmartDiff/GUI/step5.ui')  # .ui drawn in Qt Designer

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
        # Need to make the dialog window larger to show the title
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
        xs = symbols('x(1:%d)' % (self.InputDim+1))
        x_list = []
        for i, x in enumerate(xs):
            x_list.append(self._PointEval(str(x)))
        return x_list

    def _PointEval(self, string):
        '''
        :param string: x1, x2, x3, ... to put in the QInputDialog box title
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
        fs = symbols('f(1:%d)' % (self.FuncDim+1))
        f_list = []
        for i, f in enumerate(fs):
            f_list.append(self._FuncEval(str(f)))
        return f_list

    def _FuncEval(self, string):
        '''
        :param string: f1, f2, f3,... to put in the QInputDialog box title
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
        # print(f"Evaluating {self.func} at {self.val}")  # for testing only, to be commented out in the future
        # step 4
        dlg4 = FourthDiag(self.InputDim, self.FuncDim, self.val, self.func)
        dlg4.exec_()


class FourthDiag(QtWidgets.QDialog, Ui_FourthDiag):

    def __init__(self, InputDim, FuncDim, val, func):
        QtWidgets.QDialog.__init__(self)
        # load a dialogue based on user input from step one
        self.InputDim = InputDim
        self.FuncDim = FuncDim
        self.val = val
        self.func = func

        self.setupUi(self)
        Ui_FourthDiag, QtBaseClass4 = uic.loadUiType('SmartDiff/GUI/step4.ui')
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
        xs = symbols('x(1:%d)' % (self.InputDim + 1))
        fs = symbols('f(1:%d)' % (self.FuncDim + 1))
        var_msg = ""
        func_msg = ""
        for i, x in enumerate(self.val):
            var_msg += f"{str(xs[i])}={self.val[i]}\n"
        for i, f in enumerate(self.func):
            func_msg += f"{str(fs[i])}={self.func[i]}\n"
        self.ValInput.setPlainText(var_msg)
        # self.ValInput.setlineWrap(True)
        self.ValInput.setReadOnly(True)
        self.ValInput.setStyleSheet("background: white; color: black")
        self.FuncInput.setPlainText(func_msg)
        # self.FuncInput.setWordWrap(True)
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
        dlg5 = FifthDiag(self.InputDim, self.FuncDim, val, der, msg, self.DisVal)
        dlg5.exec_()

    ### NEED TO UPDATE
    def compValDer(self):
        '''
        Format the user input in step 2 and 3 into strings and call AD modules in solvers
        :return:
        np.array, np.array, str: function value, function derivative, error message
        '''
        err_msg = ""
        # instantiate a formatter object
        formatter = PyExpression_Formatter()
        xs = symbols('x(1:%d)' % (self.InputDim+1))
        var_map = {str(xs[i]): self.val[i] for i in range(self.InputDim)}
        var_map.update({"x": self.val[0]})
        if self.InputDim == 2:
            var_map.update({"y": self.val[1]})
        if self.InputDim == 3:
            var_map.update({"y": self.val[1], "z": self.val[2]})
        func_map = {"pi": math.pi, "e": math.e, "power": power, "log": log, "exp": exp, "sqrt": sqrt, "sin": sin,
                    "cos": cos, "tan": tan, "arcsin": arcsin, "arccos": arccos, "arctan": arctan, "sinh": sinh,
                    "cosh": cosh, "tanh": tanh}
        var_map.update(func_map)
        # Get user input and check if it's valid
        for i, f in enumerate(self.func):
            is_valid = formatter.is_valid_input(f)
            if is_valid == 0:
                try:
                    AD_out = eval(f, var_map)
                    val = AD_out.val
                    der = AD_out.der
                    return np.array([val]), np.array([der]), err_msg  # need to change for higher dim
                except ValueError as e:
                    return np.zeros(1), np.zeros(1), str(e)+" "
                except AttributeError as e:
                    return np.zeros(1), np.zeros(1), str(e)+" "
                except ZeroDivisionError as e:
                    return np.zeros(1), np.zeros(1), str(e)+" "
            else:
                if is_valid == 1:
                    return np.zeros(1), np.zeros(1), "Input function has unmatched parenthesis!"
                else:
                    return np.zeros(1), np.zeros(1), "Input function contains invalid character!"


class FifthDiag(QtWidgets.QDialog, Ui_FifthDiag):

    def __init__(self, InputDim, FuncDim, Val, Der, Msg, DisVal):
        QtWidgets.QDialog.__init__(self)
        # load a dialogue based on user input from step one
        self.InputDim = InputDim
        self.FuncDim = FuncDim
        self.val = Val
        self.der = Der
        self.msg = Msg
        self.DisVal = DisVal
        Ui_FifthDiag, QtBaseClass5 = uic.loadUiType('SmartDiff/GUI/step5.ui')
        Ui_FifthDiag.__init__(self)
        self.setupUi(self)

        # populate the boxes based on user input in step 2 and 3
        self.ResultDisplay()
        # stop the program when user clicks quit
        self.quitButton.clicked.connect(self.onClickQuit)

    ### NEED TO CHANGE
    def ResultDisplay(self):
        '''
        Display the results of the SmartDiff, based on user input
        :return:
        None
        '''
        self.ErrMsg.setWordWrap(True)
        fval_list = [self.f1Val, self.f2Val, self.f3Val]
        der_list = [self.f11Der, self.f12Der, self.f13Der,
                    self.f21Der, self.f22Der, self.f23Der,
                    self.f31Der, self.f32Der, self.f33Der]
        for fval in fval_list:
            fval.setText("N/A")
            fval.setEnabled(False)
            fval.setStyleSheet("background: white; color: black")
        for der in der_list:
            der.setText("N/A")
            der.setEnabled(False)
            der.setStyleSheet("background: white; color: black")

        if self.InputDim == 1 and self.FuncDim == 1:
            # display error message, if any
            if self.msg == "":
                self.ErrMsg.setText("Success! See results below")
                if self.DisVal:
                    self.f1Val.setText(str(np.round(self.val[0], 2)))
                    self.f11Der.setText(str(np.round(self.der[0], 2)))
                else:
                    self.f11Der.setText(str(np.round(self.der[0], 2)))
            else:
                self.ErrMsg.setText("Failure: " + self.msg +
                                    "Close windows of step 4 and 5 and start again from step 1.")
        else:
            raise NotImplementedError

    def onClickQuit(self):
        '''
        stop the program when user clicks quit
        :return:
        None
        '''
        sys.exit()


#     def PointEval1(self, var):
#         '''
#         User input the number to evaluate
#         Note: Use the input validation function to only allow float
#         :return:
#         None
#         '''
#         # try:
#         #     self.val_vec = np.array([float(self.xVal.text())])
#         # except ValueError:
#         #     ty = type(self.xVal.text())
#         #     self.WarnLabel.setText(f"Warning: You entered {ty}. Please enter a float")
#         reg_ex = QtCore.QRegExp("[-+]?[0-9]*\.?[0-9]+")  # regex for float
#         input_validator = QtGui.QRegExpValidator(reg_ex)
#         var.setValidator(input_validator)
#         var.setMaxLength(6)
#         var.setAlignment(QtCore.Qt.AlignRight)
#         text, okPressed = QtWidgets.QInputDialog.getText(self, "Get text", "Your name:", var, "")
#         if okPressed and text != '':
#             print(text)
#             return text
#         # def enterPress():
#         #     return txt_input[-1]
#         # def textchanged(text):
#         #     print("content of the txt box: " + text)
#         #     txt_input.append(text)
#         # var.textChanged.connect(textchanged)
#         # return ent.clicked.connect(enterPress())
#         # if var.text() == "":
#         #     return 0
#         # return var.text()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
