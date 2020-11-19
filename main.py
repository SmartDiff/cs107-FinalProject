import sys
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import numpy as np
from SmartDiff.preprocess.pyexpr_formatter import PyExpression_Formatter
from SmartDiff.solvers.element_op import *

# global Ui_MainWindow, Ui_SecondDiag
Ui_MainWindow, QtBaseClass = uic.loadUiType('SmartDiff/GUI/step1.ui') # .ui drawn in Qt Designer
Ui_FourthDiag, QtBaseClass4 = uic.loadUiType('SmartDiff/GUI/step4.ui')

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setWindowTitle("SmartDiff")
        self.setupUi(self)
        self.Instructions.setWordWrap(True)

        self.FuncDim = 1  # default
        self.InputDim = 1  # default
        self.val = np.zeros(1)
        self.func = None

        # once OK button is pressed, go to step 2 and 3 and 4
        self.OKButton.clicked.connect(self.onClickOK)


    def SetDimInput(self):
        '''
        :return:
        user input of function dimension and input dimension
        '''
        self.FuncDim = int(self.FuncDimBox.currentText())
        self.InputDim = int(self.InputDimBox.currentText())

    def PointEval(self):
        '''
        get user input values for each variable
        :return:
        np array of the user input (size = self.InputDim from step 1)
        '''
        if self.InputDim == 1:
            num = self._PointEval("x_1")
            return np.array([num])
        elif self.InputDim > 1:
            raise NotImplementedError

    def _PointEval(self, string):
        '''
        :param string: x, y, z to put in the QInputDialog box title
        :return:
        double, user input, default 0, min -100, max 100, up to 4 decimals
        '''
        # Need to make the dialog window larger to show the title
        num, okPressed = QtWidgets.QInputDialog.getDouble(self, "Step 2: Input the evaluating point", string+" value:",
                                                          0, -100, 100, 4)
        if okPressed and num != '':
            return num

    def FuncEval(self):
        '''
        get user input function to differentiate
        :return:
        a list of user input function (each element is a component of the function)
        '''

        if self.InputDim == 1:
            func = self._FuncEval("1st")
            return list([func])

    def _FuncEval(self, string):
        '''
        :param string: x, y, z to put in the QInputDialog box title
        :return:
        str, user input
        '''
        # Need to make the dialog window larger to show the title
        func, okPressed = QtWidgets.QInputDialog.getText(self, "Step 3: Input the function",
                                                         string+" component:",
                                                         QtWidgets.QLineEdit.Normal, "")
        if okPressed and func != '':
            return func

    def onClickOK(self):
        '''
        Trigger step 2: User puts in values of the variables to evaluate
        Trigger step 3: User puts in the function to evaluate and differentiate (working on this now)
        Trigger step 4: User confirms the input is correct and chooses whether to show the function value
        :return:
        None
        '''
        self.SetDimInput()
        # step 2
        self.val = self.PointEval()
        # step 3
        self.func = self.FuncEval()
        print(f"Evaluating {self.func} at {self.val}")  # for testing only, to be commented out in the future
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
        self.importUI()
        self.setupUi(self)

        self.DisVal = False
        # populate the boxes based on user input in step 2 and 3
        self.SetupValFunc()
        # checkBox to select whether the user wants to show the function values
        self.checkBox.clicked.connect(self.setDisVal)
        # click OK button to start the computation
        self.OKButton.clicked.connect(self.onClickOK)

    def importUI(self):
        '''
        Import the .ui based on user input dimensions
        :return:
        None
        '''
        if self.InputDim == 1 and self.FuncDim == 1:
            Ui_FourthDiag, QtBaseClass4 = uic.loadUiType('SmartDiff/GUI/step4.ui')
            Ui_FourthDiag.__init__(self)
        else:
            raise NotImplementedError

    def SetupValFunc(self):
        if self.InputDim == 1 and self.FuncDim == 1:
            self.xVal_1.setText(str(self.val[0]))
            self.xVal_2.setText("N/A")
            self.xVal_3.setText("N/A")
            self.FuncInput_1.setText(self.func[0])
            self.FuncInput_2.setText("N/A")
            self.FuncInput_3.setText("N/A")
        else:
            raise NotImplementedError

    def setDisVal(self):
        self.DisVal = not self.DisVal

    def onClickOK(self):
        '''
        Format the user input in step 2 and 3 into strings that can be put into AD modules in solvers
        :return:
        '''
        # testing out the parser and the AD modules
        formatter = PyExpression_Formatter()
        print(f"parser output {formatter.format_to_pyexpr(str(self.val[0]))}")
        print(f"parser output {formatter.format_to_pyexpr(self.func[0])}")
        x = AD(eval(formatter.format_to_pyexpr(str(self.val[0]))))
        f = formatter.format_to_pyexpr(x)
        if isinstance(f, AD):
            print("the resulting function is an AD object")
        else:
            print("Nope")
        if self.DisVal:
            print(f.val, f.der)
        else:
            print(f.der)

#
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
    # is there a way to make the .ui loaded not global but accessible in classes?
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
