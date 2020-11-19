import sys
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import numpy as np

global Ui_MainWindow, Ui_SecondDiag
Ui_MainWindow, QtBaseClass = uic.loadUiType('GUI/step1.ui') # .ui drawn in Qt Designer
# Ui_SecondDiag, QtBaseClass2 = uic.loadUiType('GUI/step2_1.ui')

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setWindowTitle("SmartDiff")
        self.setupUi(self)

        self.FuncDim = 1  # default
        self.InputDim = 1  # default
        self.val = np.zeros(1)
        self.func = None

        # once OK button is pressed, go to step 2 and 3
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
            num = self._PointEval("x")
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
        Trigger step 2: User put in values of the variables to evaluate
        Trigger step 3: User put in the function to evaluate and differentiate (working on this now)
        :return:
        None
        '''
        self.SetDimInput()
        # step 2
        self.val = self.PointEval()
        # step 3
        self.func = self.FuncEval()
        # self.UISetupStep2()
        # dlg2 = SecondDiag(self.InputDim)
        # self.SecondDiag()
        # dlg2.exec_()
        # self.func = self.FuncEval()



# class ThirdDiag(QtWidgets.QInputDialog, Ui_SecondDiag):
#
#     def __init__(self, InputDim):
#         QtWidgets.QInputDialog.__init__(self)
#         # load a dialogue based on user input from step one
#         self.InputDim = InputDim
#         Ui_SecondDiag.__init__(self)
#         self.setupUi(self)
#
#         # initialize output values
#         self.val_vec = np.zeros(self.InputDim)
#         self.proceed = True
#
#         # input points to evaluate
#         self.PointEval()
#
#         # Prev button
#         self.prevButton.clicked.connect(self.onClickPrev)  # need to implement this
#
#         # OK button
#         self.OKButton.clicked.connect(self.onClickOK)
#
#     def PointEval(self):
#         if self.InputDim == 1:
#             txt = self.PointEval1(self.xVal)
#
#         else:  # to add after putting in more input dimensions
#             raise NotImplementedError
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
#
#     def onClickPrev(self):
#         self.proceed = False
#
#     def onClickOK(self):
#         pass



if __name__ == "__main__":
    # there are some issues here with the ui classes, is there a way to make it not global but accessible in classes?
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
