import sys
# from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QToolBar, QAction, QStatusBar, QCheckBox
# from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import numpy as np
from decimal import Decimal

global Ui_MainWindow, Ui_SecondDiag
Ui_MainWindow, QtBaseClass = uic.loadUiType('GUI/step1.ui')
# Ui_SecondDiag, QtBaseClass2 = uic.loadUiType('GUI/step2_1.ui')

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.FuncDim = 1  # default
        self.InputDim = 1  # default
        self.val = np.zeros(1)
        self.func = None

        # OK button
        self.OKButton.clicked.connect(self.onClickOK)

    def SetDimInput(self):
        self.FuncDim = int(self.FuncDimBox.currentText())
        self.InputDim = int(self.InputDimBox.currentText())

    def PointEval(self, qle):
        if self.InputDim == 1:
            num = self._PointEval(qle, "x")
            return np.array([num])
        elif self.InputDim > 1:
            raise NotImplementedError


    def _PointEval(self, qle, string):
        # Need to make the dialog window larger to show the title
        num, okPressed = QtWidgets.QInputDialog.getDouble(self, "Step 2: Input the evaluating point", string+" value:",
                                                           0, -100, 100, 4)
        if okPressed and num != '':
            return num

    def onClickOK(self):
        self.SetDimInput()
        # self.UISetupStep2()
        # dlg2 = SecondDiag(self.InputDim)
        # self.SecondDiag()
        # dlg2.exec_()
        var = QtWidgets.QLineEdit()
        self.val = self.PointEval(var)
        # self.func = self.FuncEval()

    def onClickPrev(self):
        self.proceed = False


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
