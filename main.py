import sys
# from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QToolBar, QAction, QStatusBar, QCheckBox
# from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import numpy as np
from decimal import Decimal

# class MainWindow(QMainWindow):
#
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)
#         # self.windowTitleChanged.connect(self.onWindowTitleChange)
#         # self.windowTitleChanged.connect(lambda x: self.my_custom_fn())
#         # self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x))
#         # self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x, 25))
#         self.setWindowTitle("SmartDiff")
#         label = QLabel("An app to calculate the values and derivatives of complicated functions")
#         label.setAlignment(Qt.AlignTop)
#         self.setCentralWidget(label)
#
#         toolbar = QToolBar("Test toolbar")
#         self.addToolBar(toolbar)
#
#         button_action = QAction("0", self)
#         # button_action.setStatusTip("THIS IS THE BUTTON")
#         button_action.triggered.connect(self.onMyToolBarClickButton)
#         button_action.setCheckable(True)
#         toolbar.addAction(button_action)
#
#         toolbar.addSeparator()
#
#         button_action1 = QAction("1", self)
#         # button_action.setStatusTip("THIS IS THE BUTTON")
#         button_action1.triggered.connect(self.onMyToolBarClickButton)
#         button_action1.setCheckable(True)
#         toolbar.addAction(button_action1)
#
#         toolbar.addSeparator()
#
#         button_action2 = QAction("2", self)
#         # button_action.setStatusTip("THIS IS THE BUTTON")
#         button_action2.triggered.connect(self.onMyToolBarClickButton)
#         button_action2.setCheckable(True)
#         toolbar.addAction(button_action2)
#
#         toolbar.addSeparator()
#
#         toolbar.addWidget(QLabel("Show values"))
#         toolbar.addWidget(QCheckBox())
#
#         toolbar.addSeparator()
#
#         toolbar.addWidget(QLabel("Show derivatives"))
#         toolbar.addWidget(QCheckBox())
#
#         self.setStatusBar(QStatusBar(self))
#
#     def onWindowTitleChange(self, s):
#         print(s)
#
#     def my_custom_fn(self, a="Hello!", b=5):
#         print(a, b)
#
#     def onMyToolBarClickButton(self, s): # can have a different value for each digit/operation
#         print("click", s)
#
#     def contextMenuEvent(self, event):
#         print("Context menu event!")
#         super(MainWindow, self).contextMenuEvent(event)
#
# class CustomButton(QPushButton):
#
#     def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
#         super(CustomButton, self).keyPressEvent(a0)

global Ui_MainWindow, Ui_SecondDiag
Ui_MainWindow, QtBaseClass = uic.loadUiType('step1.ui')
Ui_SecondDiag, QtBaseClass2 = uic.loadUiType('step2_1.ui')

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.FuncDim = 1  # default
        self.InputDim = 1  # default
        self.val = np.zeros(1)
        self.proceed = True
        # global Ui_SecondDiag

        # OK button
        self.OKButton.clicked.connect(self.onClickOK)

    def SetDimInput(self):
        self.FuncDim = int(self.FuncDimBox.currentText())
        self.InputDim = int(self.InputDimBox.currentText())

    def UISetupStep2(self):
        if self.FuncDim == 1 and self.InputDim == 1:
            Ui_SecondDiag, QtBaseClass2 = uic.loadUiType('step2_1.ui')  # remember to change the path!!!
        else:
            raise NotImplementedError

    def onClickOK(self):
        self.SetDimInput()
        self.UISetupStep2()
        dlg2 = SecondDiag(self.InputDim)
        dlg2.exec_()

    def onClickPrev(self):
        self.proceed = False


class SecondDiag(QtWidgets.QDialog, Ui_SecondDiag):

    def __init__(self, InputDim):
        QtWidgets.QDialog.__init__(self)
        # load a dialogue based on user input from step one
        self.InputDim = InputDim
        Ui_SecondDiag.__init__(self)
        self.setupUi(self)

        # initialize output values
        self.val_vec = np.zeros(self.InputDim)
        self.proceed = True

        # input points to evaluate
        self.PointEval()

        # Prev button
        self.prevButton.clicked.connect(self.onClickPrev)  # need to implement this

        # OK button
        self.OKButton.clicked.connect(self.onClickOK)

    def PointEval(self):
        if self.InputDim == 1:
            self.PointEval1(self.xVal)
            # print(str(val))
        else:  # to add after putting in more input dimensions
            raise NotImplementedError

    def PointEval1(self, var):
        '''
        User input the number to evaluate
        Note: Use the input validation function to only allow float
        :return:
        None
        '''
        # try:
        #     self.val_vec = np.array([float(self.xVal.text())])
        # except ValueError:
        #     ty = type(self.xVal.text())
        #     self.WarnLabel.setText(f"Warning: You entered {ty}. Please enter a float")
        reg_ex = QtCore.QRegExp("[-+]?[0-9]*\.?[0-9]+")  # regex for float
        input_validator = QtGui.QRegExpValidator(reg_ex)
        var.setValidator(input_validator)
        var.setMaxLength(6)
        var.setAlignment(QtCore.Qt.AlignRight)
        def enterPress():
            print("Enter")
        def textchanged(text):
            print("contents of input box: " + text)
        var.textChanged.connect(textchanged)
        print(var.returnPressed())
        # if var.text() == "":
        #     return 0
        # return var.text()

    def onClickPrev(self):
        self.proceed = False

    def onClickOK(self):
        pass



if __name__ == "__main__":
    # there are some issue here
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
