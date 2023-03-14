# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.training_button = QtWidgets.QPushButton(self.centralwidget)
        self.training_button.setGeometry(QtCore.QRect(100, 600, 100, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.training_button.setFont(font)
        self.training_button.setObjectName("training_button")
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(880, 600, 100, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.exit_button.setFont(font)
        self.exit_button.setObjectName("exit_button")
        self.dataset_button = QtWidgets.QPushButton(self.centralwidget)
        self.dataset_button.setGeometry(QtCore.QRect(60, 390, 100, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.dataset_button.setFont(font)
        self.dataset_button.setObjectName("dataset_button")
        self.show_file_path = QtWidgets.QTextEdit(self.centralwidget)
        self.show_file_path.setGeometry(QtCore.QRect(200, 390, 821, 40))
        self.show_file_path.setObjectName("show_file_path")
        self.epoch_label = QtWidgets.QLabel(self.centralwidget)
        self.epoch_label.setGeometry(QtCore.QRect(70, 460, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.epoch_label.setFont(font)
        self.epoch_label.setObjectName("epoch_label")
        self.learning_rate_label = QtWidgets.QLabel(self.centralwidget)
        self.learning_rate_label.setGeometry(QtCore.QRect(70, 520, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.learning_rate_label.setFont(font)
        self.learning_rate_label.setObjectName("learning_rate_label")
        self.learning_rate_spinbox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.learning_rate_spinbox.setGeometry(QtCore.QRect(200, 520, 90, 30))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.learning_rate_spinbox.setFont(font)
        self.learning_rate_spinbox.setSingleStep(0.01)
        self.learning_rate_spinbox.setObjectName("learning_rate_spinbox")
        self.epoch_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.epoch_spinbox.setGeometry(QtCore.QRect(200, 460, 90, 30))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.epoch_spinbox.setFont(font)
        self.epoch_spinbox.setMaximum(10000)
        self.epoch_spinbox.setObjectName("epoch_spinbox")
        self.weight_label = QtWidgets.QLabel(self.centralwidget)
        self.weight_label.setGeometry(QtCore.QRect(770, 460, 120, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.weight_label.setFont(font)
        self.weight_label.setObjectName("weight_label")
        self.show_train_accuracy = QtWidgets.QTextEdit(self.centralwidget)
        self.show_train_accuracy.setGeometry(QtCore.QRect(480, 460, 240, 40))
        self.show_train_accuracy.setObjectName("show_train_accuracy")
        self.show_test_accuracy = QtWidgets.QTextEdit(self.centralwidget)
        self.show_test_accuracy.setGeometry(QtCore.QRect(480, 520, 240, 40))
        self.show_test_accuracy.setObjectName("show_test_accuracy")
        self.show_weight = QtWidgets.QTextEdit(self.centralwidget)
        self.show_weight.setGeometry(QtCore.QRect(860, 460, 160, 40))
        self.show_weight.setObjectName("show_weight")
        self.widget = MplWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(50, -1, 981, 320))
        self.widget.setObjectName("widget")
        self.epoch_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.epoch_label_2.setGeometry(QtCore.QRect(350, 460, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.epoch_label_2.setFont(font)
        self.epoch_label_2.setObjectName("epoch_label_2")
        self.test_accuracy_label = QtWidgets.QLabel(self.centralwidget)
        self.test_accuracy_label.setGeometry(QtCore.QRect(350, 520, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.test_accuracy_label.setFont(font)
        self.test_accuracy_label.setObjectName("test_accuracy_label")
        self.weight_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.weight_label_2.setGeometry(QtCore.QRect(770, 520, 120, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.weight_label_2.setFont(font)
        self.weight_label_2.setObjectName("weight_label_2")
        self.show_bias = QtWidgets.QTextEdit(self.centralwidget)
        self.show_bias.setGeometry(QtCore.QRect(860, 520, 160, 40))
        self.show_bias.setObjectName("show_bias")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.training_button.setText(_translate("MainWindow", "Training"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.dataset_button.setText(_translate("MainWindow", "Dataset"))
        self.epoch_label.setText(_translate("MainWindow", "Epoch:"))
        self.learning_rate_label.setText(_translate("MainWindow", "Learning rate:"))
        self.weight_label.setText(_translate("MainWindow", "Weight:"))
        self.epoch_label_2.setText(_translate("MainWindow", "Train accuracy:"))
        self.test_accuracy_label.setText(_translate("MainWindow", "Test accuracy:"))
        self.weight_label_2.setText(_translate("MainWindow", "Bias:"))
from mplwidget import MplWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())