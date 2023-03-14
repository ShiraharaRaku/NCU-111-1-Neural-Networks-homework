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
        self.show_file_path = QtWidgets.QTextEdit(self.centralwidget)
        self.show_file_path.setGeometry(QtCore.QRect(270, 400, 201, 40))
        self.show_file_path.setObjectName("show_file_path")
        self.current_dataset_label = QtWidgets.QLabel(self.centralwidget)
        self.current_dataset_label.setGeometry(QtCore.QRect(100, 400, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.current_dataset_label.setFont(font)
        self.current_dataset_label.setObjectName("current_dataset_label")
        self.choose_dataset_abel = QtWidgets.QLabel(self.centralwidget)
        self.choose_dataset_abel.setGeometry(QtCore.QRect(100, 470, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.choose_dataset_abel.setFont(font)
        self.choose_dataset_abel.setObjectName("choose_dataset_abel")
        self.basic_button = QtWidgets.QPushButton(self.centralwidget)
        self.basic_button.setGeometry(QtCore.QRect(240, 470, 100, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.basic_button.setFont(font)
        self.basic_button.setObjectName("basic_button")
        self.bonus_button = QtWidgets.QPushButton(self.centralwidget)
        self.bonus_button.setGeometry(QtCore.QRect(370, 470, 100, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.bonus_button.setFont(font)
        self.bonus_button.setObjectName("bonus_button")
        self.train_label = QtWidgets.QLabel(self.centralwidget)
        self.train_label.setGeometry(QtCore.QRect(100, 610, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.train_label.setFont(font)
        self.train_label.setObjectName("train_label")
        self.train_button = QtWidgets.QPushButton(self.centralwidget)
        self.train_button.setGeometry(QtCore.QRect(220, 610, 101, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.train_button.setFont(font)
        self.train_button.setObjectName("train_button")
        self.choose_image_label = QtWidgets.QLabel(self.centralwidget)
        self.choose_image_label.setGeometry(QtCore.QRect(660, 400, 341, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.choose_image_label.setFont(font)
        self.choose_image_label.setObjectName("choose_image_label")
        self.random_label = QtWidgets.QLabel(self.centralwidget)
        self.random_label.setGeometry(QtCore.QRect(100, 540, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.random_label.setFont(font)
        self.random_label.setObjectName("random_label")
        self.image_input_label = QtWidgets.QLabel(self.centralwidget)
        self.image_input_label.setGeometry(QtCore.QRect(150, 30, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.image_input_label.setFont(font)
        self.image_input_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_input_label.setObjectName("image_input_label")
        self.image_output_label = QtWidgets.QLabel(self.centralwidget)
        self.image_output_label.setGeometry(QtCore.QRect(440, 30, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.image_output_label.setFont(font)
        self.image_output_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_output_label.setObjectName("image_output_label")
        self.random_rate_spinbox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.random_rate_spinbox.setGeometry(QtCore.QRect(280, 540, 101, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.random_rate_spinbox.setFont(font)
        self.random_rate_spinbox.setMaximum(1.0)
        self.random_rate_spinbox.setSingleStep(0.01)
        self.random_rate_spinbox.setObjectName("random_rate_spinbox")
        self.associate_label = QtWidgets.QLabel(self.centralwidget)
        self.associate_label.setGeometry(QtCore.QRect(660, 470, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.associate_label.setFont(font)
        self.associate_label.setObjectName("associate_label")
        self.associate_button = QtWidgets.QPushButton(self.centralwidget)
        self.associate_button.setGeometry(QtCore.QRect(780, 470, 101, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.associate_button.setFont(font)
        self.associate_button.setObjectName("associate_button")
        self.image_input_widget = MplInput(self.centralwidget)
        self.image_input_widget.setGeometry(QtCore.QRect(80, 90, 270, 270))
        self.image_input_widget.setObjectName("image_input_widget")
        self.image_output_widget = MplOutput(self.centralwidget)
        self.image_output_widget.setGeometry(QtCore.QRect(410, 90, 270, 270))
        self.image_output_widget.setObjectName("image_output_widget")
        self.choose_image_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.choose_image_spinbox.setGeometry(QtCore.QRect(730, 400, 51, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        self.choose_image_spinbox.setFont(font)
        self.choose_image_spinbox.setMinimum(1)
        self.choose_image_spinbox.setObjectName("choose_image_spinbox")
        self.epoch_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.epoch_spinbox.setGeometry(QtCore.QRect(770, 540, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        self.epoch_spinbox.setFont(font)
        self.epoch_spinbox.setMinimum(1)
        self.epoch_spinbox.setMaximum(10000)
        self.epoch_spinbox.setObjectName("epoch_spinbox")
        self.epoch_label = QtWidgets.QLabel(self.centralwidget)
        self.epoch_label.setGeometry(QtCore.QRect(660, 540, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.epoch_label.setFont(font)
        self.epoch_label.setObjectName("epoch_label")
        self.image_answer_widget = MplAnswer(self.centralwidget)
        self.image_answer_widget.setGeometry(QtCore.QRect(740, 90, 270, 270))
        self.image_answer_widget.setObjectName("image_answer_widget")
        self.image_answer_label = QtWidgets.QLabel(self.centralwidget)
        self.image_answer_label.setGeometry(QtCore.QRect(830, 30, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.image_answer_label.setFont(font)
        self.image_answer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_answer_label.setObjectName("image_answer_label")
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(780, 610, 101, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.exit_button.setFont(font)
        self.exit_button.setObjectName("exit_button")
        self.exit_label = QtWidgets.QLabel(self.centralwidget)
        self.exit_label.setGeometry(QtCore.QRect(660, 610, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.exit_label.setFont(font)
        self.exit_label.setObjectName("exit_label")
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
        self.current_dataset_label.setText(_translate("MainWindow", "目前選擇資料集："))
        self.choose_dataset_abel.setText(_translate("MainWindow", "選擇資料集："))
        self.basic_button.setText(_translate("MainWindow", "Basic"))
        self.bonus_button.setText(_translate("MainWindow", "Bonus"))
        self.train_label.setText(_translate("MainWindow", "開始訓練："))
        self.train_button.setText(_translate("MainWindow", "train"))
        self.choose_image_label.setText(_translate("MainWindow", "選擇第　　　　張圖片"))
        self.random_label.setText(_translate("MainWindow", "訓練資料雜訊機率："))
        self.image_input_label.setText(_translate("MainWindow", "輸入圖片"))
        self.image_output_label.setText(_translate("MainWindow", "聯想結果"))
        self.associate_label.setText(_translate("MainWindow", "開始聯想："))
        self.associate_button.setText(_translate("MainWindow", "associate"))
        self.epoch_label.setText(_translate("MainWindow", "迭代次數："))
        self.image_answer_label.setText(_translate("MainWindow", "對應圖片"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.exit_label.setText(_translate("MainWindow", "關閉程式："))
from mplwidget import MplAnswer, MplInput, MplOutput