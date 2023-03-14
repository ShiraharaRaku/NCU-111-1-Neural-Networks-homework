#-*-coding:utf-8-*-
from PyQt5 import QtWidgets, uic
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")  # 聲明使用QT5

from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.patches import Rectangle, Circle
from PyQt5.QtCore import QTimer

import sys
import copy

from UI import Ui_MainWindow


class MatplotlibWidget(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        self.trainfile = ""
        self.testfile = ""

        self.train_data = np.empty((0, 0, 0), int)
        self.test_data = np.empty((0, 0, 0), int)
        self.ans_data = np.empty((0, 0, 0), int)
        self.number = 0
        self.width = 0
        self.height = 0

        self.traincheck = 0

        self.choose_index = 0
        self.epoch = 1
        self.random_rate = 0.0


    def setup_control(self):  # botton 連接加在這裡
        self.ui.basic_button.clicked.connect(self.open_basic)
        self.ui.bonus_button.clicked.connect(self.open_bonus)
        self.ui.train_button.clicked.connect(self.train)
        self.ui.associate_button.clicked.connect(self.associate)
        self.ui.random_rate_spinbox.valueChanged.connect(self.random_change)
        self.ui.choose_image_spinbox.valueChanged.connect(self.show_input)
        self.ui.exit_button.clicked.connect(self.exit)

    def open_basic(self):
        self.traincheck = 0
        self.trainfile = ("Hopfield_dataset/Basic_Training.txt")
        self.testfile = ("Hopfield_dataset/Basic_Testing.txt")  # start path
        self.number = 3
        self.width = 9
        self.height = 12

        self.ui.show_file_path.setText("Basic")
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_file_path.setFont(font)

        self.load_train()
        self.load_test()
        self.show_input()

    def open_bonus(self):
        self.traincheck = 0
        self.trainfile = ("Hopfield_dataset/Bonus_Training.txt")
        self.testfile = ("Hopfield_dataset/Bonus_Testing.txt")  # start path
        self.number = 15
        self.width = 10
        self.height = 10

        self.ui.show_file_path.setText("Bonus")
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_file_path.setFont(font)

        self.load_train()
        self.load_test()
        self.show_input()

    def check_data(self):
        if self.trainfile == "":
            self.ui.show_file_path.setText("請選擇資料集!!")
            font = QtGui.QFont()
            font.setFamily("Adobe 繁黑體 Std B")
            font.setPointSize(14)
            font.setBold(True)
            font.setWeight(75)
            self.ui.show_file_path.setFont(font)
            return False
        else:
            return True

    def load_train(self):
        self.random_rate = self.ui.random_rate_spinbox.value() * 100
        self.train_data = np.full((self.number, self.height, self.width), -1, dtype=np.int)
        r = np.random.randint(100, size=(self.number, self.height, self.width))

        nimage = 0
        with open(self.trainfile) as f:
            lines = f.readlines()
            nline = 0

            for line in lines:
                nchar = 0
                for c in line:
                    if c == '1':
                        if r[nimage, nline, nchar] >= self.random_rate:
                            self.train_data[nimage, nline, nchar] = 1
                        else:
                            self.train_data[nimage, nline, nchar] = -1

                    elif c == ' ':
                        if r[nimage, nline, nchar] < self.random_rate:
                            self.train_data[nimage, nline, nchar] = 1

                    elif c == '\n':
                        if nchar == 0:
                            nline = -1
                            nimage += 1

                    nchar += 1
                nline += 1




    def load_test(self):

        self.test_data = np.full((self.number, self.height, self.width), -1, dtype=np.int)

        nimage = 0
        with open(self.testfile) as f:
            lines = f.readlines()
            nline = 0

            for line in lines:
                nchar = 0
                for c in line:
                    if c == '1':
                        self.test_data[nimage, nline, nchar] = 1
                    elif c == '\n':
                        if nchar == 0:
                            nline = -1
                            nimage += 1

                    nchar += 1
                nline += 1


    def train(self):
        if self.check_data():
            self.traincheck = 1
            p = self.width * self.height
            self.w = np.zeros([p, p])

            for n in range(self.number):
                x = self.train_data[n]
                x = x.reshape(p, -1)
                self.w += np.dot(x, x.transpose())

            self.w = (self.w - self.number*np.identity(p)) / p
            self.theta = np.zeros(p)


            for i in range(p):
                for j in range(p):
                    self.theta[i] += self.w[i][j]


    def random_change(self):
        self.traincheck = 0
        if self.check_data():
            self.load_train()
            self.show_input()



    def show_input(self):

        if self.check_data():
            self.choose_index = self.ui.choose_image_spinbox.value()-1
            if self.choose_index < self.number:
                self.ui.image_input_widget.canvas.ax.imshow(self.test_data[self.choose_index], cmap='Greys', interpolation='none')
                self.ui.image_input_widget.canvas.draw()
                self.ui.image_input_label.setText("輸入圖片")
                font = QtGui.QFont()
                font.setFamily("Adobe 繁黑體 Std B")
                font.setPointSize(14)
                font.setBold(True)
                font.setWeight(75)
                self.ui.image_input_label.setFont(font)

                self.ui.image_output_widget.canvas.ax.cla()
                self.ui.image_output_widget.canvas.ax.axis('off')
                self.ui.image_output_widget.canvas.draw()


                self.ui.image_answer_widget.canvas.ax.imshow(self.train_data[self.choose_index], cmap='Greys',
                                                            interpolation='none')
                self.ui.image_answer_widget.canvas.draw()
                self.ui.image_answer_label.setText("對應圖片")
                font = QtGui.QFont()
                font.setFamily("Adobe 繁黑體 Std B")
                font.setPointSize(14)
                font.setBold(True)
                font.setWeight(75)
                self.ui.image_answer_label.setFont(font)


            else:
                self.ui.image_input_widget.canvas.ax.cla()
                self.ui.image_input_widget.canvas.ax.axis('off')
                self.ui.image_input_widget.canvas.draw()
                self.ui.image_input_label.setText("圖片不存在！")
                font = QtGui.QFont()
                font.setFamily("Adobe 繁黑體 Std B")
                font.setPointSize(14)
                font.setBold(True)
                font.setWeight(75)
                self.ui.image_input_label.setFont(font)

                self.ui.image_output_widget.canvas.ax.cla()
                self.ui.image_output_widget.canvas.ax.axis('off')
                self.ui.image_output_widget.canvas.draw()
                self.ui.image_output_label.setText("")
                font = QtGui.QFont()
                font.setFamily("Adobe 繁黑體 Std B")
                font.setPointSize(14)
                font.setBold(True)
                font.setWeight(75)
                self.ui.image_output_label.setFont(font)

                self.ui.image_answer_widget.canvas.ax.cla()
                self.ui.image_answer_widget.canvas.ax.axis('off')
                self.ui.image_answer_widget.canvas.draw()
                self.ui.image_answer_label.setText("")
                font = QtGui.QFont()
                font.setFamily("Adobe 繁黑體 Std B")
                font.setPointSize(14)
                font.setBold(True)
                font.setWeight(75)
                self.ui.image_answer_label.setFont(font)

    def associate(self):
        self.epoch = self.ui.epoch_spinbox.value()
        if self.check_data() and self.choose_index < self.number and self.traincheck == 1:
            p = self.width * self.height

            self.ans_data = copy.deepcopy(self.test_data[self.choose_index])
            x = self.ans_data.reshape(p)

            for e in range(self.epoch):
                update = np.random.randint(p)  # 隨機改
                value = 0

                for i in range(p):
                    value += self.w[update][i] * x[i]

                value -= self.theta[update]
                if value > 0:
                    x[update] = 1
                elif value < 0:
                    x[update] = -1

            self.ans_data = x.reshape(self.height, self.width)

            self.ui.image_output_widget.canvas.ax.imshow(self.ans_data, cmap='Greys',
                                                         interpolation='none')
            self.ui.image_output_widget.canvas.draw()

            self.ui.image_output_label.setText("聯想結果")
            font = QtGui.QFont()
            font.setFamily("Adobe 繁黑體 Std B")
            font.setPointSize(14)
            font.setBold(True)
            font.setWeight(75)
            self.ui.image_output_label.setFont(font)

        elif self.check_data() and self.choose_index < self.number and self.traincheck == 0:
            self.ui.image_output_label.setText("請先按下開始訓練")
            font = QtGui.QFont()
            font.setFamily("Adobe 繁黑體 Std B")
            font.setPointSize(14)
            font.setBold(True)
            font.setWeight(75)
            self.ui.image_output_label.setFont(font)

    def exit(self):
        app.quit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    p = MatplotlibWidget()
    p.show()
    sys.exit(app.exec())
