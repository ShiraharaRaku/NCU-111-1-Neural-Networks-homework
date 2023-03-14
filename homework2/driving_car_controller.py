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


from UI import Ui_MainWindow


def GeneralEquation(p1, p2):
    # 一般式 Ax+By+C=0
    A = p2[1] - p1[1]  # y 相減
    B = p1[0] - p2[0]  # x 相減
    C = p2[0] * p1[1] - p1[0] * p2[1]
    return A, B, C


def GetIntersectPointofLines(A1, B1, C1, A2, B2, C2):
    m = A1 * B2 - A2 * B1
    if m != 0:
        x = (C2 * B1 - C1 * B2) / m
        y = (C1 * A2 - C2 * A1) / m
        cross_point = np.array([x, y], dtype="float64")
        return cross_point
    else:
        return np.array([np.infty, np.infty])



class MatplotlibWidget(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        self.filename = ""
        self.points = np.empty([0, 2], float)
        self.pred = np.array([], int)
        self.epoch = 100
        self.learning_rate = 0.01
        self.k = 7

        self.terminal_points = np.array([], int)  # left, upper, right, lower
        self.border_points_xy = np.empty([0, 2], float)
        self.border_points_GeneralEquation = np.empty([0, 3], float)

        self.car_degree = 90
        self.car_length = 6
        self.car_point = np.array([0, 0], float)
        self.car_wheel = 0

        self.move_track = np.array([0, 0, 90], float)  # x, y, degree
        self.l_cross_track = np.empty([0, 2], float)
        self.f_cross_track = np.empty([0, 2], float)
        self.r_cross_track = np.empty([0, 2], float)
        self.move_distance_4d = np.empty([0, 4], float)
        self.move_distance_6d = np.empty([0, 6], float)

        self.data_4d = np.empty([0, 3], float)
        self.wheel_4d = np.array([], float)
        self.kpoint_4d = np.empty([0, 3], float)

        self.data_6d = np.empty([0, 5], float)
        self.wheel_6d = np.array([], float)
        self.kpoint_6d = np.empty([0, 5], float)

        self.w = 0
        self.sigma = np.array([], float)
        self.phi_basis = np.array([0, 0], float)

        self.stopflag = 0

        self.mytimer = QTimer(self)
        self.mytimer.timeout.connect(self.car_move)

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.play_animation)
        self.animation_count = 0

        self.open_borderfile()
        self.plot_border()
        self.plot_car(self.car_point, self.car_degree)





    def reset(self):
        self.mytimer.stop()
        self.animation_timer.stop()
        self.animation_count = 0

        self.move_track = np.empty([0, 3], float)
        self.move_track = np.append(self.move_track, np.array([[0, 0, 90]], dtype='float64'), axis=0)
        self.l_cross_track = np.empty([0, 2], float)
        self.f_cross_track = np.empty([0, 2], float)
        self.r_cross_track = np.empty([0, 2], float)

        self.phi_basis = np.array([0, 0], float)
        self.car_degree = 90
        self.car_length = 6
        self.car_point = np.array([0, 0], float)
        self.car_wheel = 0
        self.stopflag = 0



    def setup_control(self):  # botton 連接加在這裡
        self.ui.train4dAll_button.clicked.connect(self.open_train4dAll)
        self.ui.train6dAll_button.clicked.connect(self.open_train6dAll)
        self.ui.train_button.clicked.connect(self.train_control)
        self.ui.play_button.clicked.connect(self.animation_botton)
        self.ui.exit_button.clicked.connect(self.exit)

    def open_train4dAll(self):
        self.filename = ("train4dAll.txt")  # start path
        self.ui.show_dataset.setText("train4dAll")
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_dataset.setFont(font)

    def open_train6dAll(self):
        self.filename = ("train6dAll.txt")  # start path
        self.ui.show_dataset.setText("train6dAll")
        font = QtGui.QFont()
        font.setFamily("Adobe 繁黑體 Std B")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.ui.show_dataset.setFont(font)


    def open_borderfile(self):
        self.borderfilename = "軌道座標點.txt"
        with open(self.borderfilename) as f:
            line = f.readline()
            car_x, car_y, self.car_degree = line.split(",")
            self.car_degree = float(self.car_degree)
            self.car_point = np.array([car_x, car_y], dtype="float64")
            line = f.readline()
            left_x, upper_y = line.split(",")
            line = f.readline()
            right_x, lower_y = line.split(",")

            self.terminal_points = np.append(self.terminal_points, np.array([left_x], dtype="float64"), axis=0)
            self.terminal_points = np.append(self.terminal_points, np.array([upper_y], dtype="float64"), axis=0)
            self.terminal_points = np.append(self.terminal_points, np.array([right_x], dtype="float64"), axis=0)
            self.terminal_points = np.append(self.terminal_points, np.array([lower_y], dtype="float64"), axis=0)

            lines = f.readlines()
            for line in lines:
                x, y = line.split(",")
                self.border_points_xy = np.append(self.border_points_xy, np.array([[x, y]], dtype="float64"), axis=0)

        for i in range(len(self.border_points_xy)-1):

            a, b, c = GeneralEquation(self.border_points_xy[i], self.border_points_xy[i+1])
            self.border_points_GeneralEquation = np.append(self.border_points_GeneralEquation,
                                                          np.array([[a, b, c]], dtype="float64"), axis=0)

    def plot_border(self):
        self.ui.widget.canvas.ax.axis(xmin=-20, xmax=45)  # 設定x軸顯示範圍
        self.ui.widget.canvas.ax.axis(ymin=-10, ymax=55)  # 設定y軸顯示範圍
        self.ui.widget.canvas.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        self.ui.widget.canvas.ax.yaxis.set_major_locator(plt.MultipleLocator(10))

        self.ui.widget.canvas.ax.plot([-6, 6], [0, 0])
        self.ui.widget.canvas.ax.plot([10, 10], [40, 40], color='#5c3d85')

        for i in range(len(self.border_points_xy)-1):

            p1x = self.border_points_xy[i+1][0]
            p2x = self.border_points_xy[i][0]
            p1y = self.border_points_xy[i+1][1]
            p2y = self.border_points_xy[i][1]

            self.ui.widget.canvas.ax.plot([p1x, p2x],  # x 方向
                                          [p1y, p2y],  # y 方向
                                          color='#5c3d85')
        left_x = self.terminal_points[0]
        lower_y = self.terminal_points[3]
        right_x = self.terminal_points[2]
        upper_y = self.terminal_points[1]
        rect = Rectangle((int(left_x), int(lower_y)), int(right_x) - int(left_x), int(upper_y) - int(lower_y),
                         color='#bba3ff')
        self.ui.widget.canvas.ax.add_patch(rect)

    def plot_car(self, car_point1, car_degree):
        # 畫車
        radian_degree = np.array(np.radians(car_degree), dtype="float64")  # 轉成弧度以計算三角函數
        car_point2 = np.array([car_point1[0] + 6 * round(np.cos(radian_degree), 6),
                               car_point1[1] + 6 * round(np.sin(radian_degree), 6)],
                              dtype="float64")

        self.ui.widget.canvas.ax.plot([car_point1[0], car_point2[0]], [car_point1[1], car_point2[1]],
                                      color='r')
        circle = Circle(xy=car_point1, radius=3, color='#38b2a9', fill=None)
        self.ui.widget.canvas.ax.add_patch(circle)
        self.ui.widget.canvas.draw()

    def sensor_detect(self, direction):
        # 180 - 車子角度 才是與 x 軸的夾角 (+90 為 y 軸正向)

        min_distance = np.Inf
        radian_degree = np.array(np.radians(self.car_degree-direction), dtype="float64")  # 轉成弧度以計算三角函數

        car_point2 = np.array([self.car_point[0]+3*round(np.cos(radian_degree), 6),
                              self.car_point[1]+3*round(np.sin(radian_degree), 6)],
                              dtype="float64")


        a1, b1, c1 = GeneralEquation(self.car_point, car_point2)
        cross_point = [np.Inf, np.Inf]

        for i in range(len(self.border_points_GeneralEquation)):
            temp = GetIntersectPointofLines(a1, b1, c1,
                                            self.border_points_GeneralEquation[i][0],
                                            self.border_points_GeneralEquation[i][1],
                                            self.border_points_GeneralEquation[i][2])

            d1 = np.linalg.norm(temp-self.border_points_xy[i]) + np.linalg.norm(temp-self.border_points_xy[i+1])
            d2 = np.linalg.norm(self.border_points_xy[i]-self.border_points_xy[i+1])
            direct_d1 = np.linalg.norm(temp-car_point2) + np.linalg.norm(car_point2-self.car_point)
            direct_d2 = np.linalg.norm(temp-self.car_point)

            if (temp[0] != np.infty and temp[1] != np.infty) \
                    and min_distance > np.linalg.norm(self.car_point-temp) \
                    and round(d1, 6) == round(d2, 6)\
                    and round(direct_d1, 6) == round(direct_d2, 6):
                # 確定交點在牆壁的線段上
                # 確定交點沒有取到反方向
                # round 是為了避免小數點誤差，導致判斷失誤，所以導致取不到的情況

                cross_point = temp
                min_distance = round(np.linalg.norm(self.car_point-temp), 4)

        return min_distance, cross_point

    def move_equation(self):
        # ϕ(t) 是模型車與水平軸的角度 (self.car_degree)
        # b 是模型車的長度 (self.car_length)
        # x 與 y 是模型車的座標位置 (self.car_point)
        # θ(t) 是模型車方向盤所打的角度 (self.car_wheel)

        car_degree = np.radians(self.car_degree)
        car_wheel = np.radians(self.car_wheel)

        x = self.car_point[0] \
            + np.cos(car_degree + car_wheel) \
            + np.sin(car_wheel) * np.sin(car_degree)
        y = self.car_point[1] \
            + np.sin(car_degree + car_wheel) \
            - np.sin(car_wheel) * np.cos(car_degree)

        new_degree = self.car_degree - np.degrees(np.arcsin(2*np.sin(np.radians(self.car_wheel))/self.car_length))
        new_degree = new_degree % 360

        self.car_point = np.array([x, y], dtype="float64")

        if new_degree > 270:
            self.car_degree = new_degree - 360
        else:
            self.car_degree = new_degree

        self.move_track = np.append(self.move_track, np.array([[x, y, self.car_degree]], dtype="float64"), axis=0)

    def animation_botton(self):
        if self.filename != "":
            self.animation_timer.stop()
            self.animation_count = 0
            self.plot_border()
            self.animation_timer.start(100)


    def play_animation(self):
        count = self.animation_count
        self.ui.widget.canvas.ax.cla()
        for i in range(count):
            self.ui.widget.canvas.ax.scatter(self.move_track[i][0], self.move_track[i][1], s=5, c='#6376c5')

        point = [self.move_track[count][0], self.move_track[count][1]]
        degree = self.move_track[count][2]

        if count < len(self.move_track) - 1:
            self.ui.widget.canvas.ax.plot([point[0], self.l_cross_track[count][0]], [point[1], self.l_cross_track[count][1]],
                                          color='r', linestyle='--')
            self.ui.widget.canvas.ax.plot([point[0], self.f_cross_track[count][0]], [point[1], self.f_cross_track[count][1]],
                                          color='r', linestyle='--')
            self.ui.widget.canvas.ax.plot([point[0], self.r_cross_track[count][0]], [point[1], self.r_cross_track[count][1]],
                                          color='r', linestyle='--')

        self.plot_border()

        self.plot_car(point, degree)

        if count < len(self.move_track)-1:
            if self.filename == "train4dAll.txt":
                self.ui.sensor_show_label.setText(str(self.move_distance_4d[count][0]) + "              "  # 14 個空白
                                                  + str(self.move_distance_4d[count][1]) + "              "
                                                  + str(self.move_distance_4d[count][2]))
            elif self.filename == "train6dAll.txt":
                self.ui.sensor_show_label.setText(str(self.move_distance_6d[count][0]) + "              "  # 14 個空白
                                                  + str(self.move_distance_6d[count][1]) + "              "
                                                  + str(self.move_distance_6d[count][2]))

            font = QtGui.QFont()
            font.setFamily("Adobe 繁黑體 Std B")
            font.setPointSize(14)
            font.setBold(True)
            font.setWeight(75)
            self.ui.sensor_show_label.setFont(font)
            self.animation_count += 1

        else:
            self.animation_timer.stop()


    def car_move(self):
        if self.stopflag == 1:
            self.mytimer.stop()
            self.writefile()
        else:
            left_distance, l_cross = self.sensor_detect(-45)
            front_distance, f_cross = self.sensor_detect(0)
            right_distance, r_cross = self.sensor_detect(45)

            self.l_cross_track = np.append(self.l_cross_track, np.array([l_cross]), axis=0)
            self.f_cross_track = np.append(self.f_cross_track, np.array([f_cross]), axis=0)
            self.r_cross_track = np.append(self.r_cross_track, np.array([r_cross]), axis=0)

            if self.filename == "train4dAll.txt":
                distance = [front_distance, left_distance, right_distance]
            elif self.filename == "train6dAll.txt":
                distance = [front_distance, left_distance, right_distance, self.car_point[0], self.car_point[1]]

            self.ui.widget.canvas.ax.cla()

            for i in range(len(self.move_track)):
                self.ui.widget.canvas.ax.scatter(self.move_track[i][0], self.move_track[i][1], s=5, c='#6376c5')

            self.ui.widget.canvas.ax.plot([self.car_point[0], l_cross[0]], [self.car_point[1], l_cross[1]],
                                          color='r', linestyle='--')
            self.ui.widget.canvas.ax.plot([self.car_point[0], f_cross[0]], [self.car_point[1], f_cross[1]],
                                          color='r', linestyle='--')
            self.ui.widget.canvas.ax.plot([self.car_point[0], r_cross[0]], [self.car_point[1], r_cross[1]],
                                          color='r', linestyle='--')

            self.plot_border()
            self.plot_car(self.car_point, self.car_degree)

            change_wheel = self.theta
            for i in range(len(self.center_points)):
                basis = self.Gaussian_basis(distance, self.center_points[i], self.sigma)
                change_wheel += np.sum(self.w[i] * basis, axis=0)

            if change_wheel < 0:
                change_wheel = change_wheel % 40 - 40
            else:
                change_wheel = change_wheel % 40



            self.car_wheel = round(change_wheel, 3)
            self.move_equation()


            self.ui.sensor_show_label.setText("{:.4f}".format(front_distance) + "               "  # 15 個空白
                                              + "{:.4f}".format(left_distance) + "               "
                                              + "{:.4f}".format(right_distance))
            font = QtGui.QFont()
            font.setFamily("Adobe 繁黑體 Std B")
            font.setPointSize(14)
            font.setBold(True)
            font.setWeight(75)
            self.ui.sensor_show_label.setFont(font)





            # 撞牆判定

            for i in range(-180, 180):
                i_dis, i_cross = self.sensor_detect(i)
                if i_dis == np.Inf:
                    self.stopflag = 1
                    self.ui.widget.canvas.ax.cla()

                    self.ui.widget.canvas.ax.set_title("!!!!!!CAR ACCIDENT!!!!!!")
                    for i in range(len(self.move_track)):
                        self.ui.widget.canvas.ax.scatter(self.move_track[i][0], self.move_track[i][1], s=5, c='#6376c5')

                    self.plot_border()
                    self.plot_car(self.car_point, self.car_degree)

                    break


            # 終點判定
            # self.terminal_points : left, upper, right, lower
            radius = self.car_length / 2
            if self.terminal_points[0] - radius < self.car_point[0] < self.terminal_points[2] + radius \
                    and self.terminal_points[3] - radius < self.car_point[1] < self.terminal_points[1] + radius:
                self.stopflag = 1

                self.ui.widget.canvas.ax.cla()

                for i in range(len(self.move_track)):
                    self.ui.widget.canvas.ax.scatter(self.move_track[i][0], self.move_track[i][1], s=5, c='#6376c5')

                self.plot_border()
                self.plot_car(self.car_point, self.car_degree)


            if self.filename == "train4dAll.txt":
                data = np.array([front_distance, left_distance, right_distance, self.car_wheel])
                self.move_distance_4d = np.append(self.move_distance_4d, np.array([data]), axis=0)
            elif self.filename == "train6dAll.txt":
                data = np.array([front_distance, left_distance, right_distance, self.car_point[0], self.car_point[1], self.car_wheel], dtype="float64")
                self.move_distance_6d = np.append(self.move_distance_6d, np.array([data]), axis=0)


    def Gaussian_basis(self, x, m, sigma):
        return np.exp(-(x-m) ** 2 / (2 * sigma ** 2))


    def train_control(self):
        self.reset()
        if self.filename == "train4dAll.txt":
            self.train_4d()
              # find m，得到 self.center_points
        elif self.filename == "train6dAll.txt":
            self.train_6d()
        else:
            self.ui.show_dataset.setText("choose dataset!!!")
            font = QtGui.QFont()
            font.setFamily("Adobe 繁黑體 Std B")
            font.setPointSize(14)
            font.setBold(True)
            font.setWeight(75)
            self.ui.show_dataset.setFont(font)

        if self.filename != "":
            self.k_means() # find m，得到 self.center_points
            # find sigma，基底函數方差
            cmax = 0.0  # 為所選取中心點之間的最大距離
            for i in range(len(self.center_points)):
                x = self.center_points[i]
                for j in range(len(self.center_points)):
                    y = self.center_points[j]
                    if (i != j):
                        temp_distance = np.linalg.norm(x - y)
                        if cmax < temp_distance:
                            cmax = temp_distance

            self.sigma = cmax / np.sqrt(2 * (len(self.center_points) + 1))

            input_basis = np.empty([0, self.k], float)

            self.w = np.random.rand(len(self.center_points))

            if self.filename == "train4dAll.txt":
                output = np.array([], float)
                for i in range(len(self.data_4d)):
                    linear_combination = 0.0

                    for j in range(len(self.center_points)):
                        basis = self.Gaussian_basis(self.data_4d[i], self.center_points[j], self.sigma)
                        input_basis = np.append(input_basis, basis)  # phi j
                        linear_combination += self.w[j] * basis   # output = w j * phi j

                    output = np.append(output, np.sum(linear_combination, axis=0))


                #  train, find w
                self.theta = 0
                for epoch in range(self.epoch):
                    for i in range(len(self.data_4d)):
                        for j in range(len(self.center_points)):
                            Fx = output[j] + self.theta
                            if Fx < 0:
                                Fx = Fx % 40 - 40
                            else:
                                Fx = Fx % 40
                            self.w[j] = self.w[j] - self.learning_rate * (self.wheel_4d[i] - Fx) * input_basis[j]
                            self.theta = self.theta + self.learning_rate * (self.wheel_4d[i] - Fx)

                    for i in range(len(self.data_4d)):
                        linear_combination = 0.0
                        for j in range(len(self.center_points)):
                            basis = self.Gaussian_basis(self.data_4d[i], self.center_points[j], self.sigma)
                            linear_combination = linear_combination + self.w[j] * basis

                        output[i] = np.sum(linear_combination, axis=0) + self.theta





            elif self.filename == "train6dAll.txt":
                output = np.array([], float)
                for i in range(len(self.data_6d)):
                    linear_combination = 0.0

                    for j in range(len(self.center_points)):
                        basis = self.Gaussian_basis(self.data_6d[i], self.center_points[j], self.sigma)
                        input_basis = np.append(input_basis, basis)  # phi j
                        linear_combination += self.w[j] * basis  # output = w j * phi j

                    output = np.append(output, np.sum(linear_combination, axis=0))

                #  train, find w
                self.theta = 0
                for epoch in range(self.epoch):
                    for i in range(len(self.data_6d)):
                        for j in range(len(self.center_points)):
                            Fx = output[j] + self.theta
                            if Fx < 0:
                                Fx = Fx % 40 - 40
                            else:
                                Fx = Fx % 40

                            self.w[j] = self.w[j] - self.learning_rate * (self.wheel_6d[i] - Fx) * input_basis[j]

                            self.theta = self.theta + self.learning_rate * (self.wheel_6d[i] - Fx)


                    for i in range(len(self.data_6d)):
                        linear_combination = 0.0
                        for j in range(len(self.center_points)):
                            basis = self.Gaussian_basis(self.data_6d[i], self.center_points[j], self.sigma)
                            linear_combination = linear_combination + self.w[j] * basis

                        output[i] = np.sum(linear_combination, axis=0) + self.theta


            # 跑車車
            self.car_move()
            self.mytimer.start(100)



    def train_4d(self):
        if self.data_4d.size == 0:
            with open(self.filename) as f:
                lines = f.readlines()
                for line in lines:
                    d_f, d_r, d_l, w_degree = line.split(" ")
                    self.data_4d = np.append(self.data_4d, np.array([[d_f, d_r, d_l]],
                                                                    dtype="float64"), axis=0)
                    self.wheel_4d = np.append(self.wheel_4d, np.array(w_degree, dtype="float64"))


    def train_6d(self):
        if self.data_6d.size == 0:
            with open(self.filename) as f:
                lines = f.readlines()
                for line in lines:
                    x, y, d_f, d_r, d_l, w_degree = line.split(" ")
                    self.data_6d = np.append(self.data_6d, np.array([[x, y, d_f, d_r, d_l]],
                                                                    dtype="float64"), axis=0)
                    self.wheel_6d = np.append(self.wheel_6d, np.array(w_degree, dtype="float64"))


    def k_means(self):
        # 找初始中心點

        if self.filename == "train4dAll.txt":
            m, n = self.data_4d.shape
        elif self.filename == "train6dAll.txt":
            m, n = self.data_6d.shape
        select = np.random.choice(m, 1)  # 隨機取 1 個 index


        # 再取 k-1 個
        for i in range(self.k-1):
            all_distance = np.empty((m, 0))
            min_distance = np.Inf

            for j in select:
                temp_distance = np.array([], float)
                if self.filename == "train4dAll.txt":
                    for point in self.data_4d:
                        temp_distance = np.append(temp_distance, np.linalg.norm(point-self.data_4d[j]))
                elif self.filename == "train6dAll.txt":
                    for point in self.data_6d:
                        temp_distance = np.append(temp_distance, np.linalg.norm(point-self.data_6d[j]))
                temp_distance = temp_distance.reshape(-1, 1)

                all_distance = np.c_[all_distance, temp_distance]

            min_distance = all_distance.min(axis=1).reshape(-1, 1)
            index = np.argmax(min_distance)
            select = np.append(select, index)

        if self.filename == "train4dAll.txt":
            self.center_points = self.data_4d[select]  # find m


        elif self.filename == "train6dAll.txt":
            self.center_points = self.data_6d[select]  # find m


        while True:
            point_class = {}
            for i in range(self.k):
                point_class[i] = np.empty((0, n))

            if self.filename == "train4dAll.txt":
                for i in range(m):
                    min_distance = np.Inf
                    center_class = -1
                    for center in range(len(self.center_points)):
                        if np.linalg.norm(self.data_4d[i] - self.center_points[center]) < min_distance:
                            min_distance = np.linalg.norm(self.data_4d[i] - self.center_points[center])
                            center_class = center
                    point_class[center_class] = np.r_[point_class[center_class], self.data_4d[i].reshape(1, -1)]


            elif self.filename == "train6dAll.txt":
                for i in range(m):
                    min_distance = np.Inf
                    center_class = -1
                    for center in range(len(self.center_points)):
                        if np.linalg.norm(self.data_6d[i] - self.center_points[center]) < min_distance:
                            min_distance = np.linalg.norm(self.data_6d[i] - self.center_points[center])
                            center_class = center
                    point_class[center_class] = np.r_[point_class[center_class], self.data_6d[i].reshape(1, -1)]

            centers = np.empty((0, n))


            for i in range(self.k):
                center = np.mean(point_class[i], axis=0).reshape(1, -1)
                centers = np.r_[centers, center]

            result = np.all(centers == self.center_points)
            if result == True:
                break
            else:
                self.center_points = centers


    def writefile(self):
        # 開始寫檔
        print("write file")
        if self.filename == "train4dAll.txt":
            path = 'track4D.txt'
            with open(path, 'w') as f:
                for i in range(len(self.move_distance_4d)):
                    f.write("{:.4f}".format(self.move_distance_4d[i][0]))
                    f.write(' ')
                    f.write("{:.4f}".format(self.move_distance_4d[i][1]))
                    f.write(' ')
                    f.write("{:.4f}".format(self.move_distance_4d[i][2]))
                    f.write(' ')
                    f.write("{:.3f}".format(self.move_distance_4d[i][3]))
                    f.write('\n')

        elif self.filename == "train6dAll.txt":
            path = 'track6D.txt'
            with open(path, 'w') as f:
                for i in range(len(self.move_distance_6d)):
                    f.write("{:.4f}".format(self.move_distance_6d[i][0]))
                    f.write(' ')
                    f.write("{:.4f}".format(self.move_distance_6d[i][1]))
                    f.write(' ')
                    f.write("{:.4f}".format(self.move_distance_6d[i][2]))
                    f.write(' ')
                    f.write("{:.4f}".format(self.move_distance_6d[i][3]))
                    f.write(' ')
                    f.write("{:.4f}".format(self.move_distance_6d[i][4]))
                    f.write(' ')
                    f.write("{:.3f}".format(self.move_distance_6d[i][5]))
                    f.write('\n')

    def exit(self):
        app.quit()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    p = MatplotlibWidget()
    p.show()
    sys.exit(app.exec())