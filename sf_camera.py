# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sf_camera.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_sfCamera(object):
    def setupUi(self, sfCamera):
        sfCamera.setObjectName("sfCamera")
        sfCamera.resize(910, 900)
        self.progressBar = QtWidgets.QProgressBar(sfCamera)
        self.progressBar.setGeometry(QtCore.QRect(630, 100, 241, 30))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar_2 = QtWidgets.QProgressBar(sfCamera)
        self.progressBar_2.setGeometry(QtCore.QRect(630, 150, 241, 30))
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.progressBar_3 = QtWidgets.QProgressBar(sfCamera)
        self.progressBar_3.setGeometry(QtCore.QRect(630, 200, 241, 30))
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setObjectName("progressBar_3")
        self.progressBar_4 = QtWidgets.QProgressBar(sfCamera)
        self.progressBar_4.setGeometry(QtCore.QRect(630, 250, 241, 30))
        self.progressBar_4.setProperty("value", 0)
        self.progressBar_4.setObjectName("progressBar_4")
        self.progressBar_5 = QtWidgets.QProgressBar(sfCamera)
        self.progressBar_5.setGeometry(QtCore.QRect(630, 300, 241, 30))
        self.progressBar_5.setProperty("value", 0)
        self.progressBar_5.setObjectName("progressBar_5")
        self.progressBar_6 = QtWidgets.QProgressBar(sfCamera)
        self.progressBar_6.setGeometry(QtCore.QRect(630, 350, 241, 30))
        self.progressBar_6.setProperty("value", 0)
        self.progressBar_6.setObjectName("progressBar_6")
        self.progressBar_7 = QtWidgets.QProgressBar(sfCamera)
        self.progressBar_7.setGeometry(QtCore.QRect(630, 400, 241, 30))
        self.progressBar_7.setProperty("value", 0)
        self.progressBar_7.setObjectName("progressBar_7")
        self.label_2 = QtWidgets.QLabel(sfCamera)
        self.label_2.setGeometry(QtCore.QRect(120, 410, 250, 40))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(sfCamera)
        self.label_3.setGeometry(QtCore.QRect(500, 90, 130, 40))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(sfCamera)
        self.label_4.setGeometry(QtCore.QRect(500, 140, 130, 40))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(sfCamera)
        self.label_5.setGeometry(QtCore.QRect(500, 190, 130, 40))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(sfCamera)
        self.label_6.setGeometry(QtCore.QRect(500, 240, 130, 40))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(sfCamera)
        self.label_7.setGeometry(QtCore.QRect(500, 290, 130, 40))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(sfCamera)
        self.label_8.setGeometry(QtCore.QRect(500, 340, 130, 40))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(sfCamera)
        self.label_9.setGeometry(QtCore.QRect(500, 390, 130, 40))
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(sfCamera)
        self.label_10.setGeometry(QtCore.QRect(50, 50, 420, 315))
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.pushButton = QtWidgets.QPushButton(sfCamera)
        self.pushButton.setGeometry(QtCore.QRect(590, 30, 101, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(sfCamera)
        self.pushButton_2.setGeometry(QtCore.QRect(730, 30, 101, 41))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(sfCamera)
        QtCore.QMetaObject.connectSlotsByName(sfCamera)

    def retranslateUi(self, sfCamera):
        _translate = QtCore.QCoreApplication.translate
        sfCamera.setWindowTitle(_translate("sfCamera", "SoulFlow-实时拍摄"))
        self.label_2.setText(_translate("sfCamera", "推理速度:"))
        self.label_3.setText(_translate("sfCamera", "快乐"))
        self.label_4.setText(_translate("sfCamera", "惊讶"))
        self.label_5.setText(_translate("sfCamera", "其他"))
        self.label_6.setText(_translate("sfCamera", "厌恶"))
        self.label_7.setText(_translate("sfCamera", "恐惧"))
        self.label_8.setText(_translate("sfCamera", "压抑"))
        self.label_9.setText(_translate("sfCamera", "悲伤"))
        self.label_10.setText(_translate("sfCamera", "(视频缩略图)"))
        self.pushButton.setText(_translate("sfCamera", "开始"))
        self.pushButton_2.setText(_translate("sfCamera", "停止"))

