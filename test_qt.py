from soulflow import *
import cv2
import time
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, QtChart
from ui_MainWindow import Ui_MainWindow
from sf_camera import Ui_sfCamera
from sf_video import Ui_sfVideo


class SfCamera(QtWidgets.QWidget):
    def __init__(self, sf:SoulFlow, parent=None) -> None:
        super().__init__(parent)
        self.ui = Ui_sfCamera()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.start)
        self.ui.pushButton_2.clicked.connect(self.stop)
        self.sf = sf
        self.flag = 0

        self.chart = QtChart.QChart()
        self.chartView = QtChart.QChartView(self)
        self.chartView.setGeometry(QtCore.QRect(40, 450, 850, 420))
        self.chartView.setChart(self.chart)
        self.series0 = QtChart.QLineSeries()
        self.series0.setName("快乐")
        self.series1 = QtChart.QLineSeries()
        self.series1.setName("惊讶")
        self.series2 = QtChart.QLineSeries()
        self.series2.setName("其他")
        self.series3 = QtChart.QLineSeries()
        self.series3.setName("厌恶")
        self.series4 = QtChart.QLineSeries()
        self.series4.setName("恐惧")
        self.series5 = QtChart.QLineSeries()
        self.series5.setName("压抑")
        self.series5.setColor(QtGui.QColor(80, 80, 80))
        self.series6 = QtChart.QLineSeries()
        self.series6.setName("悲伤")
        self.series6.setColor(QtGui.QColor(150, 150, 150))
        self.axisX = QtChart.QValueAxis()
        self.axisY = QtChart.QValueAxis()
        self.axisX.setRange(0, 100)
        self.axisY.setRange(0, 100)
        self.chart.addSeries(self.series0)
        self.chart.addSeries(self.series1)
        self.chart.addSeries(self.series2)
        self.chart.addSeries(self.series3)
        self.chart.addSeries(self.series4)
        self.chart.addSeries(self.series5)
        self.chart.addSeries(self.series6)
        self.chart.setAxisX(self.axisX, self.series0)
        self.chart.setAxisY(self.axisY, self.series0)
        self.chart.setAxisX(self.axisX, self.series1)
        self.chart.setAxisY(self.axisY, self.series1)
        self.chart.setAxisX(self.axisX, self.series2)
        self.chart.setAxisY(self.axisY, self.series2)
        self.chart.setAxisX(self.axisX, self.series3)
        self.chart.setAxisY(self.axisY, self.series3)
        self.chart.setAxisX(self.axisX, self.series4)
        self.chart.setAxisY(self.axisY, self.series4)
        self.chart.setAxisX(self.axisX, self.series5)
        self.chart.setAxisY(self.axisY, self.series5)
        self.chart.setAxisX(self.axisX, self.series6)
        self.chart.setAxisY(self.axisY, self.series6)
    
    def __del__(self):
        print("Camera closed.")
    
    def start(self):
        if self.flag == 1:
            return
        self.flag = 1
        camera = cv2.VideoCapture(-1)
        if not camera.isOpened():
            print("camera_thread: Can't start camera.")
            return
        i = 0
        f = 0
        imaget = np.zeros([192, 256, 3], dtype=np.int8)
        t = time.time()
        while self.flag == 1:
            status, frame = camera.read()
            if status:
                image = cv2.resize(frame, [256, 192])
                imaget[:, :, 0] = image[:, :, 2]
                imaget[:, :, 1] = image[:, :, 1]
                imaget[:, :, 2] = image[:, :, 0]
                qimg = QtGui.QImage(imaget.data, 256, 192, QtGui.QImage.Format_RGB888)
                qimg = qimg.scaled(420, 315)
                self.ui.label_10.setPixmap(QtGui.QPixmap(qimg))
                QtWidgets.qApp.processEvents()
                self.sf.set_sample(0, image)
                if f != 1:
                    f = 2
            else:
                print("camera_thread: Warning: Read image from camera failed.")
                if f == 1:
                    f = -1
                else:
                    f = -2
            
            if f * f == 1:
                result = self.sf.predict_gru().reshape([7])
                n0 = int(result[0] * 100)
                n1 = int(result[1] * 100)
                n2 = int(result[2] * 100)
                n3 = int(result[3] * 100)
                n4 = int(result[4] * 100)
                n5 = int(result[5] * 100)
                n6 = int(result[6] * 100)
                if i % 100 == 0:
                    self.series0.clear()
                    self.series1.clear()
                    self.series2.clear()
                    self.series3.clear()
                    self.series4.clear()
                    self.series5.clear()
                    self.series6.clear()
                self.series0.append(i % 100, n0)
                self.series1.append(i % 100, n1)
                self.series2.append(i % 100, n2)
                self.series3.append(i % 100, n3)
                self.series4.append(i % 100, n4)
                self.series5.append(i % 100, n5)
                self.series6.append(i % 100, n6)
                self.ui.progressBar.setValue(n0)
                self.ui.progressBar_2.setValue(n1)
                self.ui.progressBar_3.setValue(n2)
                self.ui.progressBar_4.setValue(n3)
                self.ui.progressBar_5.setValue(n4)
                self.ui.progressBar_6.setValue(n5)
                self.ui.progressBar_7.setValue(n6)
                self.ui.label_2.setText("推理速度: " + str(1 / (time.time() - t)))
                QtWidgets.qApp.processEvents()
                t = time.time()
            elif f == 2:
                f = 1
            
            if f == 1:
                self.sf.forward_gru(0)
            
            i += 1
        camera.release()
        self.series0.clear()
        self.series1.clear()
        self.series2.clear()
        self.series3.clear()
        self.series4.clear()
        self.series5.clear()
        self.series6.clear()
    
    def stop(self):
        self.flag = 0
    
    def closeEvent(self, event):
        print("closeEvent")
        self.flag = 0
        super().closeEvent(event)


class SfVideo(QtWidgets.QWidget):
    def __init__(self, sf:SoulFlow, parent=None) -> None:
        super().__init__(parent)
        self.ui = Ui_sfVideo()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.predict)
        self.sf = sf
        self.flag = 0

        self.chart = QtChart.QChart()
        self.chartView = QtChart.QChartView(self)
        self.chartView.setGeometry(QtCore.QRect(40, 450, 850, 420))
        self.chartView.setChart(self.chart)
        self.series0 = QtChart.QLineSeries()
        self.series0.setName("快乐")
        self.series1 = QtChart.QLineSeries()
        self.series1.setName("惊讶")
        self.series2 = QtChart.QLineSeries()
        self.series2.setName("其他")
        self.series3 = QtChart.QLineSeries()
        self.series3.setName("厌恶")
        self.series4 = QtChart.QLineSeries()
        self.series4.setName("恐惧")
        self.series5 = QtChart.QLineSeries()
        self.series5.setName("压抑")
        self.series5.setColor(QtGui.QColor(80, 80, 80))
        self.series6 = QtChart.QLineSeries()
        self.series6.setName("悲伤")
        self.series6.setColor(QtGui.QColor(150, 150, 150))
        self.axisX = QtChart.QValueAxis()
        self.axisY = QtChart.QValueAxis()
        self.axisY.setRange(0, 100)
        self.chart.addSeries(self.series0)
        self.chart.addSeries(self.series1)
        self.chart.addSeries(self.series2)
        self.chart.addSeries(self.series3)
        self.chart.addSeries(self.series4)
        self.chart.addSeries(self.series5)
        self.chart.addSeries(self.series6)
        self.chart.setAxisX(self.axisX, self.series0)
        self.chart.setAxisY(self.axisY, self.series0)
        self.chart.setAxisX(self.axisX, self.series1)
        self.chart.setAxisY(self.axisY, self.series1)
        self.chart.setAxisX(self.axisX, self.series2)
        self.chart.setAxisY(self.axisY, self.series2)
        self.chart.setAxisX(self.axisX, self.series3)
        self.chart.setAxisY(self.axisY, self.series3)
        self.chart.setAxisX(self.axisX, self.series4)
        self.chart.setAxisY(self.axisY, self.series4)
        self.chart.setAxisX(self.axisX, self.series5)
        self.chart.setAxisY(self.axisY, self.series5)
        self.chart.setAxisX(self.axisX, self.series6)
        self.chart.setAxisY(self.axisY, self.series6)
    
    def __del__(self):
        print("Video closed.")
    
    def predict(self):
        self.series0.clear()
        self.series1.clear()
        self.series2.clear()
        self.series3.clear()
        self.series4.clear()
        self.series5.clear()
        self.series6.clear()
        curPath = QtCore.QDir.currentPath()
        title = "选择视频文件" 
        filt = "视频文件(*.wmv *.avi *.mp4);;所有文件(*.*)"
        fileName, flt = QtWidgets.QFileDialog.getOpenFileName(self, title, curPath, filt)
        if (fileName == ""):
            return
        print(fileName)
        self.sf.resetTemp()
        video = cv2.VideoCapture(fileName)
        if not video.isOpened():
            print("predict: Can't read video:", fileName)
            return
        i = 0
        imaget = np.zeros([192, 256, 3], dtype=np.int8)
        t = time.time()
        while True:
            print("predict: t1")
            status, frame = video.read()
            if status:
                image = cv2.resize(frame, [256, 192])
                imaget[:, :, 0] = image[:, :, 2]
                imaget[:, :, 1] = image[:, :, 1]
                imaget[:, :, 2] = image[:, :, 0]
                qimg = QtGui.QImage(imaget.data, 256, 192, QtGui.QImage.Format_RGB888)
                qimg = qimg.scaled(420, 315)
                self.ui.label_10.setPixmap(QtGui.QPixmap(qimg))
                self.ui.label.setText("帧: " + str(i + 1))
                self.sf.set_sample(0, image)
            else:
                break

            self.sf.forward_gru(0)
            QtWidgets.qApp.processEvents()
            result = self.sf.predict_gru().reshape([7])
            print("predict: t2")
            n0 = int(result[0] * 100)
            n1 = int(result[1] * 100)
            n2 = int(result[2] * 100)
            n3 = int(result[3] * 100)
            n4 = int(result[4] * 100)
            n5 = int(result[5] * 100)
            n6 = int(result[6] * 100)
            self.series0.append(i, n0)
            self.series1.append(i, n1)
            self.series2.append(i, n2)
            self.series3.append(i, n3)
            self.series4.append(i, n4)
            self.series5.append(i, n5)
            self.series6.append(i, n6)
            self.ui.progressBar.setValue(n0)
            self.ui.progressBar_2.setValue(n1)
            self.ui.progressBar_3.setValue(n2)
            self.ui.progressBar_4.setValue(n3)
            self.ui.progressBar_5.setValue(n4)
            self.ui.progressBar_6.setValue(n5)
            self.ui.progressBar_7.setValue(n6)
            self.ui.label_2.setText("推理速度: " + str(1 / (time.time() - t)))
            t = time.time()
            print("predict: t3")

            i += 1
            self.axisX.setRange(0, i)
        
        video.release()


class SfMainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数，创建窗体
        self.ui = Ui_MainWindow()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面
        self.sf = SoulFlow(1, "soulflow.pkl")

        self.ui.tabWidget.setVisible(False)
        self.ui.tabWidget.clear()  # 清除所有页面
        self.ui.tabWidget.setTabsClosable(True)  # Page有关闭按钮
        self.ui.tabWidget.setDocumentMode(True)

        self.setCentralWidget(self.ui.tabWidget)
        self.setAutoFillBackground(True)  # 自动绘制背景

        self.__pic = QtGui.QPixmap("splash.png")  # 载入背景图片到内存，提高绘制速度

    def __del__(self):
        self.sf.free()
        print("All resources released.")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, self.ui.mainToolBar.height(), self.width(),
                           self.height()-self.ui.mainToolBar.height()-self.ui.statusBar.height(),
                           self.__pic)
        super().paintEvent(event)

    @QtCore.pyqtSlot()
    def on_actWidget_triggered(self):
        print("on_actWidget_triggered")
        form = SfCamera(self.sf, self)
        form.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        form.setWindowFlag(QtCore.Qt.Window, True)
        form.show()

    @QtCore.pyqtSlot()
    def on_actWindow_triggered(self):
        print("on_actWindow_triggered")
        form = SfVideo(self.sf, self)
        form.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        form.setWindowFlag(QtCore.Qt.Window, True)
        form.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = SfMainWindow()
    main_window.show()
    sys.exit(app.exec_())
