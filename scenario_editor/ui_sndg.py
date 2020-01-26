# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sndg1.ui',
# licensing of 'sndg1.ui' applies.
#
# Created: Tue May 28 10:21:30 2019
#      by: pyside2-uic  running on PySide2 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(958, 867)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setMinimumSize(QtCore.QSize(800, 800))
        self.graphicsView.setMaximumSize(QtCore.QSize(800, 800))
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout_2.addWidget(self.graphicsView)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label5 = QtWidgets.QLabel(self.centralwidget)
        self.label5.setStyleSheet("color: rgba(0, 0, 0, 0.);")
        self.label5.setObjectName("label5")
        self.verticalLayout_3.addWidget(self.label5)
        self.label4 = QtWidgets.QLabel(self.centralwidget)
        self.label4.setStyleSheet("color: rgba(0, 0, 0, 0.);")
        self.label4.setObjectName("label4")
        self.verticalLayout_3.addWidget(self.label4)
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setStyleSheet("color: rgba(0, 0, 0, 0.);")
        self.label3.setObjectName("label3")
        self.verticalLayout_3.addWidget(self.label3)
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setStyleSheet("color: rgba(0, 0, 0, 0.);")
        self.label2.setObjectName("label2")
        self.verticalLayout_3.addWidget(self.label2)
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setStyleSheet("color: rgba(0, 0, 0, 0.);")
        self.label1.setObjectName("label1")
        self.verticalLayout_3.addWidget(self.label1)
        self.label0 = QtWidgets.QLabel(self.centralwidget)
        self.label0.setStyleSheet("color: rgba(0, 0, 0, 0.);")
        self.label0.setObjectName("label0")
        self.verticalLayout_3.addWidget(self.label0)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.slider = QtWidgets.QSlider(self.centralwidget)
        self.slider.setMaximum(100)
        self.slider.setOrientation(QtCore.Qt.Vertical)
        self.slider.setObjectName("slider")
        self.horizontalLayout_3.addWidget(self.slider)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.sendButton = QtWidgets.QPushButton(self.centralwidget)
        self.sendButton.setObjectName("sendButton")
        self.verticalLayout_2.addWidget(self.sendButton)
        self.getButton = QtWidgets.QPushButton(self.centralwidget)
        self.getButton.setObjectName("getButton")
        self.verticalLayout_2.addWidget(self.getButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.estimateBox = QtWidgets.QCheckBox(self.centralwidget)
        self.estimateBox.setObjectName("estimateBox")
        self.verticalLayout_2.addWidget(self.estimateBox)
        self.estimateButton = QtWidgets.QPushButton(self.centralwidget)
        self.estimateButton.setObjectName("estimateButton")
        self.verticalLayout_2.addWidget(self.estimateButton)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 958, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("font: 75 11pt \"Ubuntu\";")
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "Social Navigation Dataset Generator", None, -1))
        self.label5.setText(QtWidgets.QApplication.translate("MainWindow", "perfect", None, -1))
        self.label4.setText(QtWidgets.QApplication.translate("MainWindow", "very good", None, -1))
        self.label3.setText(QtWidgets.QApplication.translate("MainWindow", "good", None, -1))
        self.label2.setText(QtWidgets.QApplication.translate("MainWindow", "acceptable", None, -1))
        self.label1.setText(QtWidgets.QApplication.translate("MainWindow", "undesirable", None, -1))
        self.label0.setText(QtWidgets.QApplication.translate("MainWindow", "unacceptable", None, -1))
        self.sendButton.setText(QtWidgets.QApplication.translate("MainWindow", "send\n"
"context\n"
"assessment", None, -1))
        self.getButton.setText(QtWidgets.QApplication.translate("MainWindow", "get new\n"
"sample", None, -1))
        self.estimateBox.setText(QtWidgets.QApplication.translate("MainWindow", "automatically\n"
"estimate", None, -1))
        self.estimateButton.setText(QtWidgets.QApplication.translate("MainWindow", "estimate", None, -1))

