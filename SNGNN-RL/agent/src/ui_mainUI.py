# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_guiDlg(object):
    def setupUi(self, guiDlg):
        if not guiDlg.objectName():
            guiDlg.setObjectName(u"guiDlg")
        guiDlg.resize(250, 486)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(guiDlg.sizePolicy().hasHeightForWidth())
        guiDlg.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(guiDlg)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_nhumans = QLabel(guiDlg)
        self.label_nhumans.setObjectName(u"label_nhumans")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_nhumans.setFont(font)

        self.verticalLayout.addWidget(self.label_nhumans)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_min_nhumans = QLabel(guiDlg)
        self.label_min_nhumans.setObjectName(u"label_min_nhumans")

        self.gridLayout.addWidget(self.label_min_nhumans, 0, 0, 1, 1)

        self.nhumans_max = QSpinBox(guiDlg)
        self.nhumans_max.setObjectName(u"nhumans_max")
        self.nhumans_max.setMaximum(20)
        self.nhumans_max.setValue(5)

        self.gridLayout.addWidget(self.nhumans_max, 1, 1, 1, 1)

        self.label_max_nhumans = QLabel(guiDlg)
        self.label_max_nhumans.setObjectName(u"label_max_nhumans")

        self.gridLayout.addWidget(self.label_max_nhumans, 0, 1, 1, 1)

        self.nhumans_lambda = QDoubleSpinBox(guiDlg)
        self.nhumans_lambda.setObjectName(u"nhumans_lambda")
        self.nhumans_lambda.setDecimals(3)
        self.nhumans_lambda.setMaximum(10.000000000000000)
        self.nhumans_lambda.setSingleStep(0.100000000000000)
        self.nhumans_lambda.setValue(5.000000000000000)

        self.gridLayout.addWidget(self.nhumans_lambda, 1, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.label_nwandhumans = QLabel(guiDlg)
        self.label_nwandhumans.setObjectName(u"label_nwandhumans")
        self.label_nwandhumans.setFont(font)

        self.verticalLayout.addWidget(self.label_nwandhumans)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_max_nwandhumans = QLabel(guiDlg)
        self.label_max_nwandhumans.setObjectName(u"label_max_nwandhumans")

        self.gridLayout_2.addWidget(self.label_max_nwandhumans, 0, 1, 1, 1)

        self.label_min_nwandhumans = QLabel(guiDlg)
        self.label_min_nwandhumans.setObjectName(u"label_min_nwandhumans")

        self.gridLayout_2.addWidget(self.label_min_nwandhumans, 0, 0, 1, 1)

        self.nwandhumans_max = QSpinBox(guiDlg)
        self.nwandhumans_max.setObjectName(u"nwandhumans_max")
        self.nwandhumans_max.setMaximum(20)
        self.nwandhumans_max.setValue(5)

        self.gridLayout_2.addWidget(self.nwandhumans_max, 1, 1, 1, 1)

        self.nwandhumans_lambda = QDoubleSpinBox(guiDlg)
        self.nwandhumans_lambda.setObjectName(u"nwandhumans_lambda")
        self.nwandhumans_lambda.setDecimals(3)
        self.nwandhumans_lambda.setMaximum(10.000000000000000)
        self.nwandhumans_lambda.setSingleStep(0.100000000000000)
        self.nwandhumans_lambda.setValue(5.000000000000000)

        self.gridLayout_2.addWidget(self.nwandhumans_lambda, 1, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_2)

        self.label_nplants = QLabel(guiDlg)
        self.label_nplants.setObjectName(u"label_nplants")
        self.label_nplants.setFont(font)

        self.verticalLayout.addWidget(self.label_nplants)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_min_nplants = QLabel(guiDlg)
        self.label_min_nplants.setObjectName(u"label_min_nplants")

        self.gridLayout_3.addWidget(self.label_min_nplants, 1, 0, 1, 1)

        self.label_max_nplants = QLabel(guiDlg)
        self.label_max_nplants.setObjectName(u"label_max_nplants")

        self.gridLayout_3.addWidget(self.label_max_nplants, 1, 1, 1, 1)

        self.nplants_max = QSpinBox(guiDlg)
        self.nplants_max.setObjectName(u"nplants_max")
        self.nplants_max.setMaximum(20)
        self.nplants_max.setValue(5)

        self.gridLayout_3.addWidget(self.nplants_max, 2, 1, 1, 1)

        self.nplants_lambda = QDoubleSpinBox(guiDlg)
        self.nplants_lambda.setObjectName(u"nplants_lambda")
        self.nplants_lambda.setDecimals(3)
        self.nplants_lambda.setMaximum(10.000000000000000)
        self.nplants_lambda.setSingleStep(0.100000000000000)
        self.nplants_lambda.setValue(2.000000000000000)

        self.gridLayout_3.addWidget(self.nplants_lambda, 2, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_3)

        self.label_nplants_2 = QLabel(guiDlg)
        self.label_nplants_2.setObjectName(u"label_nplants_2")
        self.label_nplants_2.setFont(font)

        self.verticalLayout.addWidget(self.label_nplants_2)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_min_ntables = QLabel(guiDlg)
        self.label_min_ntables.setObjectName(u"label_min_ntables")

        self.gridLayout_4.addWidget(self.label_min_ntables, 0, 0, 1, 1)

        self.label_max_ntables = QLabel(guiDlg)
        self.label_max_ntables.setObjectName(u"label_max_ntables")

        self.gridLayout_4.addWidget(self.label_max_ntables, 0, 1, 1, 1)

        self.ntables_max = QSpinBox(guiDlg)
        self.ntables_max.setObjectName(u"ntables_max")
        self.ntables_max.setMaximum(20)
        self.ntables_max.setValue(5)

        self.gridLayout_4.addWidget(self.ntables_max, 1, 1, 1, 1)

        self.ntables_lambda = QDoubleSpinBox(guiDlg)
        self.ntables_lambda.setObjectName(u"ntables_lambda")
        self.ntables_lambda.setDecimals(3)
        self.ntables_lambda.setMaximum(10.000000000000000)
        self.ntables_lambda.setSingleStep(0.100000000000000)
        self.ntables_lambda.setValue(5.000000000000000)

        self.gridLayout_4.addWidget(self.ntables_lambda, 1, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_4)

        self.label_nrelations = QLabel(guiDlg)
        self.label_nrelations.setObjectName(u"label_nrelations")
        self.label_nrelations.setFont(font)

        self.verticalLayout.addWidget(self.label_nrelations)

        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_max_nrelations = QLabel(guiDlg)
        self.label_max_nrelations.setObjectName(u"label_max_nrelations")

        self.gridLayout_5.addWidget(self.label_max_nrelations, 0, 1, 1, 1)

        self.label_min_nrelations = QLabel(guiDlg)
        self.label_min_nrelations.setObjectName(u"label_min_nrelations")

        self.gridLayout_5.addWidget(self.label_min_nrelations, 0, 0, 1, 1)

        self.nrelations_max = QSpinBox(guiDlg)
        self.nrelations_max.setObjectName(u"nrelations_max")
        self.nrelations_max.setMaximum(20)
        self.nrelations_max.setValue(5)

        self.gridLayout_5.addWidget(self.nrelations_max, 1, 1, 1, 1)

        self.nrelations_lambda = QDoubleSpinBox(guiDlg)
        self.nrelations_lambda.setObjectName(u"nrelations_lambda")
        self.nrelations_lambda.setDecimals(3)
        self.nrelations_lambda.setMaximum(10.000000000000000)
        self.nrelations_lambda.setSingleStep(0.100000000000000)
        self.nrelations_lambda.setValue(5.000000000000000)

        self.gridLayout_5.addWidget(self.nrelations_lambda, 1, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_5)

        self.include_walls = QCheckBox(guiDlg)
        self.include_walls.setObjectName(u"include_walls")

        self.verticalLayout.addWidget(self.include_walls)

        self.draw_graph = QCheckBox(guiDlg)
        self.draw_graph.setObjectName(u"draw_graph")

        self.verticalLayout.addWidget(self.draw_graph)

        self.quitButton = QPushButton(guiDlg)
        self.quitButton.setObjectName(u"quitButton")

        self.verticalLayout.addWidget(self.quitButton)


        self.retranslateUi(guiDlg)

        QMetaObject.connectSlotsByName(guiDlg)
    # setupUi

    def retranslateUi(self, guiDlg):
        guiDlg.setWindowTitle(QCoreApplication.translate("guiDlg", u"Configuration", None))
        self.label_nhumans.setText(QCoreApplication.translate("guiDlg", u"Number of humans", None))
        self.label_min_nhumans.setText(QCoreApplication.translate("guiDlg", u"\u03bb", None))
        self.label_max_nhumans.setText(QCoreApplication.translate("guiDlg", u"Max", None))
        self.label_nwandhumans.setText(QCoreApplication.translate("guiDlg", u"Number of wandering humans", None))
        self.label_max_nwandhumans.setText(QCoreApplication.translate("guiDlg", u"Max", None))
        self.label_min_nwandhumans.setText(QCoreApplication.translate("guiDlg", u"\u03bb", None))
        self.label_nplants.setText(QCoreApplication.translate("guiDlg", u"Number of plants", None))
        self.label_min_nplants.setText(QCoreApplication.translate("guiDlg", u"\u03bb", None))
        self.label_max_nplants.setText(QCoreApplication.translate("guiDlg", u"max", None))
        self.label_nplants_2.setText(QCoreApplication.translate("guiDlg", u"Number of tables", None))
        self.label_min_ntables.setText(QCoreApplication.translate("guiDlg", u"\u03bb", None))
        self.label_max_ntables.setText(QCoreApplication.translate("guiDlg", u"Max", None))
        self.label_nrelations.setText(QCoreApplication.translate("guiDlg", u"Number of relations", None))
        self.label_max_nrelations.setText(QCoreApplication.translate("guiDlg", u"Max", None))
        self.label_min_nrelations.setText(QCoreApplication.translate("guiDlg", u"\u03bb", None))
        self.include_walls.setText(QCoreApplication.translate("guiDlg", u"walls", None))
        self.draw_graph.setText(QCoreApplication.translate("guiDlg", u"draw graph", None))
        self.quitButton.setText(QCoreApplication.translate("guiDlg", u"quit", None))
    # retranslateUi

