# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainUI.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_guiDlg(object):
    def setupUi(self, guiDlg):
        if not guiDlg.objectName():
            guiDlg.setObjectName(u"guiDlg")
        guiDlg.resize(1042, 819)
        self.verticalLayout = QVBoxLayout(guiDlg)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_5)

        self.configuration = QPushButton(guiDlg)
        self.configuration.setObjectName(u"configuration")
        self.configuration.setCheckable(True)

        self.horizontalLayout.addWidget(self.configuration)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.regenerate = QPushButton(guiDlg)
        self.regenerate.setObjectName(u"regenerate")

        self.horizontalLayout.addWidget(self.regenerate)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.label_2 = QLabel(guiDlg)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.contributor = QLineEdit(guiDlg)
        self.contributor.setObjectName(u"contributor")

        self.horizontalLayout.addWidget(self.contributor)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.quit = QPushButton(guiDlg)
        self.quit.setObjectName(u"quit")

        self.horizontalLayout.addWidget(self.quit)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.label = QLabel(guiDlg)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QSize(1024, 768))
        self.label.setMaximumSize(QSize(10240, 7680))
        self.label.setSizeIncrement(QSize(1, 1))
        self.label.setScaledContents(True)

        self.verticalLayout.addWidget(self.label)


        self.retranslateUi(guiDlg)

        QMetaObject.connectSlotsByName(guiDlg)
    # setupUi

    def retranslateUi(self, guiDlg):
        guiDlg.setWindowTitle(QCoreApplication.translate("guiDlg", u"Social Navigation Dataset Generator", None))
        self.configuration.setText(QCoreApplication.translate("guiDlg", u"configuration", None))
        self.regenerate.setText(QCoreApplication.translate("guiDlg", u"regenerate", None))
        self.label_2.setText(QCoreApplication.translate("guiDlg", u"contributor's unique id:", None))
        self.contributor.setText(QCoreApplication.translate("guiDlg", u"default", None))
        self.quit.setText(QCoreApplication.translate("guiDlg", u"quit", None))
        self.label.setText("")
    # retranslateUi

