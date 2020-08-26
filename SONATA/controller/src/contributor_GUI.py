import sys
from PySide2 import QtCore
from PySide2.QtCore import SIGNAL
from PySide2.QtWidgets import (QLabel, QLineEdit, QPushButton, QApplication,
    QVBoxLayout, QDialog, QWidget, QFormLayout)
from PySide2.QtCore import QSettings

val = None

class CForm(QDialog):
    def __init__(self, parent=None):
        super(CForm, self).__init__(parent)
        self.val = None
        self.le = QLineEdit()
        self.le.setObjectName(u"Default")
        self.settings = QSettings("dataset", "xxx")
        le = self.settings.value("le", "default")
        self.le.setText(le)
        self.le.textChanged.connect(self.contributor_changed)
        self.pb = QPushButton()
        self.pb.setObjectName("Enter")
        self.pb.setText("Enter") 
        layout = QFormLayout()
        layout.addWidget(self.le)
        layout.addWidget(self.pb)
        self.setLayout(layout)
        self.pb.clicked.connect(lambda: self.buttonClick())
        self.setWindowTitle("Enter Contributor's Name")

    @QtCore.Slot()
    def contributor_changed(self):
        self.settings.setValue("le", self.le.text())
        self.settings.setValue("contributor", self.le.text())
        self.val = self.le.text()
        print(self.val)

    def buttonClick(self):
        print(self.val)
        self.close()
        
