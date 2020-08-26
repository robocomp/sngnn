import math

from PySide2 import QtCore, QtGui, QtWidgets

class Robot(QtWidgets.QGraphicsItem):
    BoundingRect = QtCore.QRectF(-20, -20, 40, 40)

    def __init__(self):
        super(Robot, self).__init__()

    def boundingRect(self):
        return Robot.BoundingRect

    def paint(self, painter, option, widget):
        # Body
        painter.setBrush(QtCore.Qt.red)
        bodyPolygon = QtGui.QPolygon()
        bodyPolygon.append(QtCore.QPoint(-20, 20))
        bodyPolygon.append(QtCore.QPoint(-20, -13))
        bodyPolygon.append(QtCore.QPoint(-13, -20))

        bodyPolygon.append(QtCore.QPoint(13, -20))
        bodyPolygon.append(QtCore.QPoint(20, -13))
        bodyPolygon.append(QtCore.QPoint(20, 20))

        bodyPolygon.append(QtCore.QPoint(-20, 20))
        painter.drawPolygon(bodyPolygon)
        # Wheels
        painter.setBrush(QtCore.Qt.black)
        painter.drawRect(+18-4, -8, 8, 16)
        painter.drawRect(-18-4, -8, 8, 16)

        nosePolygon = QtGui.QPolygon()
        nosePolygon.append(QtCore.QPoint(0, -22))
        nosePolygon.append(QtCore.QPoint(-10, -15))
        nosePolygon.append(QtCore.QPoint(10, -15))
        nosePolygon.append(QtCore.QPoint(0, -22))
        painter.drawPolygon(nosePolygon)
