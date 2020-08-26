import math

from PySide import QtCore, QtGui

class Interaction(QtGui.QGraphicsItem):
    def __init__(self, a, b):
        super(Interaction, self).__init__()
        self.a = a
        self.b = b
        self.xPos = (self.a.xPos + self.b.xPos) / 2
        self.yPos = (self.a.yPos + self.b.yPos) / 2
        s1 = self.a.xPos - self.b.xPos
        s2 = self.a.yPos - self.b.yPos
        d = math.sqrt(s1*s1 + s2*s2)
        angle = math.atan2(s2, s1)

        self.length = d
        self.setPos(self.xPos, self.yPos)
        self.BoundingRect = QtCore.QRectF(-d/2, -10, d, 20)
        self.setRotation(angle*180./math.pi)
        self.setZValue(2)

    def setInteractionObject(self, obj):
        self.interactionObject = obj

    def boundingRect(self):
        return self.BoundingRect

    def paint(self, painter, option, widget):
        # Body.
        painter.drawRect(-self.length/2, -10, self.length, 20)
