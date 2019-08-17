import math

from PySide import QtCore, QtGui

from polygonmisc import rotatePolygon, translatePolygon

class RegularObject(QtGui.QGraphicsItem):
    BoundingRect = QtCore.QRectF(-20, -20, 40, 40)

    def __init__(self, id, xPos, yPos, angle):
        super(RegularObject, self).__init__()
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.setAngle(angle)
        self.setPos(self.xPos, self.yPos)
        self.colour = QtCore.Qt.green

    @classmethod
    def from_json(RegularObject, json_data):
        id = json_data['id']
        xPos = json_data['xPos']
        yPos = json_data['yPos']
        angle = json_data['orientation']
        return RegularObject(id, xPos, yPos, angle)


    def setAngle(self, a):
        self.angle = a
        if self.angle > 180.:
            self.angle = -360.+self.angle
        self.setRotation(self.angle)

    def polygon(self):
        w = 20
        h = 20
        polygon = QtGui.QPolygon()
        polygon.append( QtCore.QPoint(-w, -h) )
        polygon.append( QtCore.QPoint(-w, +h) )
        polygon.append( QtCore.QPoint(+w, +h) )
        polygon.append( QtCore.QPoint(+w, -h) )
        polygon.append( QtCore.QPoint(-w, -h) )
        polygon = rotatePolygon(polygon, theta=self.angle*math.pi/180.)
        polygon = translatePolygon(polygon, tx=self.xPos, ty=self.yPos)
        return polygon


    def boundingRect(self):
        return RegularObject.BoundingRect

    def paint(self, painter, option, widget):
        # Body
        painter.setBrush(self.colour)
        painter.drawRect(self.BoundingRect)
