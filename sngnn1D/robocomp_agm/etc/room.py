import random

from PySide import QtCore, QtGui

from polygonmisc import translatePolygon, movePolygon


def getRectRoom():
    w = min(10, abs(random.normalvariate(1.5, 2.5))+1.5)/2 * 100
    h = min(10, abs(random.normalvariate(3.0, 2.5))+4.0)/2 * 100
    polygon = QtGui.QPolygon()
    polygon.append( QtCore.QPoint(-w, -h) )
    polygon.append( QtCore.QPoint(-w, +h) )
    polygon.append( QtCore.QPoint(+w, +h) )
    polygon.append( QtCore.QPoint(+w, -h) )
    return polygon


def getRobotPolygon():
    w = h = 25
    polygon = QtGui.QPolygon()
    polygon.append( QtCore.QPoint(-w, -h) )
    polygon.append( QtCore.QPoint(-w, +h) )
    polygon.append( QtCore.QPoint(+w, +h) )
    polygon.append( QtCore.QPoint(+w, -h) )
    return polygon


class Room(QtGui.QGraphicsItem):
    def __init__(self, poly=None):
        super(Room, self).__init__()
        self.poly = None
        if poly is not None:
            self.poly = QtGui.QPolygon()
            for p in poly:
                #print(p)
                self.poly.append( QtCore.QPoint(p.x(), p.y()) )
        else:
            # Generate a room so that the robot will be inside the room, not colliding with the walls
            robot = getRobotPolygon()
            while self.poly is None:
                # Generate three randomised polygons
                p1 = translatePolygon(getRectRoom())
                p2 = translatePolygon(getRectRoom())
                p3 = translatePolygon(getRectRoom())
                # The room is generated as: (p1 - p2) + p3
                pRes = p1
                if random.randint(0,1) == 0:
                    pRes = pRes.subtracted(p2)
                if random.randint(0,1) == 0:
                    pRes = pRes.united(p3)
                # Perform an additional check to verify that
                # the polygon is not a degenerate one
                error_found = False
                l = pRes.toList()[:-1]
                for e in l:
                    if l.count(e) > 1:
                        error_found = True
                # If the polygon has passed all the checks, go ahead
                if error_found == False:
                    pRes = movePolygon(pRes)
                    pRobot = pRes.united(robot)
                    if len(pRes) == len(pRobot):
                        self.poly = pRes



    def boundingRect(self):
        return self.poly.boundingRect()

    def paint(self, painter, option, widget):
        painter.drawPolygon(self.poly)

    def containsPolygon(self, p):
        poly2 = self.poly.united(p)
        return len(self.poly) == len(poly2)
